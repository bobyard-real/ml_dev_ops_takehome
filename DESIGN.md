# Design Document: ML Inference Service

This document explains the engineering decisions behind each change to the inference service, aimed at downstream engineers who will operate, extend, or review this system.

---

## 1. Inference Improvements

### 1a. Result Caching (LRU, xxhash-keyed)

**What:** An in-process LRU cache in `PretrainedImageClassifier` that maps `xxh64(raw image bytes)` to the `InferenceResult`. On a cache hit, we skip image decoding, preprocessing, and model inference entirely.

**Why xxhash instead of MD5:**
xxhash (xxh64) is ~3x faster than MD5 for hashing. Since we're using the hash as a content-addressable cache key (not for security), we don't need cryptographic properties. xxhash has excellent collision resistance for non-adversarial inputs and is widely used in databases and caches (e.g., LZ4, RocksDB).

**Why hash raw bytes, not the normalized tensor:**
Hashing happens *before* any image processing. This means we avoid the cost of opening the image, EXIF transpose, resize, and normalization on cache hits. Collision probability is negligible at our cache scale (256 entries).

**Why `OrderedDict` instead of `functools.lru_cache`:**
`lru_cache` decorates a function and requires hashable arguments. Our input is a file-like `BinaryIO` object, which isn't hashable. `OrderedDict` with manual `move_to_end()` / `popitem(last=False)` gives us LRU semantics with full control over the eviction policy.

**Tradeoff:**
- Memory: each cached `InferenceResult` is tiny (model name + 3 label/score pairs + dimensions), so 256 entries is negligible. The real memory concern would be caching the raw image bytes, which we intentionally don't do.
- Staleness: the cache never invalidates. If the model were swapped at runtime, cached results would be stale. This is acceptable because the model is loaded once at startup and lives for the process lifetime.
- Cache size is tunable via `INFERENCE_CACHE_MAX_SIZE` env var without code changes.

**Measured impact:** ~640x speedup on cache hit (82ms -> 0.1ms).

### 1b. Preprocessing Pipeline Optimization

**What:** Three changes to reduce CPU work and memory allocations on cache-miss requests:

1. **Fused normalization constants:** Pre-compute `1/(255*std)` and `mean/std` at init time. The original code did `array / 255.0`, then `(array - mean) / std`, then `.astype(float32)` — three array-wide operations plus a copy. The fused version does `array *= scale` then `array -= shift` — two in-place operations, no copy.

2. **BytesIO instead of re-seeking:** On a cache miss, the original code called `uploaded_image.seek(0)` to re-read from the Django `UploadedFile`. We already have the bytes in memory (read for hashing), so we wrap them in `BytesIO` instead of seeking the original file object. This avoids a second read through Django's file handling layer.

3. **`numpy.ascontiguousarray` on transpose:** After HWC→CHW transpose, the array is non-contiguous in memory. `ascontiguousarray` produces a single contiguous buffer that ORT can consume without internal copies.

**Load test results (local, 50 requests, 5 concurrent, same image):**

| Metric | Before | After | Change |
|---|---|---|---|
| RPS | 45.0 | 52.6 | +17% |
| p50 latency | 31ms | 26ms | -16% |
| p95 latency | 284ms | 216ms | -24% |
| p99 latency | 292ms | 216ms | -26% |

The gains are most visible on cache-miss requests (p95/p99), where the preprocessing pipeline runs end-to-end. Cache-hit requests (p50) also benefit from the faster xxhash.

### 1c. ORT Session Tuning

**What:** Configured `onnxruntime.SessionOptions` with explicit thread counts, memory pattern optimization, graph optimization level, and optimized model serialization.

**Key settings and rationale:**

| Setting | Value | Why |
|---|---|---|
| `intra_op_num_threads` | `0` (auto) | ORT auto-detects core count. Avoids hardcoding a value that breaks on different hardware. |
| `inter_op_num_threads` | `1` | We use `ORT_SEQUENTIAL` execution mode. SqueezeNet's graph is nearly linear — there are very few independent nodes to parallelize across, so extra inter-op threads would sit idle. |
| `execution_mode` | `ORT_SEQUENTIAL` | Nodes execute one at a time, concentrating all intra-op threads on each op. Better than `ORT_PARALLEL` for small sequential models. |
| `enable_mem_pattern` | `True` | Lets ORT pre-plan memory allocation patterns based on the first inference, reducing allocation overhead on subsequent runs. |
| `graph_optimization_level` | `ORT_ENABLE_ALL` | Enables all optimization passes: constant folding, node fusion, layout optimization. |
| `optimized_model_filepath` | `*.optimized.onnx` | Serializes the optimized graph to disk. On subsequent process starts, ORT loads the pre-optimized graph directly instead of re-running optimization passes. |

**Tradeoff:**
- The `.optimized.onnx` file may contain hardware-specific optimizations (e.g., NCHW->NCHWc transforms). It should not be copied between different CPU architectures. This is acceptable for containerized deployments where the build environment matches runtime.
- Thread settings are exposed as `ORT_INTRA_OP_THREADS` and `ORT_INTER_OP_THREADS` env vars for tuning in different deployment contexts (e.g., Fargate with 0.5 vCPU vs. a 16-core bare metal box).

### 1d. Model Warm-Start

**What:** Two parts:
1. A dummy inference (`numpy.zeros` input) runs during `PretrainedImageClassifier.__init__()` to trigger lazy memory allocations, JIT compilation, and thread pool initialization.
2. `AppConfig.ready()` in `inference/apps.py` calls `get_pretrained_image_classifier()` at Django startup, so the model is loaded and warmed before the first HTTP request arrives.

**Why this matters:**
Without warm-start, the first real request pays a one-time penalty for memory allocation, thread pool spinup, and potential JIT passes in ORT's execution providers. In a containerized deployment, this directly impacts health check timing — if the first health check arrives before the model is warm, the container may be marked unhealthy and killed before it's ready.

**Tradeoff:**
- Adds ~80ms to server startup time. Acceptable because it shifts latency from the critical path (user requests) to the non-critical path (container boot).
- The eager load in `AppConfig.ready()` means Django management commands (e.g., `migrate`, `collectstatic`) also load the model. This is a minor annoyance in development but has no production impact since those commands aren't run in the serving container.

---

## 2. Observability

### 2a. Prometheus Metrics (`/metrics`)

**What:** `prometheus_client` library exposing metrics on `GET /metrics/` in Prometheus exposition format.

**Metrics exposed:**

| Metric | Type | Purpose |
|---|---|---|
| `inference_request_total{endpoint, status}` | Counter | Request volume by endpoint and HTTP status. Alert on error rate spikes. |
| `inference_request_duration_seconds{endpoint}` | Histogram | End-to-end latency including image decode + inference. Use for SLO tracking. |
| `inference_model_duration_seconds` | Histogram | Just `session.run()` time. Isolates model performance from preprocessing overhead. |
| `inference_cache_hits_total` | Counter | Cache effectiveness. If hit rate is low, cache size may need increasing. |
| `inference_cache_misses_total` | Counter | Paired with hits to compute hit ratio. |
| `inference_cache_size` | Gauge | Current cache occupancy. Alerts if approaching max. |
| `inference_model_loaded` | Gauge | 1 when model is warm, 0 otherwise. Used by the `/ready` endpoint. |

**Histogram bucket choices:**
- Request latency buckets go up to 2.5s because image upload + inference on a large image under load could approach that.
- Model latency buckets top out at 0.5s because `session.run()` alone for SqueezeNet should never exceed that — if it does, something is wrong (thread contention, memory pressure).

**Tradeoff:**
- `prometheus_client` uses a global registry. In multi-process deployments (e.g., gunicorn with prefork workers), each worker has its own counter values. For production, you'd need `prometheus_client.multiprocess` mode or a StatsD-based exporter. For this single-process / low-worker-count deployment, the default registry is fine.
- The `/metrics` endpoint is unauthenticated. In production, restrict access via network policy or middleware.

### 2b. Health and Readiness Endpoints

**`GET /health/`** — Liveness probe. Always returns `200 {"status": "ok"}` if the Django process is up. Used by the ALB health check and ECS task health check to detect crashed processes.

**`GET /ready/`** — Readiness probe. Returns `503` until the model is loaded and warm-started, then `200`. This distinction matters for rolling deployments: the load balancer should not route traffic to a container that's still loading the model.

**Why separate endpoints:**
The ALB health check targets `/health/` (cheap, always passes). The ECS task-level health check also uses `/health/` via a `CMD-SHELL` Python one-liner. If we needed a Kubernetes-style deployment, `/ready/` would be the readiness probe and `/health/` the liveness probe — same pattern applies to ECS with proper target group configuration.

---

## 3. CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/ci.yml`)

**Two jobs, sequential dependency:**

1. **`test`** — Installs dependencies, runs `pytest tests/ -v`. The smoke tests validate that the Django app boots, the home view returns 200, the health endpoint works, and the metrics endpoint returns Prometheus data.
2. **`build`** — Runs only if tests pass. Builds the Docker image tagged with the git SHA. On `main`, also tags as `latest`.

**Why SHA-based tagging:**
Every commit produces a unique, immutable image tag. This enables exact rollback ("deploy SHA abc123") without ambiguity. The `latest` tag on main is a convenience for development, not for production deploys.

**ECR push:**
After tests pass and the image builds, the workflow authenticates to AWS via repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) and pushes the SHA-tagged image to ECR. On `main`, it also pushes the `latest` tag. This means every merge to main produces a deployable artifact in ECR, and any commit SHA can be used for rollback.

**Tradeoff:**
- No integration tests with actual inference (would require model download in CI, adding ~30s). The smoke tests validate the app skeleton; inference correctness is covered by the model's own validation and manual testing.

---

## 4. Container Image (`Dockerfile`)

**Base image:** `python:3.12-slim` — minimal Debian (Trixie) with Python, no unnecessary packages.

**Key decisions:**

- **System deps (`libglib2.0-0`, `libgl1`):** Required by Pillow for image processing. Installed before `pip install` to leverage Docker layer caching. Note: `libgl1-mesa-glx` was removed in Debian Trixie; `libgl1` is the replacement.
- **Model baked into image:** `ensure_model_assets()` runs at build time, so the model ONNX file is in the image. This means containers don't need network access to download the model at startup, eliminating a cold-start failure mode and reducing startup time.
- **Gunicorn with 2 workers:** Django's `runserver` is single-threaded and not production-grade. Gunicorn provides proper request handling. 2 workers is sized for the Fargate task's 512 CPU units (0.5 vCPU) — more workers would cause thread contention.
- **`PYTHONUNBUFFERED=1`:** Ensures logs appear in CloudWatch immediately instead of being buffered.

**Tradeoff:**
- Baking the model into the image means a model update requires a new image build and deploy. For a ~5MB model like SqueezeNet, this is fine. For large models (multi-GB), you'd pull from S3 at startup or use an EFS mount instead.

---

## 5. Terraform / Infrastructure (`terraform/`)

### Architecture

```
Internet -> ALB (port 80) -> ECS Fargate Service -> Container (port 8000)
                                |                        |
                          Auto-scaling              CloudWatch Logs
                        (CPU + request count)
```

### Key Decisions

**Default VPC:** Uses the AWS account's default VPC and subnets. The assignment spec says custom VPC is a bonus item. The default VPC works for a single-service deployment and avoids the complexity of NAT gateways, route tables, and subnet CIDR planning.

**Fargate (not EC2):** No instances to manage, patch, or scale. The task definition specifies exact CPU/memory, and AWS handles placement. For a small inference service, this is the right abstraction level.

**Resource sizing (512 CPU / 1024 MiB):**
- SqueezeNet is ~5MB and inference takes ~5-10ms on CPU. This doesn't need a large instance.
- 1024 MiB gives headroom for the Python runtime, ORT memory arena, and the image processing pipeline.
- If latency under load becomes an issue, scale horizontally (autoscaling handles this) before scaling vertically (CPU/memory).

**ALB, not direct task access:**
Even with a single task, the ALB provides health checking, connection draining during deploys, and a stable DNS name. It's also required for ECS service-managed deployments to work properly with rolling updates.

**Security groups — least privilege:**
- ALB SG: allows inbound HTTP (80) from anywhere, outbound to anywhere.
- ECS SG: allows inbound only from the ALB SG on the container port. No direct internet access to the container.

**Container Insights enabled:** Provides CPU/memory utilization metrics at the ECS level without additional instrumentation. Pairs with the application-level Prometheus metrics for full observability.

**60-second `startPeriod` on ECS health check:** Gives the container time to download dependencies (if any were missed), load the model, and run the warm-start inference before ECS starts checking health. Without this, the container could be killed during model initialization.

### ECR Repository

**What:** Terraform manages the ECR repository (`ml-inference`) with scan-on-push enabled and `force_delete = true` for clean teardown.

**Why Terraform-managed (not manual):** Keeps the image registry lifecycle consistent with the rest of the infrastructure. The ECS task definition references `aws_ecr_repository.main.repository_url` directly, eliminating hardcoded account IDs or registry URLs.

### Remote State (S3 + DynamoDB)

**What:** Terraform state is stored in S3 (`ml-inference-tfstate`) with DynamoDB locking (`ml-inference-tflock`). A separate bootstrap module (`terraform/bootstrap/`) creates these resources.

**Key properties:**
- **S3 versioning enabled:** Every state change is versioned, enabling recovery from bad applies.
- **AES-256 encryption at rest:** State files may contain sensitive outputs.
- **Public access blocked:** All four S3 public access block settings are enabled.
- **DynamoDB PAY_PER_REQUEST:** State locking is infrequent; on-demand billing avoids paying for idle capacity.

**Why a separate bootstrap module:** The S3 bucket and DynamoDB table that *store* state can't be managed *by* the state they store (chicken-and-egg). The bootstrap module uses local state and is run once.

**Multi-environment support:** The backend key defaults to `infra/terraform.tfstate` but can be overridden per environment:
```bash
terraform init -backend-config="key=infra/staging/terraform.tfstate"
terraform apply -var="environment=staging"
```

### Autoscaling

**What:** Application Auto Scaling on the ECS service with two target-tracking policies:

| Policy | Metric | Target | Scale-out cooldown | Scale-in cooldown |
|---|---|---|---|---|
| CPU | `ECSServiceAverageCPUUtilization` | 70% | 60s | 120s |
| Request count | `ALBRequestCountPerTarget` | 100 req/target | 60s | 120s |

**Why two policies:** CPU scaling catches compute-bound load (large images, complex preprocessing). Request count scaling catches I/O-bound load (many small requests that don't spike CPU). Whichever triggers first wins.

**Capacity range:** Min 1, max 4 (configurable via `autoscaling_min` / `autoscaling_max` variables). The asymmetric cooldowns (faster scale-out, slower scale-in) prevent flapping during traffic spikes.

### Rollback Strategy

**Two rollback mechanisms:**

1. **Automatic (ECS circuit breaker):** The ECS service has `deployment_circuit_breaker` enabled with `rollback = true`. If new tasks fail health checks during a rolling deployment, ECS automatically rolls back to the previous task definition. No operator intervention required.

2. **Manual (SHA-pinned deploys):** Every CI push creates a SHA-tagged image in ECR. To roll back to a known-good version:
   ```bash
   terraform apply -var="image_tag=abc123def456..."
   ```
   This updates the ECS task definition to pull a specific image, triggering a new deployment.

**Why both:** The circuit breaker handles the "deploy broke something obvious" case automatically. SHA-pinned rollback handles "deploy introduced a subtle regression discovered later" — you can point to any previous build without rebuilding.

### CloudWatch Dashboard

**What:** A Terraform-managed CloudWatch dashboard (`ml-inference-prod`) with 6 panels:

| Panel | Metrics | Why |
|---|---|---|
| ECS CPU & Memory | `CPUUtilization`, `MemoryUtilization` | Spot resource exhaustion, validate sizing |
| ALB Request Count | `RequestCount` | Traffic volume, correlate with autoscaling |
| ALB Response Time | avg, p95, p99 `TargetResponseTime` | SLO tracking, latency regression detection |
| HTTP Status Codes | 2xx, 4xx, 5xx counts | Error rate monitoring |
| Running Task Count | `RunningTaskCount` via Container Insights | Verify autoscaling behavior |
| Healthy/Unhealthy Targets | `HealthyHostCount`, `UnHealthyHostCount` | Deployment health, detect failing tasks |

**Why CloudWatch over Prometheus + Grafana:** The `/metrics/` endpoint already exposes Prometheus-format application metrics (inference latency, cache hit rate, model status). For infrastructure-level metrics (CPU, memory, request count, response time), CloudWatch is zero-maintenance — no additional containers to deploy or manage. In production, the natural next step would be AWS Managed Prometheus (AMP) to scrape the `/metrics/` endpoint and Grafana for unified dashboards combining both infrastructure and application metrics.

### What's Not Included (and Why)

| Feature | Why skipped | What you'd add |
|---|---|---|
| HTTPS/TLS | Requires a domain and ACM certificate. | ACM cert + HTTPS listener on ALB. |
| Custom VPC | Default VPC is sufficient for single-service deployment. | VPC module with public/private subnets, NAT gateway, and proper CIDR planning. |

---

## Environment Variables Reference

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `INFERENCE_CACHE_MAX_SIZE` | `256` | `services.py` | Max cached inference results |
| `ORT_INTRA_OP_THREADS` | `0` (auto) | `services.py` | Parallelism within individual ORT ops |
| `ORT_INTER_OP_THREADS` | `1` | `services.py` | Parallelism between independent graph nodes |
| `DJANGO_SECRET_KEY` | insecure default | `settings.py` | Django secret key (must set in production) |
| `DJANGO_DEBUG` | `True` | `settings.py` | Debug mode toggle |
| `DJANGO_ALLOWED_HOSTS` | `*` | `settings.py` | Comma-separated allowed hostnames |

## Terraform Variables Reference

| Variable | Default | Purpose |
|---|---|---|
| `aws_region` | `us-east-1` | AWS region for all resources |
| `app_name` | `ml-inference` | Used for resource naming and ECR repo name |
| `environment` | `prod` | Environment label (prod, staging) — affects resource name prefix |
| `image_tag` | `latest` | Docker image tag to deploy (git SHA for pinned rollback, `latest` for current) |
| `container_port` | `8000` | Port the container listens on |
| `cpu` | `512` | Fargate task CPU units (1024 = 1 vCPU) |
| `memory` | `1024` | Fargate task memory in MiB |
| `desired_count` | `1` | Baseline number of ECS tasks |
| `autoscaling_min` | `1` | Minimum tasks (autoscaling floor) |
| `autoscaling_max` | `4` | Maximum tasks (autoscaling ceiling) |
