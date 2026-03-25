# ML App — Design Document

This document explains the key engineering decisions made in this project for the benefit of engineers maintaining or extending it.

---

## Changes from Base Repo

The original repo provided a working Django app serving a SqueezeNet ONNX classifier behind a basic ECS + ALB deployment on Fargate. The following was added:

| Area | Change |
|---|---|
| **Inference** | Model warm-start via `AppConfig.ready()`; concurrency semaphore around `session.run()` |
| **Caching** | Distributed inference cache backed by ElastiCache Redis; `cache_hit` field on `InferenceResult` |
| **Infrastructure** | ElastiCache Terraform module (dev/staging/prod); `MAX_CONCURRENT_INFERENCES` + `REDIS_URL` env vars in task definitions |
| **HTTPS** | ACM certificate per environment; ALB HTTPS listener + HTTP→HTTPS redirect; port 443 on ALB security group |
| **CI/CD** | `test` job added before `build`; `build` gated on passing tests; `pytest` + `pytest-django` dev dependencies |
| **Tests** | `tests/` from scratch — unit tests for cache serialization, API smoke tests via Django test client |
| **Security** | `CSRF_TRUSTED_ORIGINS` for all three environment domains |
| **Docs** | `docs/inference-cache-key.md` cache key trade-off analysis; this design document |
| **CloudFormation** | Added `elasticache:*`, `cloudwatch:PutDashboard`, `acm:*`, `route53:*` to `TerraformDeployerPolicy` |

---

## Table of Contents

1. [Inference Optimizations](#1-inference-optimizations)
2. [Distributed Inference Cache](#2-distributed-inference-cache)
3. [Infrastructure](#3-infrastructure)
4. [HTTPS & TLS](#4-https--tls)
5. [CI/CD Pipeline](#5-cicd-pipeline)
6. [Observability](#6-observability)

---

## 1. Inference Optimizations

### Warm-Start (`inference/apps.py`)

The ONNX model is loaded eagerly in `InferenceConfig.ready()` rather than on the first HTTP request.

**Why:** Without this, the first request after every deploy absorbs ~1–2s of model load time. ECS health checks wait for the task to become healthy before the ALB routes traffic, so the startup cost is invisible to users.

**Trade-off:** Container startup is ~1–2s slower. This is acceptable — ECS won't mark the task healthy until it's ready regardless.

### Concurrency Control (`inference/services.py`)

A `threading.Semaphore` caps the number of simultaneous `session.run()` calls. The limit is configured per environment via `MAX_CONCURRENT_INFERENCES`:

| Environment | vCPU | Limit |
|---|---|---|
| dev | 0.25 | 1 |
| staging | 0.25 | 2 |
| prod | 0.5 | 4 |

**Why:** Without a cap, N concurrent requests all compete for CPU simultaneously. Under load this causes all requests to degrade together rather than a predictable subset queuing. The semaphore provides backpressure without dropping requests.

**`model_ms` includes wait time intentionally.** This gives a true picture of how long a request waited for the model, surfacing queueing pressure in the `ModelDurationMs` CloudWatch metric without a separate metric.

**Trade-off:** Requests above the cap block until the semaphore is released. Under extreme load this causes timeouts rather than graceful degradation. To raise throughput, increase the ECS task CPU allocation and bump `MAX_CONCURRENT_INFERENCES` via the task definition environment variable.

---

## 2. Distributed Inference Cache

### Architecture

Cache-aside pattern implemented directly in `PretrainedImageClassifier.classify()`, backed by ElastiCache Redis. The Django cache framework was considered but rejected — it doesn't support surfacing `cache_hit` in the response, and inference caching is domain logic that belongs in the service layer.

### Cache Key

SHA256 of the **raw uploaded image bytes**, before any format normalization (EXIF transpose, RGB conversion, resize).

**Why pre-normalization:** Hashing raw bytes is cheap (no decode required) and deterministic. The correct choice depends on traffic shape — if a significant share of requests are semantically identical images with differing EXIF metadata, post-normalization hashing would improve hit rate at the cost of a more complex pipeline.

See `docs/inference-cache-key.md` for a full trade-off comparison.

### Cache Value

`InferenceResult` serialized to JSON. Two invariants are enforced:

- `cache_hit` is **never persisted as `True`** — avoids storing a stale flag in Redis
- `cache_hit` is **always `True` on deserialization** — so callers can distinguish cache hits from live inference

### `cache_hit` Field

`InferenceResult.cache_hit: bool = False` allows downstream consumers (dashboards, metrics) to filter out cache hits from latency measurements. Without this, cache hits (~0ms model time) would skew p50/p95 `ModelDurationMs` downward, masking real inference performance.

**Cache hit metrics are not yet emitted** — this is a known TODO. When added, filter `ModelDurationMs` to `cache_hit=False` to get clean inference latency.

### TTL

7 days (`INFERENCE_CACHE_TTL_SECONDS`, configurable via env var). Image classifications are deterministic and don't go stale, so a long TTL is correct. Adjust down if cache storage cost becomes a concern.

### Graceful Degradation

`_cache` is `None` when `REDIS_URL` is unset. All cache logic is gated on `if _cache is not None`, so the app runs normally without Redis (useful for local development).

---

## 3. Infrastructure

### Terraform Module Structure

```
infra/terraform/
├── modules/
│   ├── networking/     VPC, subnets, ALB, security groups, ACM cert
│   ├── ecs/            ECS cluster, service, task definition (bootstrap)
│   ├── elasticache/    Redis replication group, subnet group, security group
│   ├── monitoring/     CloudWatch dashboard
│   └── iam/            GitHub OIDC, roles
└── envs/
    ├── shared/         ECR + IAM (shared across all environments)
    ├── dev/            10.0.0.0/16, 0.25 vCPU, FARGATE_SPOT
    ├── staging/        10.1.0.0/16, 0.25 vCPU, FARGATE_SPOT
    └── prod/           10.2.0.0/16, 0.5 vCPU, FARGATE, 2 tasks
```

### ECS Task Definition Split

Terraform bootstraps the initial task definition. After the first deploy, **CI owns task definition revisions** via `aws-actions/amazon-ecs-render-task-definition`. Terraform uses `lifecycle { ignore_changes = [task_definition] }` on the ECS service to avoid conflicts.

The static task definition JSON files in `infra/task-definitions/` are the source of truth for environment variables (including `REDIS_URL`, `MAX_CONCURRENT_INFERENCES`). Update these files to change runtime configuration — CI will register a new revision on the next deploy.

### ElastiCache

| Environment | Node type | Nodes | Failover |
|---|---|---|---|
| dev | cache.t4g.micro | 1 | no |
| staging | cache.t4g.micro | 1 | no |
| prod | cache.t4g.small | 2 | yes (multi-AZ) |

Redis is placed in private subnets. The security group allows inbound 6379 only from the ECS tasks security group — no public access.

At-rest encryption is enabled on all clusters. Transit encryption is not enabled (internal VPC traffic only) to avoid certificate management overhead.

---

## 4. HTTPS & TLS

HTTPS is terminated at the ALB. The HTTP listener (port 80) issues a 301 redirect to HTTPS.

### Certificate Provisioning

ACM certificates use DNS validation. The `diyer.us` zone is managed in **Cloudflare**, not Route 53, so validation CNAMEs must be added manually. After `terraform apply` creates the certificate, the `cert_validation_records` output provides the exact records to add. Terraform waits for ACM to confirm issuance before attaching the certificate to the HTTPS listener.

DNS records in Cloudflare:
- Validation CNAMEs: **DNS only** (grey cloud) — required for ACM to verify
- ALB CNAME (`ml-app-{env}.diyer.us` → ALB hostname): **DNS only** — Cloudflare proxying is incompatible with ACM-terminated TLS on the ALB

TLS policy: `ELBSecurityPolicy-TLS13-1-2-2021-06` (TLS 1.3 preferred, TLS 1.2 minimum).

---

## 5. CI/CD Pipeline

### Flow

```
push to main
  └── test → build → deploy-dev (automatic)

push to staging branch
  └── test → build → deploy-staging (manual approval)

GitHub release published
  └── test → build → deploy-prod (manual approval)
```

### Test Job

Tests run before the Docker build. A failing test blocks the entire pipeline — no broken image is ever pushed to ECR.

The test suite avoids downloading the ONNX model in CI by mocking `get_pretrained_image_classifier` and `record_inference_metrics`. This keeps CI fast and removes the HuggingFace dependency from the test environment.

### Authentication

GitHub Actions authenticates to AWS via OIDC — no stored access keys. Separate IAM roles scope each job to its minimum required permissions (ECR push for build, ECS deploy for each environment).

### Promoting to Staging

```bash
git push origin main:staging
```

Approve the deployment in the GitHub Actions UI under the `staging` environment.

### Releasing to Production

Create and push a tag, then publish a GitHub release:

```bash
git tag v1.x.x && git push origin v1.x.x
```

---

## 6. Observability

### CloudWatch Dashboard (`ml-app-{env}`)

Five rows covering the full request path:

| Row | Metrics |
|---|---|
| Inference timing | `InferenceDurationMs`, `ModelDurationMs`, `PreprocessingDurationMs` — p50/p95/avg |
| Model quality | `TopScore` (avg/min), `ImagePixels` (avg/max) |
| ALB traffic | Request count, target response time p50/p95, 4xx/5xx error counts |
| ECS resources | CPU utilisation, memory utilisation |

### EMF Metrics

Metrics are emitted via CloudWatch Embedded Metrics Format (EMF) after each inference request. EMF writes structured JSON to stdout; the CloudWatch agent picks it up from the ECS log stream. This means metrics are zero-latency from the application's perspective and require no separate metrics SDK calls.

All metrics carry an `Environment` dimension, so dev/staging/prod are independently filterable.

### Known Gaps

- **Cache hit rate** is not yet tracked. Add a `CacheHit` metric (0/1) in `record_inference_metrics`, filtering on `result.cache_hit`. This will allow a hit-rate widget on the dashboard and alerts on cache degradation.
- **Redis connection errors** are silently swallowed (cache miss path is taken). Consider logging a warning or emitting a metric when `_cache.get()` raises an exception.
