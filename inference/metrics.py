from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "inference_request_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "End-to-end request latency",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

INFERENCE_LATENCY = Histogram(
    "inference_model_duration_seconds",
    "Model session.run() latency only",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

CACHE_HITS = Counter("inference_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("inference_cache_misses_total", "Cache misses")

CACHE_SIZE = Gauge("inference_cache_size", "Current number of cached results")

MODEL_LOADED = Gauge(
    "inference_model_loaded",
    "Whether the model is loaded and warm (1=ready, 0=not ready)",
)
