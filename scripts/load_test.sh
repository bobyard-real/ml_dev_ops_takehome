#!/usr/bin/env bash
# Basic load test for the inference service.
# Usage: ./scripts/load_test.sh [BASE_URL] [CONCURRENCY] [TOTAL_REQUESTS]
#
# Uses the bundled test image (test_data/dog.avif).

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
CONCURRENCY="${2:-5}"
TOTAL="${3:-50}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_PATH="$SCRIPT_DIR/../test_data/dog.avif"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: test image not found at $IMAGE_PATH" >&2
  exit 1
fi

echo "=== Load Test ==="
echo "Target:      $BASE_URL"
echo "Concurrency: $CONCURRENCY"
echo "Requests:    $TOTAL"
echo ""

# --- Health check ---
echo "Checking service health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health/")
if [ "$HTTP_CODE" != "200" ]; then
  echo "Service not healthy (HTTP $HTTP_CODE). Aborting." >&2
  exit 1
fi
echo "Service is healthy."
echo ""

# --- Run load test ---
TMPDIR_RESULTS=$(mktemp -d)
trap "rm -rf $TMPDIR_RESULTS" EXIT

# Write a small worker script to avoid xargs line-length issues
WORKER="$TMPDIR_RESULTS/_worker.sh"
cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
ID="$1"; URL="$2"; IMG="$3"; OUT="$4"
start=$(python3 -c "import time;print(time.monotonic())")
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -F "image=@$IMG" "$URL/api/infer/")
end=$(python3 -c "import time;print(time.monotonic())")
lat=$(python3 -c "print(round($end-$start,4))")
echo "$code $lat" > "$OUT/$ID.txt"
WORKER_EOF
chmod +x "$WORKER"

echo "Running $TOTAL requests ($CONCURRENCY concurrent)..."
echo ""

start_time=$(python3 -c "import time; print(time.monotonic())")

seq 1 "$TOTAL" | xargs -P "$CONCURRENCY" -I {} "$WORKER" {} "$BASE_URL" "$IMAGE_PATH" "$TMPDIR_RESULTS"

end_time=$(python3 -c "import time; print(time.monotonic())")

# Collect and analyze results
python3 - "$TMPDIR_RESULTS" "$TOTAL" "$start_time" "$end_time" <<'PYEOF'
import sys, os, statistics, glob

results_dir, total_str, t_start, t_end = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
total = int(total_str)
elapsed = round(t_end - t_start, 2)

latencies = []
success = errors = 0
for f in sorted(glob.glob(os.path.join(results_dir, "*.txt"))):
    if f.endswith("_worker.sh"):
        continue
    with open(f) as fh:
        parts = fh.read().strip().split()
        code, lat = parts[0], float(parts[1])
        if code == "200":
            success += 1
        else:
            errors += 1
        latencies.append(lat)

latencies.sort()
n = len(latencies)

print("=== Results ===")
print(f"Total time:  {elapsed}s")
print(f"Successful:  {success} / {total}")
print(f"Errors:      {errors}")
print(f"RPS:         {round(total / elapsed, 1)}")
print(f"Latency min: {min(latencies):.3f}s")
print(f"Latency avg: {statistics.mean(latencies):.3f}s")
print(f"Latency p50: {statistics.median(latencies):.3f}s")
print(f"Latency p95: {latencies[int(n * 0.95)]:.3f}s")
print(f"Latency p99: {latencies[int(n * 0.99)]:.3f}s")
print(f"Latency max: {max(latencies):.3f}s")
PYEOF

echo ""
echo "--- Metrics snapshot ---"
curl -s "$BASE_URL/metrics/" | grep -E "^inference_" | head -20
