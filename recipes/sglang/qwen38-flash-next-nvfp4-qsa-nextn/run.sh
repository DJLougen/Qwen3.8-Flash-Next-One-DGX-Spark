#!/usr/bin/env bash
# Recipe: sglang/qwen38-flash-next-nvfp4-qsa-nextn
#
# Qwen3.8-Flash-Next NVFP4 on a single DGX Spark (GB10, 128 GB unified memory).
# The 51.2B-parameter PLE n-gram table (47.7 GiB FP8) is streamed from the model
# directory instead of being held resident, which is what leaves room for the
# 262k context window. See README.md.
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a

: "${MODEL_PATH:?set MODEL_PATH (see env.example)}"
: "${FORK_PATH:?set FORK_PATH (see env.example)}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
IMAGE="${IMAGE:-docker.io/scitrera/dgx-spark-sglang:0.5.17}"
CTX="${CTX:-262144}"
PLE_BACKEND="${PLE_BACKEND:-mmap}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

for d in "$MODEL_PATH" "$FORK_PATH"; do
  [ -d "$d" ] || { echo "not a directory: $d" >&2; exit 1; }
done

engine_args=(
  --rm -i
  --name sglang-qwen38-flash-next
  --device nvidia.com/gpu=all
  --network host
  --shm-size 16g
  -v "$MODEL_PATH:/model:ro"
  -v "$FORK_PATH:/fork:ro"
  -e PYTHONPATH=/fork/python
  -e SGLANG_QWEN4_PLE_NVME_PATH=/model
  -e SGLANG_QWEN4_PLE_NVME_BACKEND="$PLE_BACKEND"
  -e SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=1000
  # flashinfer 0.6.15 vs the 0.6.17 floor the sgl-kernel gate asserts. The QSA,
  # NEXTN and PLE paths use no 0.6.17-only APIs; see README "Version gate".
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
)
[ -n "${SECCOMP_PROFILE:-}" ] && \
  engine_args+=(--security-opt "seccomp=$SECCOMP_PROFILE")

exec "$CONTAINER_ENGINE" run "${engine_args[@]}" "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path /model \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 2048 \
  --max-running-requests "${MAXREQ:-8}" \
  --allow-auto-truncate \
  --context-length "$CTX" \
  --mem-fraction-static 0.84 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host "$HOST" --port "$PORT"
