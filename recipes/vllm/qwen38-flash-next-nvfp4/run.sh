#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 on one DGX Spark (GB10) under vLLM.
# Fails closed: every environment setting below is load-bearing on sm_121 and
# each one was a separate failure before it was a line in this file.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VLLM_VENV="${VLLM_VENV:-/opt/llm/runtime/vllm-venv-fnext}"
VLLM_BIN="${VLLM_BIN:-${VLLM_VENV}/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-/opt/llm/models/qwen38-flash-next-nvfp4}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-flashnext}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SPEC_MTP_K="${SPEC_MTP_K:-2}"
MIN_START_MEM_GIB="${MIN_START_MEM_GIB:-8}"
MIN_SWAP_GIB="${MIN_SWAP_GIB:-48}"
API_KEY="${API_KEY:-}"

EXPECTED_VLLM_VERSION="0.1.dev20073+g8e685d198"

for required in "$VLLM_BIN" "${MODEL_DIR}/config.json"; do
    if [[ ! -e "$required" ]]; then
        printf 'error: required path not found: %s\n' "$required" >&2
        exit 2
    fi
done
for numeric_name in PORT MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
    SPEC_MTP_K MIN_START_MEM_GIB MIN_SWAP_GIB; do
    if ! [[ "${!numeric_name}" =~ ^[0-9]+$ ]]; then
        printf 'error: %s must be a nonnegative integer\n' "$numeric_name" >&2
        exit 2
    fi
done
if (( SPEC_MTP_K == 5 )); then
    printf 'error: num_speculative_tokens=5 hard-fails on this model\n' >&2
    exit 2
fi
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && -z "$API_KEY" ]]; then
    printf 'error: API_KEY is required when binding beyond loopback\n' >&2
    exit 2
fi

# The PLE n-gram table is offloaded to host memory and pages. Without swap the
# offload worker is OOM-killed during startup and vLLM reports only
# "PLE offload worker exited during startup".
swap_gib=$(awk '/^SwapTotal:/ {printf "%d", $2/1048576}' /proc/meminfo)
if (( swap_gib < MIN_SWAP_GIB )); then
    printf 'error: %d GiB swap present, %d GiB required for the PLE table\n' \
        "$swap_gib" "$MIN_SWAP_GIB" >&2
    exit 2
fi
avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
if (( avail_gib < MIN_START_MEM_GIB )); then
    printf 'error: %d GiB available, %d GiB required at launch\n' \
        "$avail_gib" "$MIN_START_MEM_GIB" >&2
    exit 2
fi

version="$("$VLLM_BIN" --version 2>&1 || true)"
if [[ "$version" != *"$EXPECTED_VLLM_VERSION"* && "${ALLOW_UNPINNED_VLLM:-0}" != "1" ]]; then
    printf 'error: expected vLLM %s; got: %s\n' "$EXPECTED_VLLM_VERSION" "$version" >&2
    exit 2
fi

# --- sm_121 environment. Each line is a measured failure, not a preference. ---
# NVFP4 needs sm_121a, not plain sm_121 (cvt.e2m1x2).
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
# DeepGEMM gates on device-capability family 120, which GB10 satisfies, but its
# FP8 blockwise kernel faults here with "CUDA error: unspecified launch failure".
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
# FP8 GDN projections deterministically hang the engine at c~32 with the default
# CUDA kernel: no error, requests simply stall.
export VLLM_GDN_DECODE_KERNEL="${VLLM_GDN_DECODE_KERNEL:-triton}"
# The 51.2 B-parameter PLE table does not fit beside the weights.
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

server_args=(
    serve "$MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST" --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-chunked-prefill
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    # PLE offload is spawned only by the multiproc executor. vLLM picks the
    # uniproc executor by default at TP=1, and the GPU side then waits forever
    # on a worker that was never started (vllm#53960).
    --distributed-executor-backend mp
    --reasoning-parser qwen3
    # Without both flags every request carrying `tools` returns HTTP 400.
    # `--tool-call-parser qwen3` is NOT valid: the tool side registers only as
    # qwen3_coder / qwen3_xml, two names for the same class.
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
)
if (( SPEC_MTP_K > 0 )); then
    server_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_MTP_K}}")
fi
if [[ -n "$API_KEY" ]]; then
    server_args+=(--api-key "$API_KEY")
fi

exec "$VLLM_BIN" "${server_args[@]}"
