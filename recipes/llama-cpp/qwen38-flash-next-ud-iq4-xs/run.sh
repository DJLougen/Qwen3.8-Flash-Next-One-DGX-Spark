#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/home/djl/llama.cpp-qwen4exp}"
BIN_DIR="${LLAMA_CPP_ROOT}/build/bin"
LLAMA_SERVER="${LLAMA_SERVER:-${BIN_DIR}/llama-server}"
MODEL_DIR="${MODEL_DIR:-/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS}"
MODEL_SHARD="${MODEL_SHARD:-${MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MODEL_SHARD_2="${MODEL_SHARD_2:-${MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf}"
MODEL_SHARD_3="${MODEL_SHARD_3:-${MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf}"
SPARK_GUARD="${SPARK_GUARD:-${SCRIPT_DIR}/tools/spark_guard.py}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/results/guard}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-12}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
SPEC_TYPE="${SPEC_TYPE:-none}"
API_KEY="${API_KEY:-}"

MIN_START_MEM_GIB="${MIN_START_MEM_GIB:-80}"
SOFT_STOP_MEM_GIB="${SOFT_STOP_MEM_GIB:-36}"
HARD_KILL_MEM_GIB="${HARD_KILL_MEM_GIB:-28}"
MAX_SWAP_GROWTH_GIB="${MAX_SWAP_GROWTH_GIB:-1}"

EXPECTED_LLAMA_COMMIT="250b61446"

for required in "$LLAMA_SERVER" "$MODEL_SHARD" "$MODEL_SHARD_2" "$MODEL_SHARD_3" "$SPARK_GUARD"; do
    if [[ ! -f "$required" ]]; then
        printf 'error: required file not found: %s\n' "$required" >&2
        exit 2
    fi
done
if [[ ! -x "$LLAMA_SERVER" ]]; then
    printf 'error: llama-server is not executable: %s\n' "$LLAMA_SERVER" >&2
    exit 2
fi
for numeric_name in CONTEXT_SIZE PARALLEL THREADS BATCH_SIZE UBATCH_SIZE \
    MIN_START_MEM_GIB SOFT_STOP_MEM_GIB HARD_KILL_MEM_GIB MAX_SWAP_GROWTH_GIB; do
    if ! [[ "${!numeric_name}" =~ ^[0-9]+$ ]]; then
        printf 'error: %s must be a nonnegative integer\n' "$numeric_name" >&2
        exit 2
    fi
done
if (( PARALLEL < 1 || PARALLEL > 2 )); then
    printf 'error: only proven PARALLEL values 1 and 2 are allowed\n' >&2
    exit 2
fi
if ! [[ "$CONTEXT_SIZE" =~ ^[0-9]+$ ]] ||
    (( CONTEXT_SIZE < 512 || CONTEXT_SIZE > 262144 )); then
    printf 'error: CONTEXT_SIZE must be an integer from 512 through 262144\n' >&2
    exit 2
fi
if (( PARALLEL == 2 )); then
    if (( CONTEXT_SIZE != 8192 )); then
        printf 'error: proven PARALLEL=2 configuration requires CONTEXT_SIZE=8192\n' >&2
        exit 2
    fi
    if (( MIN_START_MEM_GIB < 100 || SOFT_STOP_MEM_GIB < 45 || HARD_KILL_MEM_GIB < 38 )); then
        printf 'error: PARALLEL=2 requires guard floors 100/45/38 GiB or stricter\n' >&2
        exit 2
    fi
fi
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && -z "$API_KEY" ]]; then
    printf 'error: API_KEY is required when binding beyond loopback\n' >&2
    exit 2
fi

version="$({ env "LD_LIBRARY_PATH=${BIN_DIR}" "$LLAMA_SERVER" --version; } 2>&1)"
if [[ "$version" != *"$EXPECTED_LLAMA_COMMIT"* && "${ALLOW_UNPINNED_LLAMA:-0}" != "1" ]]; then
    printf 'error: expected llama.cpp commit %s; got: %s\n' \
        "$EXPECTED_LLAMA_COMMIT" "$version" >&2
    exit 2
fi

server_args=(
    -m "$MODEL_SHARD"
    --alias qwen38-ud-iq4-xs
    --host "$HOST"
    --port "$PORT"
    -c "$CONTEXT_SIZE"
    -np "$PARALLEL"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -t "$THREADS"
    -fa on
    -lm mmap
    --tensor-read-lazy on
    -ot per_layer_token_embd=CPU
    -ngl all
    -fit off
    -ctk f16
    -ctv f16
    --metrics
    --no-cache-prompt
    -cram 0
    --no-cache-idle-slots
    --reasoning-preserve
    --offline
    --log-colors off
)

case "$SPEC_TYPE" in
    none) ;;
    ngram-mod) server_args+=(--spec-type ngram-mod) ;;
    *)
        printf 'error: SPEC_TYPE must be none or ngram-mod\n' >&2
        exit 2
        ;;
esac
if [[ -n "$API_KEY" ]]; then
    server_args+=(--api-key "$API_KEY")
fi

mkdir -p "$LOG_DIR"
guard_log="${LOG_DIR}/server-ctx${CONTEXT_SIZE}-$(date -u +%Y%m%dT%H%M%SZ).jsonl"

exec python3 "$SPARK_GUARD" \
    --min-start-mem-gib "$MIN_START_MEM_GIB" \
    --soft-stop-mem-gib "$SOFT_STOP_MEM_GIB" \
    --hard-kill-mem-gib "$HARD_KILL_MEM_GIB" \
    --max-swap-growth-gib "$MAX_SWAP_GROWTH_GIB" \
    --soft-stop-grace-seconds 5 \
    --sample-interval-seconds 1 \
    --log-path "$guard_log" \
    -- env "LD_LIBRARY_PATH=${BIN_DIR}" "$LLAMA_SERVER" "${server_args[@]}"
