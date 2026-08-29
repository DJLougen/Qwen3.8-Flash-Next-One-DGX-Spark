#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="/tmp/ttft-gap-results"
mkdir -p "$LOG_DIR"

UNPATCHED="/home/djl/llama.cpp-qwen4exp/build/bin/llama-server.unpatched-250b61446"
KMTP="/home/djl/llama.cpp-qwen4exp-kmtp/build/bin/llama-server"
MODEL="/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MTP_DRAFT="/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf"
GUARD="/tmp/spark_guard.py"
BENCH="/tmp/stream_benchmark.py"
SHORT="/tmp/short.txt"
REPRO="/tmp/reproduce-module.txt"
PROMPTS="/tmp/qwen38-context-prompts"
GEN="/tmp/generate_context_prompts.py"
LD_PATH="/home/djl/llama.cpp-qwen4exp/build/bin"
MODEL_ALIAS="qwen38-ud-iq4-xs"

kill_servers() {
  local pids
  pids=$(pgrep -f 'llama.cpp-qwen4exp.*llama-server' || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 3
    kill -9 $pids 2>/dev/null || true
  fi
  pids=$(pgrep -f 'spark_guard.py.*llama-server' || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 2
  fi
  sleep 8
}

wait_mem() {
  local avail
  avail=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
  echo "MemAvailable: ${avail} GiB"
  if (( avail < 80 )); then
    echo "ERROR: MemAvailable ${avail} GiB < 80 GiB floor" >&2
    exit 1
  fi
}

wait_ready() {
  local port="$1" log="$2" binary_pattern="$3"
  for i in $(seq 1 60); do
    if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${port}/health" 2>/dev/null; then
      echo "ready after ${i}x10s"
      return 0
    fi
    if ! pgrep -f "$binary_pattern" >/dev/null; then
      echo "server died during load" >&2
      tail -20 "$log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "timeout waiting for health" >&2
  return 1
}

launch_server() {
  local tag="$1" port="$2" binary="$3" ctx="$4" batch="$5" ubatch="$6"
  shift 6
  local log="$LOG_DIR/${tag}-server.log"
  local guard_log="$LOG_DIR/${tag}-guard.jsonl"
  local bin_pattern
  bin_pattern=$(echo "$binary" | sed 's/[.[\*^$()+?{|]/\\&/g')
  kill_servers
  wait_mem
  python3 "$GUARD" \
    --min-start-mem-gib 80 --soft-stop-mem-gib 36 --hard-kill-mem-gib 28 \
    --max-swap-growth-gib 1 --soft-stop-grace-seconds 5 --sample-interval-seconds 1 \
    --log-path "$guard_log" -- \
    env LD_LIBRARY_PATH="$LD_PATH" "$binary" \
      -m "$MODEL" --alias "$MODEL_ALIAS" --host 127.0.0.1 --port "$port" \
      -c "$ctx" -np 1 -b "$batch" -ub "$ubatch" -t 12 \
      -fa on -lm mmap --tensor-read-lazy on -ot per_layer_token_embd=CPU \
      -ngl all -fit off --metrics --offline --log-colors off \
      "$@" >"$log" 2>&1 &
  echo "waiting for server $tag on port $port"
  wait_ready "$port" "$log" "$bin_pattern" || return 1
  echo "$log"
}

run_bench() {
  local port="$1" prompt="$2" label="$3" max_tokens="$4" warmup="$5" reps="$6" timeout="$7" jsonl="$8"
  shift 8
  local out="$LOG_DIR/${label}-bench.txt"
  local args=(
    --base-url "http://127.0.0.1:${port}"
    --model "$MODEL_ALIAS"
    --prompt-file "$prompt"
    --max-tokens "$max_tokens"
    --context-label "$label"
    --warmup-count "$warmup"
    --repetitions "$reps"
    --timeout "$timeout"
    --jsonl-out "$jsonl"
  )
  "$@"  # optional variation args passed as remaining - hack, use direct call below
}

echo "=== STEP 2: decode regression ===" | tee "$LOG_DIR/summary.txt"

for cfg in "decode-b512:512:128" "decode-b2048:2048:512"; do
  tag="${cfg%%:*}"
  rest="${cfg#*:}"
  batch="${rest%%:*}"
  ubatch="${rest##*:}"
  port=$((18000 + RANDOM % 1000))
  log=$(launch_server "$tag" "$port" "$UNPATCHED" 4096 "$batch" "$ubatch") || exit 1
  out="$LOG_DIR/${tag}-bench.txt"
  python3 "$BENCH" --base-url "http://127.0.0.1:${port}" --model "$MODEL_ALIAS" \
    --prompt-file "$SHORT" --max-tokens 128 --context-label "$tag" \
    --warmup-count 1 --repetitions 5 --timeout 300 \
    --jsonl-out "$LOG_DIR/${tag}.jsonl" | tee "$out"
  grep '^aggregate:' "$out" | tee -a "$LOG_DIR/summary.txt"
  kill_servers
done

echo "=== generate ctx16384 if missing ===" | tee -a "$LOG_DIR/summary.txt"
if [[ ! -f "$PROMPTS/ctx16384.txt" ]]; then
  port=$((19000 + RANDOM % 500))
  log=$(launch_server "gen16k" "$port" "$UNPATCHED" 16384 2048 512) || exit 1
  python3 "$GEN" --base-url "http://127.0.0.1:${port}" \
    --output-dir "$PROMPTS" --targets 16384
  kill_servers
fi

depths=(4096 16384 32768 65536)
declare -A depth_files=(
  [4096]="$PROMPTS/ctx4096.txt"
  [16384]="$PROMPTS/ctx16384.txt"
  [32768]="$PROMPTS/ctx32768.txt"
  [65536]="$PROMPTS/ctx65536.txt"
)

run_depth_curve() {
  local tree="$1" binary="$2"
  shift 2
  local extra_args=("$@")
  echo "=== depth curve: $tree ===" | tee -a "$LOG_DIR/summary.txt"
  for depth in "${depths[@]}"; do
    local tag="${tree}-depth${depth}"
    local port=$((20000 + depth % 5000))
    local ctx=$((depth + 256))
    local timeout=600
    (( depth >= 65536 )) && timeout=900
    log=$(launch_server "$tag" "$port" "$binary" "$ctx" 2048 512 "${extra_args[@]}") || exit 1
    out="$LOG_DIR/${tag}-bench.txt"
    python3 "$BENCH" --base-url "http://127.0.0.1:${port}" --model "$MODEL_ALIAS" \
      --prompt-file "${depth_files[$depth]}" --max-tokens 64 --context-label "$tag" \
      --warmup-count 0 --repetitions 1 --timeout "$timeout" \
      --jsonl-out "$LOG_DIR/${tag}.jsonl" | tee "$out"
    grep '^aggregate:' "$out" | tee -a "$LOG_DIR/summary.txt"
    grep -E 'prompt eval time' "$log" | tail -1 | tee -a "$LOG_DIR/${tag}-prefill.txt"
    kill_servers
  done
}

run_depth_curve "unpatched" "$UNPATCHED"

run_depth_curve "kmtp" "$KMTP" \
  --spec-type draft-mtp \
  -md "$MTP_DRAFT" -ngld 99 --spec-draft-n-max 3

echo "=== ngram combo (copy-heavy) ===" | tee -a "$LOG_DIR/summary.txt"
port=$((21000 + RANDOM % 500))
log=$(launch_server "ngram-combo" "$port" "$KMTP" 4096 2048 512 \
  --spec-type draft-mtp,ngram-mod \
  -md "$MTP_DRAFT" -ngld 99 --spec-draft-n-max 3) || exit 1
out="$LOG_DIR/ngram-combo-bench.txt"
python3 "$BENCH" --base-url "http://127.0.0.1:${port}" --model "$MODEL_ALIAS" \
  --prompt-file "$REPRO" --max-tokens 128 --context-label "ngram-combo" \
  --warmup-count 1 --repetitions 3 --timeout 300 \
  --variation-placeholder '@' \
  --jsonl-out "$LOG_DIR/ngram-combo.jsonl" | tee "$out"
grep '^aggregate:' "$out" | tee -a "$LOG_DIR/summary.txt"
grep -iE 'draft acceptance|ngram' "$log" | tail -10 | tee -a "$LOG_DIR/summary.txt" || true
kill_servers

echo "DONE" | tee -a "$LOG_DIR/summary.txt"
