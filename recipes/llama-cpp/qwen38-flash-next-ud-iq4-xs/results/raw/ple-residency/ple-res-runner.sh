#!/usr/bin/env bash
# PLE-residency arm runner for spark-d500.
# Usage: ple-res-runner.sh <label> <ctx> <ub> <lazy> <prompt> <timeout-sec> [evict|noevict]
# Assumes /tmp/qwen38-context-prompts, /tmp/spark_guard.py, /tmp/stream_benchmark.py,
# model at MODEL_GGUF. Everything logged to /tmp/ple-res-<label>-run.log.

set -u

LABEL="$1"; CTX="$2"; UB="$3"; LAZY="$4"; PROMPT="$5"; TMO="$6"; EVICT="${7:-noevict}"

MODEL_DIR=/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS
MODEL_GGUF=$MODEL_DIR/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
SERVER_BIN=/home/djl/llama.cpp-qwen4exp/build/bin/llama-server
PROMPT_DIR=/tmp/qwen38-context-prompts
RUNLOG=/tmp/ple-res-"$LABEL"-run.log
PORT=28900

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

log "=== arm $LABEL ctx=$CTX ub=$UB lazy=$LAZY prompt=$PROMPT timeout=$TMO evict=$EVICT ==="

# --- cache eviction (forces true-cold load) ---
if [ "$EVICT" = evict ]; then
    log "evicting page cache (reading 120G of DeepSeek weights)"
    python3 - << 'PY'
import glob
paths = sorted(glob.glob("/home/djl/models/DeepSeek-V4-Flash-0731/model-*.safetensors"))
n = 0
buf = memoryview(bytearray(16 * 1024 * 1024))
for p in paths:
    with open(p, "rb") as f:
        while True:
            k = f.readinto(buf)
            if not k:
                break
            n += k
    if n >= 120 * (1 << 30):
        break
print("read_bytes", n)
PY
    free -g | tee -a "$RUNLOG"
fi

# --- preconditions: no server, enough RAM, swap zero growth baseline ---
if pgrep -x llama-server >/dev/null; then
    log "ERROR llama-server already running"; exit 10
fi
AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
if [ "$AVAIL" -lt 80 ]; then
    log "ERROR MemAvailable ${AVAIL}G < 80G"; exit 11
fi
log "MemAvailable ${AVAIL}G OK"

# --- start samplers ---
vmstat 1 > /tmp/ple-res-"$LABEL"-vmstat.log 2>&1 &
echo $! > /tmp/ple-res-"$LABEL"-vmstat.pid
iostat -dx 1 nvme0n1 > /tmp/ple-res-"$LABEL"-iostat.log 2>&1 &
echo $! > /tmp/ple-res-"$LABEL"-iostat.pid
grep -E '^(pgpgin|pgpgout|pgmajfault|pgfault) ' /proc/vmstat > /tmp/ple-res-"$LABEL"-vmstat-before.txt
log "samplers started (vmstat pid $(cat /tmp/ple-res-"$LABEL"-vmstat.pid), iostat pid $(cat /tmp/ple-res-"$LABEL"-iostat.pid))"

# --- launch server under guard (nohup wrapper; guard PID is what we SIGTERM) ---
LOAD_T0=$(date +%s)
nohup python3 /tmp/spark_guard.py \
  --min-start-mem-gib 80 --soft-stop-mem-gib 36 --hard-kill-mem-gib 28 \
  --max-swap-growth-gib 1 --soft-stop-grace-seconds 5 --sample-interval-seconds 1 \
  --log-path /tmp/ple-res-"$LABEL"-guard.jsonl -- \
  env LD_LIBRARY_PATH=/home/djl/llama.cpp-qwen4exp/build/bin \
  "$SERVER_BIN" \
  -m "$MODEL_GGUF" \
  --alias qwen38-ud-iq4-xs --host 127.0.0.1 --port $PORT \
  -c "$CTX" -np 1 -b 2048 -ub "$UB" -t 12 -fa on -lm mmap \
  --tensor-read-lazy "$LAZY" -ot per_layer_token_embd=CPU -ngl all -fit off \
  --metrics --offline --log-colors off \
  > /tmp/ple-res-"$LABEL"-server.log 2>&1 &
GUARD_PID=$!
echo "$GUARD_PID" > /tmp/ple-res-"$LABEL"-guard.pid
log "guard pid $GUARD_PID; server launching; load_t0=$LOAD_T0"

# --- health poll (wait up to $TMO + 600 s; MAP_POPULATE can add ~40 s) ---
HEALTH_OK=0
for i in $(seq 1 $((TMO + 600))); do
    H=$(curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    if [ "$H" = '{"status":"ok"}' ]; then HEALTH_OK=1; break; fi
    if ! kill -0 "$GUARD_PID" 2>/dev/null; then
        log "ERROR guard exited before health (check guard jsonl)"; break
    fi
    sleep 1
done
LOAD_T1=$(date +%s)
if [ "$HEALTH_OK" = 1 ]; then
    log "health OK after $((LOAD_T1 - LOAD_T0)) s (load_t0 -> health)"
else
    log "ERROR health never became ok"
fi

# --- snapshot vmstat (load faults isolated) ---
grep -E '^(pgpgin|pgpgout|pgmajfault|pgfault) ' /proc/vmstat > /tmp/ple-res-"$LABEL"-vmstat-after-load.txt
log "vmstat after-load snapshot taken"

# --- run bench (only if healthy) ---
if [ "$HEALTH_OK" = 1 ]; then
    log "running benchmark"
    python3 /tmp/stream_benchmark.py \
      --base-url "http://127.0.0.1:$PORT" --model qwen38-ud-iq4-xs \
      --prompt-file "$PROMPT_DIR/$PROMPT" \
      --max-tokens 64 --warmup-count 0 --repetitions 1 \
      --context-label "$LABEL" --timeout "$TMO" \
      --jsonl-out /tmp/ple-res-"$LABEL".jsonl >> "$RUNLOG" 2>&1
    log "benchmark rc=$?"
else
    log "SKIP benchmark (unhealthy)"
fi

# --- snapshot vmstat (request faults isolated) ---
grep -E '^(pgpgin|pgpgout|pgmajfault|pgfault) ' /proc/vmstat > /tmp/ple-res-"$LABEL"-vmstat-after-req.txt
log "vmstat after-req snapshot taken"

# --- teardown: kill samplers, SIGTERM the guard, wait for server exit ---
VMSTAT_PID=$(cat /tmp/ple-res-"$LABEL"-vmstat.pid 2>/dev/null || true)
IOSTAT_PID=$(cat /tmp/ple-res-"$LABEL"-iostat.pid 2>/dev/null || true)
[ -n "$VMSTAT_PID" ] && kill "$VMSTAT_PID" 2>/dev/null
[ -n "$IOSTAT_PID" ] && kill "$IOSTAT_PID" 2>/dev/null
log "samplers killed"
if [ -n "${GUARD_PID:-}" ] && kill -0 "$GUARD_PID" 2>/dev/null; then
    kill -TERM "$GUARD_PID" 2>/dev/null
    log "SIGTERM sent to guard $GUARD_PID"
fi
for i in $(seq 1 120); do
    pgrep -x llama-server >/dev/null || break
    sleep 1
done
if pgrep -x llama-server >/dev/null; then
    log "WARN llama-server still alive after 120 s"
else
    log "llama-server gone (teardown complete)"
fi

# --- summary for polling ---
log "=== summary $LABEL ==="
if [ -f /tmp/ple-res-"$LABEL".jsonl ]; then
    python3 - "$LABEL" << 'PY' | tee -a "$RUNLOG"
import json, sys
label = sys.argv[1]
rows = []
with open(f"/tmp/ple-res-{label}.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
agg = [r for r in rows if r.get("label") == label or True]
print("jsonl_rows", len(rows))
for r in rows:
    keep = {k: r.get(k) for k in (
        "ttft_seconds", "ttft", "total_seconds", "prompt_tokens",
        "output_sha256", "output_text", "context_label", "label", "error")}
    keep = {k: v for k, v in keep.items() if v is not None}
    if "output_sha256" in keep and isinstance(keep["output_sha256"], str):
        keep["hash8"] = keep.pop("output_sha256")[:8]
    print("row", json.dumps(keep, ensure_ascii=False)[:600])
PY
fi
python3 - "$LABEL" << 'PY' | tee -a "$RUNLOG"
import json, sys
label = sys.argv[1]
events = []
try:
    with open(f"/tmp/ple-res-{label}-guard.jsonl") as f:
        events = [json.loads(l) for l in f if l.strip()]
except FileNotFoundError:
    pass
mems = [e["mem_available_bytes"] for e in events if "mem_available_bytes" in e and e.get("child_pid")]
print("guard_events", len(events))
if mems:
    print("min_mem_available_gib_during", round(min(mems) / (1 << 30), 2))
sw = [e["swap_used_bytes"] for e in events if "swap_used_bytes" in e and e.get("child_pid")]
if sw:
    print("max_swap_growth_gib", round((max(sw) - sw[0]) / (1 << 30), 3))
for e in events:
    if e.get("event") not in ("sample",):
        print("guard", e.get("event"), "child_exit_code", e.get("child_exit_code"))
PY
log "=== end $LABEL ==="
