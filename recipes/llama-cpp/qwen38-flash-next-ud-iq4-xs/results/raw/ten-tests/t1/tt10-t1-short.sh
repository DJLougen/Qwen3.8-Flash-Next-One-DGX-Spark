#!/bin/bash
# tt10 T1/T2 decode-regression control: warm short-prompt bench on a warm server
set -u
PORT=28900
LIB_DIR="$1"
LABEL="$2"
export LD_LIBRARY_PATH="$LIB_DIR"
nohup python3 /tmp/spark_guard.py \
  --min-start-mem-gib 80 --soft-stop-mem-gib 36 --hard-kill-mem-gib 28 \
  --max-swap-growth-gib 1 --soft-stop-grace-seconds 5 --sample-interval-seconds 1 \
  --log-path /tmp/tt10-"$LABEL"-guard.jsonl \
  -- \
  env LD_LIBRARY_PATH="$LIB_DIR" ${SERVER_BIN:-/home/djl/llama.cpp-tt10-t1/build/bin/llama-server} \
  -m /home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --alias qwen38-ud-iq4-xs --host 127.0.0.1 --port $PORT \
  -c 4096 -np 1 -b 2048 -ub 512 -t 12 -fa on -lm mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU -ngl all -fit off --metrics --offline --log-colors off \
  > /tmp/tt10-"$LABEL"-server.log 2>&1 &
GPID=$!
for i in $(seq 1 200); do
  curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PORT/health" && break
  sleep 2
done
python3 /tmp/stream_benchmark.py --base-url "http://127.0.0.1:$PORT" --model qwen38-ud-iq4-xs \
  --prompt-file /tmp/short.txt --warmup-count 1 --repetitions 3 --max-tokens 128 \
  --timeout 300 --context-label "$LABEL" --jsonl-out /tmp/tt10-"$LABEL".jsonl 2>&1 | tee /tmp/tt10-"$LABEL"-bench.log
kill -TERM $GPID 2>/dev/null
pkill -TERM -x llama-server 2>/dev/null
sleep 6
pkill -KILL -x llama-server 2>/dev/null
echo "done $LABEL"
