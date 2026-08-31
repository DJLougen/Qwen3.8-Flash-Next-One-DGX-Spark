#!/bin/bash
# tt10 T8: reference binary, np2 mixed-lane server (guard floors 100/45/38 per run.sh PARALLEL=2)
set -u
MODEL=/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PORT=28900
export LD_LIBRARY_PATH=/home/djl/llama.cpp-qwen4exp/build/bin
nohup python3 /tmp/spark_guard.py \
  --min-start-mem-gib 100 --soft-stop-mem-gib 45 --hard-kill-mem-gib 38 \
  --max-swap-growth-gib 1 --soft-stop-grace-seconds 5 --sample-interval-seconds 1 \
  --log-path /tmp/tt10-t8-guard.jsonl \
  -- \
  env LD_LIBRARY_PATH=/home/djl/llama.cpp-qwen4exp/build/bin \
  /home/djl/llama.cpp-qwen4exp/build/bin/llama-server \
  -m "$MODEL" --alias qwen38-ud-iq4-xs --host 127.0.0.1 --port $PORT \
  -c 8192 -np 2 --no-cache-prompt \
  -b 2048 -ub 512 -t 12 -fa on -lm mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU -ngl all -fit off \
  --metrics --offline --log-colors off \
  > /tmp/tt10-t8-server.log 2>&1 &
echo "guard pid $!"
