# Graph-reuse port (0xBakeer/qwen38-flash-next-spark)

## What we tested

A CUDA-graph-reuse patch from
[`0xBakeer/qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark)
commit `4c6fc3af429bff5c472511cf965751eac6b7caf2` (MIT, Copyright (c) 2026
0xBakeer) was ported onto the Qwen4Exp llama.cpp tree to test whether it
retained its graph-reuse speedup on the current QSA layout.

## What happened

- Reached `graphs reused = 127` during warmup.
- Segfaulted on the second request.
- Kernel log recorded NVIDIA `NV_ERR_NO_MEMORY` at the time of the experiment.

## Why it is not shipped

The patch does not apply cleanly to the current QSA layout and was not stable
past warmup. It was removed and is not part of any `run.sh`. Its methodology
(the varied task-shape suite and separate labeling for prompt cache,
speculative cache, and short-prompt context) informed this catalog's benchmark
hygiene, but no third-party code ships here.
