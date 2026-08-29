# Results — external comparisons and port attempts

Third-party reference work and experiments we ran against it, kept separate from
the per-recipe `results/` so each catalog headline stays backed by its own
measured run.

| Topic | Source | What | Why |
|---|---|---|---|
| [Experiment Ledger](experiment-ledger.md) | Internal catalog | Master matrix of proven, rejected, parked, and untested exploration axes | Keeps track of tested results, root causes for failures, and untested optimization vectors. |
| [NVFP4 SGLang](nvfp4-sglang-comparison.md) | [r0b0tlab](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121) | Published-number comparison vs the 2× GB10 NVFP4 + MTP SGLang stack (not rerun here) | Same model and hardware family, but different quantization, GPU count, runtime, and decoder, so figures do not transfer. |
| [Graph-reuse port](graph-reuse-port.md) | [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) | Ported a CUDA-graph-reuse patch; reached `graphs reused = 127`, then segfaulted | Recorded as tested-and-rejected so the patch is not retried or shipped. |
