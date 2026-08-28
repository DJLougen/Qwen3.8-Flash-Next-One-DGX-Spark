# Comparison: this llama.cpp recipe vs r0b0tlab NVFP4 SGLang

This is **not a bake-off**. The two stacks differ in GPU count, quantization,
runtime, speculative decoding, and measurement protocol. The table exists so
readers do not treat either project's numbers as a drop-in substitute for the
other.

| | This repository | r0b0tlab NVFP4 SGLang |
|---|---|---|
| Code | this repo, recipe `llama-cpp/qwen38-flash-next-ud-iq4-xs` | [r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang](https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang) |
| Weights | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ4_XS` revision `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f` | [r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121) |
| Hardware | **1×** NVIDIA DGX Spark GB10 | **2×** GB10, tensor parallel 2, RoCE |
| Runtime | llama.cpp Qwen4Exp ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) commit `250b61446`) | SGLang Qwen4-Exp + their SM121 patches |
| Quant | Unsloth `UD-IQ4_XS` GGUF, 93,682,584,224 bytes across 3 shards | ModelOpt NVFP4 W4A16 (`modelopt_mixed`), BF16 activations / KV, FP32 GDN state |
| Base weights | Unsloth GGUF of Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next` revision `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| PLE / n-gram table | CPU mmap, lazy row reads (`POSIX_MADV_RANDOM`); ~27 GiB not kept always-resident | 51 GiB table kept as **FP8**, mmap-free loader (`--disable-mmap`) |
| Speculative decoding | **off** in the published recipe (`ngram-mod` excluded from speed claims) | MTP NEXTN (`s3d4`): card cites **2.48× c1 / 2.28× c4** vs AR |
| Short autoregressive decode | **~25 tok/s** on 1 GPU, cache-off, ctx 4096, unpatched kernels | **~9.9 tok/s AR**; NEXTN **14.7–30.8 tok/s** depending on concurrency |
| Long context | Native 262,144 allocation succeeded; 229,874-token prompt **5.60 tok/s** unpatched | 262K / full Q200 **not** in that snapshot |
| Status | `draft` (open llama.cpp PR; experimental QSA kernels separate) | Self-quantized SM121 qualification; CUDA-graph MTP decode is a later epoch |

Sources for the right-hand column: the Hugging Face model card and the companion
GitHub README linked above. Do not copy those NEXTN multipliers onto this
llama.cpp recipe.

## What transfers

- Both stacks are GB10 / SM121 work on Qwen3.8-Flash-Next (180B-A4B hybrid GDN +
  QSA + large PLE table).
- Both treat unified memory as the constraint: this recipe uses lazy CPU PLE;
  theirs uses mmap-free reads so the table stays reclaimable.
- Neither project should be cited as a verified production serving number.

## What does not transfer

- **1 GPU vs 2 GPU.** Their AR 9.9 tok/s is on TP=2. Our ~25 tok/s AR is on one
  Spark. Different memory topology, different kernel paths.
- **GGUF IQ4 vs NVFP4 W4A16.** Different GEMM, different MoE runners, different
  numeric contract.
- **No-spec llama.cpp vs MTP SGLang.** Their 2.48× NEXTN figure is speculative
  decode. This recipe's headline 25 tok/s is autoregressive.
- **Prompt protocol.** This recipe's 25 tok/s used `tools/stream_benchmark.py`
  against `llama-server`. Overnight QSA kernel work used a different greedy
  “count 1–20” protocol and must not be mixed with 25 tok/s (see
  [`../recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md`](../recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md)).

## Related llama.cpp comparison (not NVFP4)

Methodology for this recipe's task-shape suite was also reviewed against
[`0xBakeer/qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark)
commit `4c6fc3af429bff5c472511cf965751eac6b7caf2`. That is a separate llama.cpp
tree. Its graph-reuse patch was ported here, reached `graphs reused = 127`, then
segfaulted; it is **not** shipped.
