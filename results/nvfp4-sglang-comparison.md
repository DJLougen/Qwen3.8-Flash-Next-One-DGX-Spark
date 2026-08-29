# Comparison: llama.cpp UD-IQ4_XS vs r0b0tlab NVFP4 SGLang

This note records how the default recipe in this repository differs from the
public r0b0tlab NVFP4 + SGLang stack on Qwen3.8-Flash-Next. The r0b0tlab stack
was not rerun on this repository's hardware. Figures are cited as reported by
each project; they are not interchangeable without matching hardware, weights,
runtime, and benchmark protocol.

| | This repository | r0b0tlab NVFP4 SGLang |
|---|---|---|
| Code | this repo, recipe `llama-cpp/qwen38-flash-next-ud-iq4-xs` | [r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang](https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang) |
| Weights | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ4_XS` revision `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f` | [r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121) |
| Hardware | **1×** NVIDIA DGX Spark GB10 | **2×** GB10, tensor parallel 2, RoCE |
| Runtime | llama.cpp Qwen4Exp ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) commit `250b61446`) | SGLang Qwen4-Exp + SM121 patches |
| Quant | Unsloth `UD-IQ4_XS` GGUF, 93,682,584,224 bytes across 3 shards | ModelOpt NVFP4 W4A16 (`modelopt_mixed`), BF16 activations / KV, FP32 GDN state |
| Base weights | Unsloth GGUF of Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next` revision `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| PLE / n-gram table | CPU mmap, lazy row reads (`POSIX_MADV_RANDOM`); ~27 GiB not kept always-resident | 51 GiB table kept as **FP8**, mmap-free loader (`--disable-mmap`) |
| Speculative decoding | **off** in the published recipe (`ngram-mod` excluded from headline speed) | MTP NEXTN (`s3d4`): model card cites **2.48× c1 / 2.28× c4** vs AR |
| Short autoregressive decode | **~25 tok/s** on 1 GPU, cache-off, ctx 4096, unpatched kernels | **~9.9 tok/s AR**; NEXTN **14.7–30.8 tok/s** depending on concurrency |
| Long context | Native 262,144 allocation succeeded; 229,874-token prompt **5.60 tok/s** unpatched | 262K / full Q200 **not** in that snapshot |
| Status | `draft` (open llama.cpp PR; experimental QSA kernels in sibling recipe) | Self-quantized SM121 qualification; CUDA-graph MTP decode is a later epoch |

Right-hand figures come from the Hugging Face model card and companion GitHub
README linked above.

## Shared context

- Both targets are GB10 / SM121 work on Qwen3.8-Flash-Next (180B-A4B hybrid GDN +
  QSA + large PLE table).
- Both designs assume unified memory pressure: this recipe uses lazy CPU PLE;
  the NVFP4 stack uses mmap-free loading so the table remains reclaimable.
- Neither project should be read as a production SLA without independent
  reproduction on your hardware.

## Scope boundaries

- **GPU topology.** Their AR 9.9 tok/s is on TP=2; our ~25 tok/s AR is on one
  Spark.
- **Weight format.** GGUF IQ4_XS and NVFP4 W4A16 use different kernels and
  numeric contracts.
- **Decoder mode.** Their NEXTN multipliers include speculative draft acceptance;
  this recipe’s headline decode is autoregressive.
- **Measurement.** Default short-prompt numbers use `tools/stream_benchmark.py`
  against `llama-server`. The QSA sibling recipe uses a separate greedy-count
  protocol; see
  [`../recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md`](../recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md).
