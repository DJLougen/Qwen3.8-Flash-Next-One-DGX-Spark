# Qwen3.8 Flash Next NVFP4 — vLLM, one DGX Spark

Serves [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(revision `7b71922`) on a single GB10 with the PLE n-gram table offloaded to host memory.

**Status: `draft`.** The launcher and every environment setting below are in daily use on a GB10,
but the measured numbers this author holds are for a *derived* checkpoint (see
[Faster variants](#faster-variants)). A `verified` promotion needs a run of **this** recipe against
**this** revision, which is why `tested_at` is still `null`.

## Requirements

- One DGX Spark (GB10, sm_121, 128 GB unified memory)
- **Swap.** At least 48 GiB. The PLE table is offloaded to *pageable* host memory by design; with
  no swap the offload worker is OOM-killed during startup and vLLM reports only
  `PLE offload worker exited during startup`.
- vLLM `0.1.dev20073+g8e685d198` — a preview build combining
  [vllm#53896](https://github.com/vllm-project/vllm/pull/53896) (the model) and
  [vllm#53899](https://github.com/vllm-project/vllm/pull/53899) (PLE offload). **This model is not
  on `main` and not in any release**, and *neither PR alone reproduces this tree* —
  `vllm/v1/ple_offload/` exists only on #53899. `run.sh` refuses to start against another version
  unless `ALLOW_UNPINNED_VLLM=1`.

  ⚠️ **Paths in this document are build-specific.** Since this build, #53896 renamed the package
  `vllm/models/qwen3_8_flash_next/` → `vllm/models/qwen4_exp/`, and refactored the NGram helpers
  (+108/−73 in `ple_layer.py`). Findings below still hold; the file names have moved.

## Quick start

```bash
cp env.example .env && ${EDITOR:-vi} .env
set -a && . ./.env && set +a
./run.sh
```

## Why each environment variable is there

None of these is a preference; each was a distinct failure first.

| Variable | Without it |
|---|---|
| `CUTE_DSL_ARCH=sm_121a` | NVFP4's `cvt.e2m1x2` needs `sm_121a`; plain `sm_121` silently misses |
| `VLLM_USE_DEEP_GEMM=0` | DeepGEMM gates on device-capability family 120, which GB10 satisfies, then faults with `CUDA error: unspecified launch failure` inside `deep_gemm.fp8_gemm_nt` |
| `VLLM_GDN_DECODE_KERNEL=triton` | FP8 GDN projections **deterministically hang** the engine at c≈32 with the default CUDA kernel — no error, requests simply stall |
| `VLLM_PLE_CPU_OFFLOAD=1` | The 51.2 B-parameter PLE table does not fit beside the weights |

## Three flags that are not about speed

- **`--distributed-executor-backend mp`.** `spawn_ple_offload()` is called from
  `multiproc_executor.py` and nowhere else, while vLLM selects the **uniproc** executor by default
  at TP=1. The GPU side then waits forever on a worker that was never started
  ([vllm#53960](https://github.com/vllm-project/vllm/issues/53960)).
- **`--enable-auto-tool-choice --tool-call-parser qwen3_xml`.** With only `--reasoning-parser
  qwen3`, every request carrying a `tools` field returns **HTTP 400**. Note
  `--tool-call-parser qwen3` is *not* valid — the reasoning parser is `qwen3`, but the tool side
  registers only as `qwen3_coder` / `qwen3_xml`, two aliases of one class.
- **`MAX_MODEL_LEN=32768`, not 8192.** 8192 is a benchmarking value. One code task emitted 31,115
  characters of reasoning before 12,931 of content; a short window truncates mid-thought and
  returns empty content with no error.

Send `reasoning_effort` explicitly (`low` | `medium` | `high`). The chat template defaults it to
`xhigh`, which on this author's box does not converge — the whole budget goes into `<think>` and
the response carries no content.

## Speculation

`SPEC_MTP_K=2` uses the in-checkpoint MTP layer. `k=5` **hard-fails** (the QSA ring capacity must
divide the attention block size), so `run.sh` rejects it. Do **not** combine MTP with
`--async-scheduling`: `_prepare_ngram_context` reads the CPU token mirror while it still holds
speculation's `-1` placeholders, producing a wrong n-gram context that no benchmark reveals.

## Prefix caching behaves unusually on this architecture

vLLM raises the attention block size **to 1600 tokens** to match this model's Mamba state page
(`Setting attention block size to 1600 tokens…`, then pads the mamba page by 0.88% so the two are
equal). A prompt shorter than one block therefore has **no full block to cache**: this author
measured `prefix_cache_hits_total` at **0 across 77,254 queries** before noticing why.

⚠️ **1600 is the attention group's block, not `cache_config.block_size`.** A third KV group exists
that this alignment does not cover — the QSA raw-key ring, a `CircularBufferSpec` whose block is its
ring capacity, `compress_ratio * cdiv(compress_ratio + num_speculative_tokens, compress_ratio)` = 8
with `indexer_compress_ratio 4` and MTP k=2. `v1/engine/core.py:321` sets `cache_config.block_size`
to the **minimum over all groups**, so the configured value is plausibly 8. The hit figures above
are the attention group's blocks and are unaffected, but do not read 1600 as the block size. Repeating an
identical ~1,400-token prompt yields zero hits; a ~5,700-token prompt starts hitting from the third
request. Budget at least three identical requests before concluding anything, or you measure the
warm-up and call it a defect.

## Known upstream issue affecting startup

Builds of `#53899` before commit `4e8b849b8d97` share **one** `_input_ready_event` across all
in-flight PLE requests. Under async scheduling (on by default) two batches are in flight and the
event is re-recorded mid-staging, deadlocking startup at `warmup_kernels` with the offload thread
parked in `_copy_cuda_inputs`. If your build predates that commit and hangs there,
`--no-async-scheduling` is a candidate workaround; on this author's box it costs nothing measurable
at c=1 or c=16, but the deadlock itself has never reproduced here, so that is a code-derived
suggestion rather than a verified fix.

## Faster variants

The published NVFP4 checkpoint leaves the dense projections (attention q/k/v/o, GDN
`in_proj`/`out_proj`) in BF16. Quantizing those to FP8, and then `lm_head` as well, is worth a large
multiple on single-stream decode on this hardware — the `lm_head` lever roughly **doubles** in value
once MTP is enabled, because it accelerates the draft head too. Those require a local checkpoint
build and so are deliberately **not** part of this recipe, which targets the public revision as
published.
