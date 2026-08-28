# Experimental QSA kernel track (2026-08-27/28)

> **Not the public default.** The serving recipe is
> [`../../qwen38-flash-next-ud-iq4-xs/`](../../qwen38-flash-next-ud-iq4-xs/)
> (unpatched `250b61446`). These numbers are from `/home/djl/llama.cpp-qwen4exp`
> with [`../patches/qsa-lightning-working.patch`](../patches/qsa-lightning-working.patch).
>
> Do not mix them with the unpatched **~25 tok/s** short-prompt figure in
> [`../../qwen38-flash-next-ud-iq4-xs/results/summary.md`](../../qwen38-flash-next-ud-iq4-xs/results/summary.md).

## Protocol

- Host: one NVIDIA DGX Spark (GB10), CUDA 13.0.2, driver 580.159.03
- Weights: same `UD-IQ4_XS` shards and SHA-256 as the recipe
- Binary: `llama-server` built `GGML_CUDA=ON`, arch `121a-real`
- Prompt: greedy continuation of `1 2 3 … 20` (51 completion tokens)
- `chat_template_kwargs.thinking = false`
- Output hashes locked before speed claims:
  - ctx 4096 → `2689367b205c16ce`
  - ctx 64k and 128k → `8547299278d81f66`
- CUDA graphs **were reused** on this tree (304 graphs at 64k, 563 at 128k in
  the `__ldg` configuration). That is **not** the rejected 0xBakeer graph-reuse
  patch, which segfaulted and was removed from the recipe default.

## Unpatched recipe baseline (same weights, different protocol)

From [`summary.md`](summary.md), `stream_benchmark.py`, thinking left at the
server default:

| Context | Decode tok/s | Notes |
|---|---:|---|
| 4,096 short | ~25 | cache-off TTFT 0.551 s |
| 65,536 | 11.35 | generated depth prompt |
| 131,072 | 8.16 | generated depth prompt |
| 229,874 | 5.60 | TTFT 1,218.85 s; 39.53 GiB min available |

## Patched QSA (`qsa-lightning-working.patch`)

What landed and stayed in the patch file:

- fused `ggml_get_rows_mean` + RMS weighting (`r=4`)
- `__ldg` half2 / float4 loads on lightning WMMA K and Q
- compact FA gather (`topk=2048`)
- Indexer Q padded 4→8 so lightning hits WMMA
- Vec8 f16→f32 compact-FA gather
- PDL on mean + lightning (`r=4`)

Tried and **reverted** (hash-safe but not faster, or slower):

- 4-head lightning inner-loop / in-kernel WMMA pad
- drop 8-head graph pad
- IQ4_XS 8-warp MMVQ
- 4-output-block mean CTA
- GDN extra `pdl_lc` (17.13 tok/s at 64k vs 18.41 before revert)
- partial / incremental-pool / dirty-block skip for indexer K (128k still
  dropped; QSA layers with `r=4` randomly reread ~405 MB/token)

| Ctx | tok/s | Hash | Notes |
|---|---:|---|---|
| 4,096 | 20.52 | `2689367b205c16ce` | `__ldg` on lightning K/Q; not comparable to unpatched 25 tok/s |
| 65,536 | **18.73** | `8547299278d81f66` | best 64k so far; graphs reused 304 |
| 131,072 | **15.35** | `8547299278d81f66` | PDL configuration (no `__ldg` on K/Q) |
| 131,072 | 13.96 | `8547299278d81f66` | `__ldg` K/Q; **slower than PDL at 128k**; graphs reused 563; prefill 3.75 ms/token |

Keep `__ldg` for the 64k win. At 128k, PDL without those loads was faster on
this protocol. The 64k patched **18.73 tok/s** vs unpatched **11.35 tok/s** is
the kernel result that matters. Prefill of a 64k prompt was ~3.0 ms/token
(~330 tok/s), similar before and after the lightning load change.

## Patch

Apply on the Qwen4Exp llama.cpp tree, then rebuild `llama-server`:

```text
recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/patches/qsa-lightning-working.patch
```

Ten files, 1140 lines. Do not apply the rejected PLE lazy-advice prototype or
the 0xBakeer graph-reuse patch.
