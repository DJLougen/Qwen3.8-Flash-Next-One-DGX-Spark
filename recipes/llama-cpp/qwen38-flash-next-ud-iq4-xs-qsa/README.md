# Qwen3.8 Flash Next UD-IQ4_XS — experimental QSA kernels

> **Not the public default.** Serving numbers and `run.sh` for this GGUF live in
> the unpatched sibling
> [`../qwen38-flash-next-ud-iq4-xs/`](../qwen38-flash-next-ud-iq4-xs/)
> (~25 tok/s short-prompt, 5.60 tok/s at 229k). This directory is a second
> config: the overnight QSA CUDA patch, locked output hashes, and nothing else.
>
> `run.sh` **fails closed**. The patched `llama-server` is a Spark working tree,
> not a binary in this repository.

## Why this is separate

The kernel track is faster at **long greedy-count context** (64k 11.35 → 18.73
tok/s on that protocol) and **slower / not comparable** at the sibling's short
prompt (~20.5 vs ~25 tok/s, different sampler). Shipping it as the default
would replace a better short-context recipe with a worse one.

## Patch

[`patches/qsa-lightning-working.patch`](patches/qsa-lightning-working.patch)
(10 files, 1140 lines) on llama.cpp Qwen4Exp commit `250b61446`.

Apply on a dedicated tree, rebuild `llama-server`, do not mix into the
unpatched binary used by the sibling recipe.

## Locked hashes

Greedy continuation of `1 2 3 … 20`, 51 completion tokens,
`thinking=false`:

| Ctx | tok/s | SHA-256 prefix | Notes |
|---|---:|---|---|
| 4,096 | 20.52 | `2689367b205c16ce` | `__ldg` K/Q; **not** the sibling's ~25 tok/s short prompt |
| 65,536 | **18.73** | `8547299278d81f66` | best 64k; graphs reused 304 |
| 131,072 | **15.35** | `8547299278d81f66` | PDL, no `__ldg` on K/Q |
| 131,072 | 13.96 | `8547299278d81f66` | `__ldg` K/Q; slower than PDL at 128k |

Full protocol, reverted experiments, and unpatched baselines for the same
greedy-count prompts: [`results/qsa-kernels.md`](results/qsa-kernels.md).

## Weights

Same three-shard Unsloth `UD-IQ4_XS` as the sibling, revision
`ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f`. Checksums are in the sibling
README; do not load until those match.

## Status

- Recipe ID: `llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa`
- Manifest: `draft`, `tested_at` null
- SGLang / vLLM lanes stay empty
