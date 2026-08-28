# Qwen3.8 Flash Next One — DGX Spark inference recipes

Reproducible inference recipes for **Qwen3.8-Flash-Next** on **NVIDIA DGX Spark**
(GB10). Each recipe is a directory with a manifest, environment template,
operator notes, and an executable entrypoint for one **runtime lane** and
**model revision**.

Lanes exist for **SGLang**, **llama.cpp**, and **vLLM**. Only llama.cpp has a
populated recipe today. SGLang and vLLM stay fail-closed until someone lands
measured Spark evidence in those directories.

## Current catalog

| Recipe | Runtime | Weights | Status | What is measured |
|--------|---------|---------|--------|------------------|
| [`recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/) | llama.cpp Qwen4Exp ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ4_XS` @ `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f` | **`draft`** | 1× GB10, no speculative decoding |

It remains `draft` because the llama.cpp architecture is still an open PR, and
the experimental QSA kernel patch is not the default `run.sh` path.

Headline **unpatched** numbers on one Spark (see the recipe README for
methodology):

- short prompt, cache-off: **~25 tok/s** decode, **0.551 s** TTFT, ctx 4096
- native **262,144** context allocation succeeded
- 229,874-token prompt: **5.60 tok/s** decode, **1,218.85 s** TTFT
- parallel 2 @ ctx 8192: **20.68 tok/s/request**, **32.82** aggregate output tok/s

An experimental QSA kernel patch
([`qsa-lightning-working.patch`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/patches/qsa-lightning-working.patch))
raised the 64k greedy-count protocol from **11.35 → 18.73 tok/s** with locked
output hashes. At 128k the same protocol was **15.35 tok/s** (PDL) and
**13.96 tok/s** (`__ldg` — slower). Different prompt than the 25 tok/s figure.
Details: [`results/qsa-kernels.md`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/results/qsa-kernels.md).

Closed llama.cpp [PR #27842](https://github.com/ggml-org/llama.cpp/pull/27842)
(`draft-mtp`, n-max 3) was ported onto an isolated `250b61446` tree and measured
on this GGUF with a 3.9 GiB Q8_0 MTP head converted from the local FP8
checkpoint: **~40.5 tok/s** decode at ctx 4096, **75.6%** draft accept
(**~1.6×** vs unpatched AR). Not merged upstream; not `run.sh`. Details:
[`results/mtp-draft.md`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/results/mtp-draft.md).

## Not this repository

| Project | Relation |
|---------|----------|
| [r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121) + [companion SGLang repo](https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang) | 2× GB10, ModelOpt NVFP4 W4A16, SGLang MTP NEXTN. Different quant, GPU count, and decoder. Comparison notes: [`docs/comparison-nvfp4-w4a16-sglang.md`](docs/comparison-nvfp4-w4a16-sglang.md) |
| [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp methodology reference. Graph-reuse port **segfaulted** here and is not shipped |

Do not paste their NEXTN 2.48× or NVFP4 tok/s onto this GGUF recipe.

## Runtime lanes

| Lane | Path | State |
|------|------|-------|
| llama.cpp | [recipes/llama-cpp/](recipes/llama-cpp/) | One draft recipe + experimental QSA patch |
| SGLang | [recipes/sglang/](recipes/sglang/) | Empty; use the generator. Related work is r0b0tlab’s NVFP4 stack, not duplicated here |
| vLLM | [recipes/vllm/](recipes/vllm/) | Empty; use the generator |

Runtime IDs: [`config/runtimes.json`](config/runtimes.json). Catalog rules:
[recipes/README.md](recipes/README.md).

## Recipe layout

```text
recipes/<runtime>/<slug>/
├── recipe.json    # manifest (schema v1)
├── README.md      # operator notes
├── run.sh         # entrypoint (draft stubs fail closed)
└── env.example    # non-secret environment template
```

## Create another draft

Python 3.11+; stdlib only for repository tooling.

```bash
python3 scripts/new_recipe.py \
  --runtime llama-cpp \
  --slug my-model-slug \
  --title "Short human title" \
  --summary "One-line purpose" \
  --model your-org/your-model \
  --revision main
```

Or `make new`. Mutable refs such as `main` are fine for drafts; pin an immutable
revision before `verified`.

## Validate

```bash
make validate   # recipe manifests
make test       # tooling unit tests
make check      # both
```

CI runs the validator and `python3 -m unittest discover -s tests -v`.
Inference is **not** launched in CI.

## Lifecycle

| Status | Meaning |
|--------|---------|
| `draft` | In progress; `tested_at` is null |
| `verified` | Reproduced on DGX Spark GB10 with evidence; `tested_at` required (`YYYY-MM-DD`) |
| `deprecated` | Superseded or unsafe; kept for history |

Promotion checklist: [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for manifest fields, path rules, and the
draft-to-verified checklist. Use the issue templates for a new recipe or a bug.
Pull requests should complete [.github/pull_request_template.md](.github/pull_request_template.md).
