# Qwen3.8 Flash Next One — DGX Spark inference recipes

A workspace for developing reproducible inference recipes for **NVIDIA DGX Spark** (GB10). Each future recipe will bundle manifest metadata, environment hints, and an executable entrypoint for a specific **runtime lane** and **model revision**.

> [!IMPORTANT]
> **This initial repository is setup-only.** It intentionally establishes the structure, authoring templates, validation, and runtime lanes before any inference recipe is published. Recipes will be added incrementally as model conversions, runtime commands, measurements, and Spark validation are formalized. Lane guides and candidate matrices describe where future work belongs; they are not claims that a recipe already exists or works.

## Current state

This repository is a **workspace scaffold**, not a catalog of benchmarked recipes yet.

- Supported runtime lanes are defined (`sglang`, `llama-cpp`, `vllm`), but lane directories may be empty until contributors land recipes.
- Generated recipes start as **`draft`** manifests. They **fail closed** until an author supplies a real runtime invocation in `run.sh` and documents how to reproduce results.
- Nothing here claims verified throughput, latency, or model availability until a recipe is explicitly promoted to **`verified`** with dated evidence (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Supported runtime lanes

Recipes are grouped by inference backend:

| Lane | Path | Role |
|------|------|------|
| SGLang | [recipes/sglang/](recipes/sglang/) | High-throughput GPU serving with SGLang |
| llama.cpp | [recipes/llama-cpp/](recipes/llama-cpp/) | GGUF / CPU/GPU inference via llama.cpp |
| vLLM | [recipes/vllm/](recipes/vllm/) | vLLM OpenAI-compatible serving |

The canonical registry of supported runtime IDs lives in [`config/runtimes.json`](config/runtimes.json).

## Recipe layout

Each populated recipe is a directory:

```text
recipes/<runtime>/<slug>/
├── recipe.json    # manifest (schema v1)
├── README.md      # human notes, reproduction context
├── run.sh         # executable entrypoint (draft until author completes)
└── env.example    # non-secret environment template
```

- **`<runtime>`** — one of `sglang`, `llama-cpp`, or `vllm`.
- **`<slug>`** — lowercase identifier (e.g. `qwen3-8-flash-gguf-q4`).

Lane-level `README.md` files (when present) describe runtime-specific conventions; the catalog overview is in [recipes/README.md](recipes/README.md).

## Quick start — create a draft recipe

Prerequisites: **Python 3.11+** (stdlib only for repository tooling; no install step).

### Generator CLI (recommended)

```bash
python3 scripts/new_recipe.py \
  --runtime llama-cpp \
  --slug my-model-slug \
  --title "Short human title" \
  --summary "One-line purpose" \
  --model your-org/your-model \
  --revision main
```

Optional `--root PATH` targets a non-default repository root. Mutable model refs such as `main` are suitable for a draft; pin an immutable revision before promotion to `verified`.

### Interactive Make target

```bash
make new
```

The target prompts for the same six fields without interpolating author text into shell source.

The generator rejects unknown runtime IDs, invalid slugs, and existing destinations. It creates a **draft** manifest, substitutes templates, and makes `run.sh` executable. You must still implement the actual inference command before the recipe is runnable.

## Validate recipes

```bash
make validate
# or
python3 scripts/validate_recipes.py
```

The validator scans populated recipe directories, reports actionable diagnostics, and exits nonzero on errors. Runtime lane README files are **not** recipes and are not validated as manifests.

## Tests and full check

```bash
make test    # unit tests for repository tooling
make check   # validate recipes + unit tests
```

CI runs the validator and `python3 -m unittest discover -s tests -v` on pushes and pull requests.

## Recipe lifecycle

| Status | Meaning |
|--------|---------|
| `draft` | Scaffold or work in progress; may not run; `tested_at` is null |
| `verified` | Reproduced on DGX Spark GB10 with evidence in the PR; `tested_at` is required (ISO `YYYY-MM-DD`) |
| `deprecated` | Superseded or unsafe; kept for history |

Promotion from `draft` to `verified` requires the evidence checklist in [CONTRIBUTING.md](CONTRIBUTING.md). Do not mark `verified` without a merged reproduction record.

## DGX Spark reproducibility expectations

Recipes target **NVIDIA DGX Spark** with **GB10** GPUs. Contributors should document:

- Exact **model repository and immutable revision** (commit SHA, artifact digest, or GGUF checksum) for verified recipes.
- For converted or quantized weights: exact **source revision**, conversion tool/config, bit-width policy, output repository revision, shard index, and checksums.
- Exact **runtime build or package version** (container image, wheel, or git SHA).
- **Invocation** — full command or config referenced by `run.sh`, plus any required environment from `env.example`.
- **Context and concurrency** — max sequence length, batch size, parallel requests.
- **Memory** — observed GPU/host usage at steady state.
- **TTFT and throughput** — measurement method and numbers (not marketing summaries).
- **Output validation** — how correctness was checked (golden prompts, logprobs, regression suite, etc.).

Secrets (API keys, tokens, private endpoints) must never be committed. Use `env.example` for names only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for manifest fields, path rules, PR procedure, and the full draft-to-verified checklist.

Use the issue templates to request a new recipe or report a bug. Pull requests should complete the checklist in [.github/pull_request_template.md](.github/pull_request_template.md).
