# Recipe catalog

Inference recipes live under `recipes/<runtime>/<slug>/`. Each recipe is a self-contained reproduction bundle for **NVIDIA DGX Spark (GB10)**.

## Runtime lanes

| Lane | Directory | Backend |
|------|-----------|---------|
| SGLang | [sglang/](sglang/) | [SGLang](https://github.com/sgl-project/sglang) serving |
| llama.cpp | [llama-cpp/](llama-cpp/) | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| vLLM | [vllm/](vllm/) | [vLLM](https://github.com/vllm-project/vllm) |

Lane directories may be empty while the repository is bootstrapped. Use the generator (`make new` or `scripts/new_recipe.py`) to add the first recipe in a lane.

## What belongs in a recipe

A **recipe** is a directory with these required artifacts; it may also include compact benchmark scripts, configs, and evidence:

| File | Purpose |
|------|---------|
| `recipe.json` | Manifest: model, runtime, hardware target, status, schema v1 fields |
| `README.md` | Human context: prerequisites, quirks, reproduction notes, links |
| `run.sh` | Executable operator entrypoint; CI checks shell syntax but does not launch inference |
| `env.example` | Environment variable names and safe example values (no secrets) |

### What is not a recipe

- `recipes/README.md` (this file)
- `recipes/<runtime>/README.md` — lane overview only
- Shared schema, scripts, tests, or config under `schema/`, `scripts/`, `tests/`, `config/`

The validator scans recipe directories only; lane README files are excluded.

## Status and lifecycle

- **`draft`** — scaffold or in progress; `tested_at` is null; `run.sh` may fail closed until completed.
- **`verified`** — reproduced on DGX Spark GB10 with PR evidence; `tested_at` required.
- **`deprecated`** — retained for history; points users to a replacement.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full evidence checklist required to mark a recipe `verified`.

## Create a recipe

```bash
make new
```

This interactive target prompts for runtime, slug, title, summary, model repository, and revision. For automation, use the non-interactive `scripts/new_recipe.py` command documented in the [root README](../README.md#generator-cli-recommended).

Then edit `run.sh`, `README.md`, and `env.example` until the recipe runs on your Spark host. Validate with `make validate` before opening a pull request.
