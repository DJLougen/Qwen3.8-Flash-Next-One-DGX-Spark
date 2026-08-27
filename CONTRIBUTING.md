# Contributing

Thank you for helping build reproducible inference recipes for DGX Spark. This document covers paths, manifests, evidence requirements, and pull request expectations.

## Before you start

- Read the root [README.md](README.md) for repository purpose and current state.
- Pick a runtime lane: [recipes/sglang/](recipes/sglang/), [recipes/llama-cpp/](recipes/llama-cpp/), or [recipes/vllm/](recipes/vllm/).
- Use the generator or Makefile `new` target to scaffold a **draft** — do not hand-copy directory layouts.

## Path and naming conventions

| Rule | Detail |
|------|--------|
| Recipe path | `recipes/<runtime>/<slug>/` |
| Runtime ID | Exactly `sglang`, `llama-cpp`, or `vllm` |
| Slug | Lowercase letters, digits, and hyphens; unique within the lane |
| Lane README | `recipes/<runtime>/README.md` is documentation only — not a recipe |

Each recipe directory must contain:

- `recipe.json` — manifest
- `README.md` — reproduction notes
- `run.sh` — executable entrypoint
- `env.example` — non-secret environment template

## Manifest (`recipe.json`) — schema v1

Required fields:

| Field | Value |
|-------|--------|
| `$schema` | `../../../schema/recipe.schema.json` |
| `schema_version` | `1` |
| `id` | `<runtime>/<slug>` |
| `title` | Short human title |
| `summary` | One-line description |
| `runtime` | Runtime ID |
| `model.repository` | Hugging Face org/repo or documented artifact source |
| `model.revision` | Immutable commit or artifact digest for `verified`; mutable refs are draft-only |
| `hardware.target` | `NVIDIA DGX Spark` |
| `hardware.gpu` | `GB10` |
| `entrypoint` | `run.sh` |
| `status` | `draft`, `verified`, or `deprecated` |
| `tested_at` | `null` for `draft`; ISO `YYYY-MM-DD` required for `verified` |

Validate locally before opening a PR:

```bash
make validate
```

## Draft → verified criteria

A recipe stays **`draft`** until a contributor can reproduce inference on DGX Spark GB10 and supplies evidence in the pull request.

### Required evidence for `verified`

Include all of the following in the PR description (or linked artifact with stable URL). Vague claims are not sufficient.

1. **Model identity**
   - Repository URL and immutable revision (commit SHA, artifact digest, or file checksum).
   - Any conversion or quantization step with pinned tool versions and output checksums.

2. **Runtime identity**
   - Exact runtime version: container image digest, package version, or git SHA.
   - Installation path if not obvious from `run.sh`.

3. **Command and configuration**
   - Full invocation as run on the host (or equivalent config files).
   - Environment variables documented in `env.example` with example non-secret values.
   - Flags affecting context length, batching, parallelism, and precision.

4. **Context and concurrency**
   - Max sequence length (prompt + generation).
   - Batch size and concurrent request count used for measurements.

5. **Memory**
   - GPU memory at steady state (and host RAM if relevant).
   - Note OOM boundaries if tested.

6. **TTFT and throughput**
   - Measurement method (warmup, request count, tooling).
   - Reported time-to-first-token and tokens/sec (or requests/sec) with hardware idle state noted.

7. **Output validation**
   - How outputs were verified: golden prompts, checksums, logprob checks, downstream task, or regression suite.
   - Note any nondeterminism (sampling, flash attention, etc.).

8. **Date**
   - Set `tested_at` in `recipe.json` to the date reproduction was performed (`YYYY-MM-DD`).

### What does not qualify

- Running once without documented numbers or validation.
- Numbers from a different GPU or host without a DGX Spark GB10 rerun.
- Copying marketing benchmarks or upstream README claims.
- Leaving `run.sh` as a placeholder that exits with an error.

## Secret hygiene

- **Never** commit API keys, tokens, passwords, private URLs, or license files you are not authorized to redistribute.
- `env.example` lists variable **names** and safe placeholders only.
- Use local `.env` files (gitignored) or host secret stores for real values.
- If a recipe needs credentials, document acquisition steps in the recipe `README.md`, not in committed files.

## Converted and quantized artifacts

Do not commit model weights, GGUF files, checkpoints, or converted shards. Common weight extensions are gitignored. Reference immutable artifact locations and checksums instead.

For any conversion—including mixed-bit MoE weights—keep the source model and the runtime artifact distinct in the recipe README:

- Source repository and exact source revision.
- Conversion tool, commit or container digest, full command/config, and quantization policy (including which layers or experts use each bit width).
- Output format, shard filenames, index filename, total size, and SHA-256 checksums.
- Destination repository and immutable revision that the runtime actually loads.
- Conversion completion evidence, load/inference validation, and post-conversion Spark health checks. A successful conversion alone does not make an inference recipe `verified`.

Keep benchmark evidence reviewable: commit only compact text or CSV results; place large raw captures in durable external storage and link them from the recipe README.

## Pull request procedure

1. Branch from `main`.
2. Scaffold or update recipes only under `recipes/<runtime>/<slug>/`.
3. Run `make check` locally (validation + unit tests).
4. Open a PR using the template checklist.
5. For `verified` promotion, complete every evidence item above in the PR body.
6. Respond to review feedback; update `tested_at` if you re-run reproduction on a later date.

### Scope guidance

- One recipe per PR when possible — easier to review and bisect.
- Do not change unrelated recipes or tooling in the same PR unless required for your recipe.
- Deprecate rather than delete when users may still reference an old path; set `status` to `deprecated` and explain the replacement in `README.md`.

## Getting help

- **New recipe idea** — open an issue and choose the **Recipe request** template.
- **Bug in a recipe or tooling** — open an issue and choose the **Bug report** template.
