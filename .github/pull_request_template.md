## Summary

<!-- What does this PR change? Link related issues. -->

## Recipe changes

<!-- Delete sections that do not apply. -->

- [ ] Adds a new recipe under `recipes/<runtime>/<slug>/`
- [ ] Updates an existing recipe
- [ ] Deprecates a recipe (`status: deprecated`)
- [ ] Documentation or tooling only (no recipe manifest changes)

**Recipe path(s):** <!-- e.g. recipes/llama-cpp/my-slug -->

**Manifest status:** <!-- draft | verified | deprecated -->

## Validation checklist

- [ ] `make validate` passes locally (or `python3 scripts/validate_recipes.py`)
- [ ] `make test` passes locally (or `python3 -m unittest discover -s tests -v`)
- [ ] `env.example` contains no secrets — names and safe placeholders only
- [ ] `recipe.json` fields match schema v1 and `$schema` path is correct

## Reproducibility checklist (required for `verified`)

Complete every item when promoting a recipe to **`verified`**. Leave unchecked for **`draft`** PRs.

- [ ] **Model** — exact repository URL and immutable revision (commit SHA, artifact digest, or file checksum) documented in PR
- [ ] **Converted artifacts (if applicable)** — source/output revisions, conversion tool/config, quantization policy, shard index, and checksums documented
- [ ] **Runtime** — exact version (image digest, package version, or git SHA) documented
- [ ] **Command / config** — executable `run.sh` contains the real invocation/config used on DGX Spark GB10
- [ ] **Context & concurrency** — max sequence length, batch size, parallel requests stated
- [ ] **Memory** — steady-state GPU (and host if relevant) usage recorded
- [ ] **TTFT & throughput** — measurement method and numbers included
- [ ] **Output validation** — how correctness was verified (golden prompts, tests, logprobs, etc.)
- [ ] **`tested_at`** — set in `recipe.json` to reproduction date (`YYYY-MM-DD`)

## Secret hygiene

- [ ] No API keys, tokens, passwords, or private URLs in committed files
- [ ] Recipe `README.md` explains credential acquisition if needed (not committed values)

## Test plan

<!-- Steps a reviewer can follow on DGX Spark GB10, or reason why not applicable. -->
