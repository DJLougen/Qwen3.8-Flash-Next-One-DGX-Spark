# Qwen3.8 Flash Next One — DGX Spark inference recipes

Reproducible inference recipes for **Qwen3.8-Flash-Next** on **NVIDIA DGX Spark**
(GB10). Each recipe is a directory with a manifest, environment template,
operator notes, and an executable entrypoint for one **runtime lane** and
**model revision**.

Lanes exist for **SGLang**, **llama.cpp**, and **vLLM**. Only llama.cpp has a
populated recipe today. SGLang and vLLM stay fail-closed until someone lands
measured Spark evidence in those directories.

## Results

Measured on one NVIDIA DGX Spark (GB10). Reproduce via each recipe's `run.sh`
and `results/`.

| Configuration | Prompt | Decode | TTFT |
|---|---|---:|---:|
| Default unpatched `250b61446` | 76-token short, cache off (`b2048`/`ub512`) | **~29 tok/s** | 0.15 s |
| Default unpatched `250b61446` | 4k-target depth, cold (`b2048`/`ub512`) | **~25 tok/s** | 12.65 s |
| Default unpatched | 229,874 depth | **5.60 tok/s** | 1,218.85 s |
| Default unpatched, 2 parallel | 8,192 each | **20.68 tok/s/req** (32.82 agg.) | 0.853 s |
| QSA kernels (sibling, fail-closed) | 65,536 greedy-count | **18.73 tok/s** | — |
| QSA kernels (sibling, fail-closed) | 229,859 depth | **11.55 tok/s** | 1,198.88 s |
| draft-mtp n-max 3 (isolated tree) | 4,096 | **~40.5 tok/s** (75.6% accept) | — |
| draft-mtp on QSA kernels (combined tree) | 229,859 depth | 10.2 tok/s (43% accept) — slower than kernel AR | 1,242.96 s |

Rows that differ in binary or benchmark protocol are labeled; see each recipe's
`results/` before ranking figures across rows.

Context length, ten-tests campaign 2026-08-30/31 (true-cold unless noted). Full
evidence: [`recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/results/raw/ten-tests/`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/results/raw/ten-tests/).

| Ctx | Config | TTFT (s) | Prefill tok/s | Decode tok/s | Hash | Guard min GiB |
|---|---|---:|---:|---:|---|---:|
| ~76 (warm) | Gate 0 ub512 | 0.149 | — | 28.6 | `cb7904d8` | — |
| ~76 (warm) | T1 GET_ROWS | 0.144 | — | 26.66 | `cb7904d8` | — |
| ~76 (warm) | T5 kmtp+MTP | 0.322 | — | 26.54 | — | — |
| **4k** | Gate 0 ub512 | 10.791 | 372.5 | 24.6 | `99a15d5b` | — |
| **4k** | T1 GET_ROWS ub512 | **6.806** | **599.9** | 23.42 | `99a15d5b` | 50.86 |
| **4k** | T3 ub1024 | **9.199** | 438.9 | 23.79 | `06124a4b` | — |
| **4k** | T9 kmtp ub512 | 12.011 | 335.0 | 24.95 | `c64973d8` | 50.86 |
| **64k** | Gate 0 ub512 | 170.663 | 384.7 | 14.5 | `b641e2eb` | — |
| **64k** | T1 GET_ROWS ub512 | **131.94** | **498.1** | 13.96 | `b641e2eb` | — |
| **64k** | T3 ub1024 | **160.99** | 408.4 | 14.35 | `a81283e2` | — |
| **64k** | T9 kmtp ub512 | 166.57 | 393.9 | **20.44** | `b0ea9f23` | 47.71 |
| **128k** | era f16 ub1024 | 386.77 | 339.5 | — | — | — |
| **128k** | T4 kvq8 ub1024 | 397.5 | 330.4 | 9.78 | `9b622db0` | 44.2 |
| **230k / 262k** | T4 kvf16 | — | — | — | — | **35.77 breach** |
| **230k / 262k** | T4 kvq8 ub1024 | 901.65 | 255.4 | 6.20 | `1cda86a2` | 37.97 |
| **230k** | T9 kmtp ub1024 | 922.76 | 249.6 | **12.94** | `e2875202` | 36.12 |

T1 owns 4k/64k prefill. T9 owns 64k/230k decode (QSA). T4 q8_0 is the only config that loads 262k under the 36 GiB floor.

## Current catalog

| Recipe | Role | Status | Measured |
|--------|------|--------|----------|
| [`recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/) | **Public default** — unpatched llama.cpp [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) | **`draft`** | 1× GB10, no speculative decoding |
| [`recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/`](recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa/) | Experimental QSA CUDA kernels (patch + hashes). `run.sh` fails closed | **`draft`** | Separate prompt protocol from the default |

SGLang and vLLM lanes are empty.

## Runtime lanes

| Lane | Path | State |
|------|------|-------|
| llama.cpp | [recipes/llama-cpp/](recipes/llama-cpp/) | Default unpatched recipe + fail-closed QSA kernel config |
| SGLang | [recipes/sglang/](recipes/sglang/) | Empty; use the generator |
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

## External results and references

Third-party stacks and port attempts we compared against or tested live in
[`results/`](results/): [NVFP4 SGLang](results/nvfp4-sglang-comparison.md) and
the [0xBakeer graph-reuse port](results/graph-reuse-port.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for manifest fields, path rules, and the
draft-to-verified checklist. Use the issue templates for a new recipe or a bug.
Pull requests should complete [.github/pull_request_template.md](.github/pull_request_template.md).

## License

This project is licensed under the terms of the custom permissive license with specific exclusions. See [LICENSE](LICENSE) for full legal terms and conditions.
