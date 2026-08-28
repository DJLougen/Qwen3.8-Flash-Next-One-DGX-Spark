# llama.cpp recipes (DGX Spark)

This lane holds **llama.cpp** inference recipes for **NVIDIA DGX Spark** (GB10). Each recipe is a self-contained directory under `recipes/llama-cpp/<slug>/` with a manifest, runnable entrypoint, environment template, and operator notes.

Populate recipes here when the primary runtime is **llama.cpp**—typically GGUF weights, `llama-cli` / `llama-server`, and CUDA-backed GPU offload—not when the intended path is SGLang or vLLM (those belong in sibling lanes).

---

## What belongs in this lane

| Belongs here | Does not belong here |
|--------------|----------------------|
| GGUF-based inference via llama.cpp binaries built from a pinned source commit | Hugging Face safetensors served directly by vLLM or SGLang |
| Workflows that document **how the GGUF was produced** (converter, quant method, upstream revision) | A recipe that only points at a model repo with no llama.cpp invocation plan |
| Draft or verified `llama-cli`, batch, or OpenAI-compatible **server** entrypoints | Placeholder commands that skip build/provenance documentation |
| Operator notes for unified memory, layer offload, and context sizing on Spark | Claims of verified throughput/latency without a recorded benchmark run |

Recipes start as **`draft`** in `recipe.json` and must **fail closed** in `run.sh` until the author supplies a real, tested invocation. Do not merge fabricated benchmark numbers or compatibility guarantees.

---

## Authoritative llama.cpp sources

Use upstream docs and source—not third-party summaries—for build flags, server APIs, and quantization semantics.

| Resource | URL |
|----------|-----|
| Project repository | https://github.com/ggml-org/llama.cpp |
| Build instructions (CUDA, CMake options) | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md |
| `llama-server` / OpenAI-compatible HTTP API | https://github.com/ggml-org/llama.cpp/tree/master/tools/server |
| GGUF format & conversion tooling | https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize |
| Chat-template flags and behavior | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |

When a recipe pins a commit, cite the **llama.cpp git SHA** used to build binaries separately from the **Hugging Face model revision** and the **GGUF file provenance** (see checklist below).

---

## Scaffold a new recipe

From the repository root:

```bash
python3 scripts/new_recipe.py \
  --runtime llama-cpp \
  --slug baseline-gguf \
  --title "llama.cpp baseline GGUF" \
  --summary "Draft GGUF recipe with pinned source and artifact provenance." \
  --model your-org/your-source-model \
  --revision your-tested-revision
```

Add `--root /path/to/repository` only when targeting a different checkout.

This creates `recipes/llama-cpp/<slug>/` with `recipe.json`, `README.md`, `run.sh`, and `env.example`. Edit the generated files before marking **`verified`**.

Manifest fields follow schema v1 (`schema/recipe.schema.json`): set `"runtime": "llama-cpp"`, `"id": "llama-cpp/<slug>"`, and keep `"hardware": { "target": "NVIDIA DGX Spark", "gpu": "GB10" }`. Set `"tested_at"` to an ISO date only after you have run the recipe on Spark hardware.

---

## Recipe checklist (llama.cpp on DGX Spark)

Use this as the minimum documentation bar before moving a recipe from **draft** to **verified**. Every item should be answerable from the recipe directory without guessing.

### 1. Source commit and build flags

- [ ] **llama.cpp commit SHA** (full or short) used to compile `llama-cli`, `llama-server`, and any helper binaries.
- [ ] **Build recipe**: CMake generator, `CMAKE_BUILD_TYPE`, and relevant `-D` flags from [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) (e.g. CUDA enabled, native arch flags appropriate for Spark/GB10).
- [ ] **Binary paths** or install prefix documented in `env.example` / README (avoid hard-coding user home paths in `run.sh` without env overrides).

### 2. CUDA backend

- [ ] Confirm the build used **CUDA** (not CPU-only) for GPU offload paths you document.
- [ ] Note any **required CUDA toolkit / driver** versions observed on the test Spark system (observation, not a global guarantee).
- [ ] If you rely on specific backend features (flash attention, graph optimizations), cite the upstream flag or commit that introduced them and show it in your build line.

### 3. GGUF provenance and quantization

Distinguish three revisions—do not conflate them in `recipe.json` or prose:

| Field | Meaning | Example location |
|-------|---------|------------------|
| **`model.repository` + `model.revision`** | Upstream **source weights** on Hugging Face (or other hub) | Manifest `model` object |
| **GGUF artifact** | The **file(s)** llama.cpp actually loads | Recipe README + `env.example` (`GGUF_PATH`, URL, or generation command) |
| **Quantization metadata** | Method (`Q4_K_M`, `Q8_0`, IQ quants, etc.), tool (`llama-quantize`, `convert_hf_to_gguf.py`), and **which source revision was converted** | Recipe README |

For `verified`, pin the source weights to an immutable revision and identify every loaded GGUF by checksum; a mutable repository branch or filename alone is draft-only.

- [ ] Document **how the GGUF was produced** (conversion command or download URL + checksum).
- [ ] State **quant type** and whether weights are split (multi-shard GGUF).
- [ ] If re-quantizing locally, pin the **converter script version** (same llama.cpp commit as binaries).

### 4. Tokenizer and chat template

- [ ] Identify the **chat template** expected by the model (embedded GGUF metadata, `--chat-template`, `--chat-template-file`, or model-card reference) using the flags supported by the pinned llama.cpp commit.
- [ ] Note **special tokens** / tool-call formats if the model uses them; verify template matches the GGUF metadata.
- [ ] Document a **minimal prompt/response smoke shape** in the recipe README (not necessarily a full benchmark).

### 5. Layer offload (`-ngl` / `--n-gpu-layers`)

- [ ] Record the **`--n-gpu-layers`** (or equivalent) value used on GB10 and what happens at `0` vs full offload.
- [ ] Note any **partial offload** tradeoffs you observed (latency vs memory)—observations from your run, not generic claims.
- [ ] If CPU fallback layers are required, explain why (OOM, unsupported ops, etc.).

### 6. Context, batch, and thread settings

- [ ] **`--ctx-size`** (or `-c`) and rationale relative to model defaults and Spark memory.
- [ ] **`--batch-size` / `-b`** and **`--ubatch-size`** if used.
- [ ] **CPU thread count** (`-t`) when relevant for partial CPU work.
- [ ] Any **`--rope-*`** or YaRN scaling flags for extended context—cite upstream docs.

### 7. Unified memory observations (Spark / GB10)

DGX Spark uses a **unified memory** model; document what you measured rather than assuming desktop discrete-GPU behavior.

- [ ] Peak **resident memory** during load + steady decode (how you measured: `nvidia-smi`, `/proc`, etc.).
- [ ] Behavior when approaching memory limits (OOM killer, graceful error, partial CPU offload).
- [ ] Record the pinned release's **load mode** (`--load-mode` in current builds, or its older-release equivalent) and whether mmap/direct I/O behavior affected startup or peak usage.

### 8. CLI vs server mode

- [ ] State whether the entrypoint is **`llama-cli`** (one-shot / interactive) or **`llama-server`** (HTTP).
- [ ] For server recipes: document **bind address**, port env vars, and link to [server tool docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
- [ ] For OpenAI-compatible routes, list which **API endpoints** you validated (e.g. `/v1/chat/completions`) without claiming full feature parity unless tested.

### 9. Benchmark methodology

Only record numbers you actually ran on Spark; store methodology in the recipe README so others can reproduce.

- [ ] **Hardware snapshot**: Spark system, driver, llama.cpp commit, GGUF variant.
- [ ] **Load procedure**: warm-up passes, concurrent clients (if any), prompt length distribution.
- [ ] **Metrics**: tokens/s (prefill vs decode if separated), time-to-first-token, batch size, context length.
- [ ] **Repro command**: exact flags/env (can live in a commented block or separate `bench.sh`—still **draft** until executed).

### 10. Output validation

- [ ] **Functional**: expected completion shape (stop tokens, max tokens, grammar/JSON if used).
- [ ] **Quality spot-check**: short fixed prompts and what you looked for (not a substitute for full eval).
- [ ] **Regression hook**: optional hash or snapshot of outputs for CI-local manual re-check (this repo does not run inference in CI by default).

---

## Recipes in this lane

| Slug | Intent | Status |
|------|--------|--------|
| [`qwen38-flash-next-ud-iq4-xs`](qwen38-flash-next-ud-iq4-xs/) | **Public default.** Unsloth `UD-IQ4_XS` GGUF, unpatched `run.sh` (~25 tok/s short, 5.60 at 229k) | `draft` |
| [`qwen38-flash-next-ud-iq4-xs-qsa`](qwen38-flash-next-ud-iq4-xs-qsa/) | Experimental QSA kernel patch + locked hashes. Fail-closed; does not replace the default | `draft` |

---

## Lane layout

```
recipes/llama-cpp/
├── README.md                 ← this guide (not a recipe; excluded from validation)
└── <slug>/
    ├── recipe.json           ← manifest (schema v1)
    ├── README.md             ← operator notes for this recipe
    ├── run.sh                ← entrypoint (must fail closed while draft)
    └── env.example           ← paths, ports, GGUF_LOCATION, etc.
```

Validate populated recipes from the repo root when ready:

```bash
python3 scripts/validate_recipes.py
```

Fix all errors before setting `"status": "verified"`.
