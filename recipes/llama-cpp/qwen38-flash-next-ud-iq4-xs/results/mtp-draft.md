# Experimental MTP draft (`draft-mtp`, n-max 3)

> **Not the recipe default.** `run.sh` still launches unpatched AR with
> `SPEC_TYPE=none`. This used an isolated tree at
> `/home/djl/llama.cpp-qwen4exp-mtp` (commit `250b61446` plus a local port of
> closed llama.cpp [PR #27842](https://github.com/ggml-org/llama.cpp/pull/27842)).
> That PR is **not merged**.

## What was converted

`--mtp --outtype q8_0` from the local
`/home/djl/models/Qwen3.8-Flash-Next-FP8` checkpoint (architectures
`Qwen4ExpForConditionalGeneration`). Converter logs showed the 31 `mtp.*`
tensors as `torch.bfloat16` even though the repo is tagged FP8.

Output:

```text
/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf
```

34 tensors, **3.9 GiB**. Pairs with the existing Unsloth `UD-IQ4_XS` main GGUF;
the main weights were not requantized.

## Launch (measured)

```text
llama-server \
  -m UD-IQ4_XS-00001-of-00003.gguf \
  -c 4096 -np 1 -b 2048 -ub 512 -t 12 \
  -fa on -lm mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU -ngl all -fit off \
  --spec-type draft-mtp \
  -md mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf -ngld 99 \
  --spec-draft-n-max 3
```

Spark guard: 80 / 36 / 28 GiB, swap growth 1 GiB. **Batch default moved to
`-b 2048 -ub 512` on 2026-08-29** (see recipe `run.sh`); numbers below use that
unless noted.

## Cold depth curve (`draft-mtp`, `b2048` / `ub512`, 2026-08-29)

Same deterministic `ctx*.txt` prompts and cold-server protocol as
[`summary.md`](summary.md). Tree: `/home/djl/llama.cpp-qwen4exp-kmtp/build/bin/llama-server`
with `LD_LIBRARY_PATH=/home/djl/llama.cpp-qwen4exp-kmtp/build/bin` (required for
draft load). Draft: `mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf`, `-ngld 99`,
`--spec-draft-n-max 3`.

| Target ctx | Prompt tokens | TTFT | Prefill tok/s (server) | Decode tok/s (client) |
|---:|---:|---:|---:|---:|
| 4,096 | 3,955 | 10.85 s | 370.09 | 29.84 |
| 16,384 | 16,243 | 40.32 s | 405.22 | 26.83 |
| 32,768 | 32,627 | 78.68 s | 416.49 | 23.99 |
| 65,536 | 65,395 | 167.47 s | 391.93 | 8.97* |

**vs unpatched AR (`summary.md`, same protocol):** at **4k**, MTP prefill on this
day (**370** tok/s) exceeded the unpatched verification rerun (**318** tok/s).
**16k and 32k prefill are 8.4% and 6.7% below** unpatched (405 vs 443, 416 vs
447 tok/s), outside the plan's 5% bound — expect draft overhead at depth, not
parity. 64k prefill is within ~3% but the client decode row is unreliable (see
footnote).

\*The 11-token stop is server-side and batch-sensitive — same binary/draft/prompt at `-b 512 -ub 128` (09:57 run, `/tmp/kmtp-qsa64.jsonl` on Spark) produced 64 tokens at 33.3% accept; at `-b 2048 -ub 512` accept collapses to 3.7% and the target output diverges. Treat **167 s TTFT** and **~392 tok/s prefill** as the reliable 64k-row signals, not the decode figure.

### `draft-mtp` + `ngram-mod` on copy-heavy task (labeled separately)

`--spec-type draft-mtp,ngram-mod` on `prompts/tasks/reproduce-module.txt` with
`--variation-placeholder @` (three measured runs after one warmup). **Not comparable
to prose or depth rows.**

| Metric | Median |
|---|---:|
| TTFT | 1.113 s |
| Decode tok/s | **79.86** |
| Draft accept (per-run server logs) | 0.67–1.00 |

## Protocol

- Host: one DGX Spark GB10, CUDA 13.0.2, driver 580.159.03
- Chat completions, `temperature=0`, `max_tokens=51`, `thinking=false`
- Prompt: `Continue this sequence exactly:` plus `1 2 3 … 20`
- This is **not** the kernel-track hash protocol (`2689367b205c16ce`)

## Results (2026-08-28)

Server `slot print_timing` decode (`eval time`, 51 completion tokens):

| Run | Decode tok/s | ms/token | Draft accept | Mean draft len |
|---|---:|---:|---:|---:|
| warmup | 35.00 | 28.57 | 75.556% (34/45) | 3.27 |
| 1 | 37.14 | 26.93 | 75.556% (34/45) | 3.27 |
| 2 | **40.86** | 24.47 | 75.556% (34/45) | 3.27 |
| 3 | 40.48 | 24.71 | 75.556% (34/45) | 3.27 |

Steady median **~40.5 tok/s**. Unpatched AR on this GGUF is **~25 tok/s** on the
recipe short-prompt protocol → about **1.6×**, in line with the PR's Strix Halo
median 1.63× at n-max 3. This ctx-4096 figure is not comparable to the
64k QSA-kernel **18.73 tok/s** result; context, prompt, and binary differ.

Client wall-clock (includes TTFT) was ~28 tok/s on the same three runs.

## Kernel + MTP composition (2026-08-28)

The `graph_mtp` port was also rebased onto the QSA-kernel tree into a combined
build at `/home/djl/llama.cpp-qwen4exp-kmtp`, to test whether the draft's ~1.6×
survives where the kernels actually win.

| Config | Decode | Accept | Mean len | Note |
|---|---:|---:|---:|---|
| 4k, chat temp 0 / 51 tok | **44.6 tok/s** | 83.3% (35/42) | 3.50 | kernels + MTP compose; faster than isolated ~40.5 |
| 229,859 depth prompt | **10.2 tok/s** | **43.2% (35/81)** | 2.25 | slower than kernel AR 11.55 tok/s |

At 229k, acceptance collapses to 43% and the draft's own per-step full-context
gather makes MTP a net loss vs autoregressive. The 229k run used `-c 237568`
with draft KV `q8_0` (`-ctkd q8_0 -ctvd q8_0`); default `f16` draft KV at that
context does not fit under the 36 GiB guard floor. At `-c 262144` even `q8_0`
draft KV (~16 KB/token) tripped the floor during load. Bottom line: MTP is a
short-context win, not the long-context answer.

## Why acceptance is low

Same `draft-mtp` binary, same deterministic-words prompt family (seed 380051).
Draft **weights** are always `mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf` (`--outtype
q8_0`). Draft **KV cache** uses llama.cpp defaults (`-ctkd`/`-ctvd` unset →
`f16`) except at 229k, where `f16` draft KV does not fit under the 36 GiB guard
floor and the run used `-ctkd q8_0 -ctvd q8_0`.

| Words prompt | Draft KV | Accept | Mean len |
|---:|---|---:|---:|
| 4k | `f16` (default) | 54.9% (39/71) | 2.62 |
| 32k | `f16` (default) | 63.1% (41/65) | — |
| 64k | `f16` (default) | 33.3% (31/93) | — |
| 229k | `q8_0` (memory) | 43.2% (35/81) | 2.25 |

The accept curve is **non-monotonic in depth** (32k > 4k > 64k), so the drop is
driven by *where the deterministic sequence is cut*, not by context length.
Before 2026-08-29, `graph_mtp` ran dense attention while the trunk ran indexer
top-k (see `build_layer_attn(…, nullptr, …)`). That mismatch did not translate
into a smooth accept decay — and wiring the draft's own QSA indexer back in
(verified below) still leaves accept **unchanged** at 4k and 64k.

One clear effect survives:

- **Task (~30 points).** `Continue 1 2 3 …` is 83.3% at 4k; deterministic words
  hover ~33–63% (content-dependent). The MTP head agrees with the trunk far
  more on a memorized continuation than on arbitrary text.

Consequence: accept is content-bound, not fixed by sparse-vs-dense draft attention.
The draft's own QSA indexer was wired back (see below); accept did not move.

## QSA indexer wiring (verified 2026-08-29, build12)

The draft GGUF carries its own QSA indexer (`blk.48.indexer.*`, 512 expert
weights). Wiring it back into hybrid-idx is **verified end-to-end** in
`/home/djl/llama.cpp-qwen4exp-kmtp` (uncommitted). Main and draft GGUFs are
restored on Spark under `~/models/Qwen3.8-Flash-Next-UD-IQ4_XS/`.

Changes (scoped to `QWEN4EXP && MTP`; other MTP drafts keep plain KV):

- **A** — remove `LLM_ARCH_QWEN4EXP` from `mtp_on_hybrid_qwen` so the draft keeps
  hybrid-idx memory.
- **B** — filter flip: `mtp_draft` → layers `il >= n_layer()` for attn/recr/idx.
- **C** — `graph_mtp` uses `build_inp_mem_hybrid()` + `mctx_hyb` from
  `inp_mem->mctx`, like the trunk.
- **D** — `llm_graph_input_mem_hybrid::set_input`: guard recurrent `s_copy` on
  `inp_rs->s_copy->buffer` (not `n_rs > 0`; that is sequence count, not layer
  count). Without this, load-time SIGSEGV / `GGML_ASSERT(buffer)` in
  `set_input`.
- **E** — `llama_memory_hybrid{,_idx}::seq_rm`: skip recurrent `seq_rm` when the
  draft has no recurrent tensors (`has_layer_tensors()`). Without this,
  post-decode cleanup aborts: `failed to remove sequence 0 with p0=3956`.

Earlier crash chain (same tree, superseded by A–E):

1. Naive `mctx_hyb` pass-through → SIGSEGV `get_n_stream` (plain KV cast).
2. `build_attn_inp_kv()` in `graph_mtp` → SIGSEGV `build_input_v_idxs` (wrong
   downcast for hybrid-idx).

### Dense vs QSA-wired draft (deterministic-words, seed 380051)

Same prompts, `temperature=0`, `max_tokens=64`, `n-max=3`, spark guard
80/36/28 GiB. Same draft GGUF and default draft KV `f16` (no `-ctkd`/`-ctvd` on
either side). Client metrics from `stream_benchmark.py`; accept from server
`draft acceptance`.

| Context | Config | Draft KV | Accept | Decode tok/s | TTFT |
|---:|---|---:|---:|---:|---:|
| 4k words | Dense MTP (baseline) | `f16` | 54.9% (39/71) | 26.6 | 15.3 s |
| 4k words | QSA-wired (build12) | `f16` | 54.9% (39/71) | 31.0 | 12.8 s |
| 64k words | Dense MTP (baseline) | `f16` | **33.33%** (31/93) | 16.7 | 234.0 s |
| 64k words | QSA-wired (build12) | `f16` | **33.33%** (31/93) | 17.3 | 232.4 s |

Accept is **identical** at both depths; small decode/TTFT deltas are run noise, not
an accept lever. **Further accept chasing is parked** — the curve stays
content/cut-point dominated (32k baseline 63.1% unchanged; QSA-wired 32k not
remeasured).

## Entropy-gated MTP draft length (`--spec-draft-p-min`, 2026-08-29)

The kmtp tree supports `--spec-draft-p-min` (default `0.0`; `common/speculative.cpp:798`), which halts draft token generation when candidate top-1 probability $p < p_{\min}$.

Measured on `ctx4096.txt` (deterministic words, seed 380051), `temperature=0`, `max_tokens=64`, `n-max=3`, under `spark_guard.py` (80/36/28 GiB):

| Config (`--spec-draft-p-min`) | Output Tokens | TTFT | Decode tok/s (client) | Server Eval Time | Draft Accept (server) | Finding |
|---|---:|---:|---:|---:|---:|---|
| `0.0` (baseline, ungated) | 64 | 13.04 s | **28.38 tok/s** | 30.62 ms/tok | **66.67% (42/63)** | Full $N=3$ draft chains generated; steady MTP speedup. |
| `0.6` (moderate gate) | 3* | 11.55 s | 13.45 tok/s | 71.40 ms/tok | — | Gate halts drafting immediately on high-entropy text ($p < 0.60$). |
| `0.9` (strict gate) | 3* | 12.07 s | 12.54 tok/s | 75.18 ms/tok | — | Gate halts drafting on nearly every token ($p < 0.90$). |

\*On arbitrary/high-entropy text, top-1 token probability is frequently $<0.60$. With $p_{\min} \ge 0.60$, the gate correctly suppresses low-confidence draft tokens, but in the current kmtp hybrid-memory implementation, early draft termination exposes non-consecutive position warnings and halts generation early. `--spec-draft-p-min 0.0` remains the required setting for `draft-mtp`. Evidence: `raw/mtp-pmin/`.

## MTP draft sampling & temperature interaction (2026-08-29)

Tested on `ctx4096.txt` (deterministic words, seed 380051), `max_tokens=64`, `n-max=3`, under `spark_guard.py` (80/36/28 GiB):

| Target Request | Draft Sampler Mode | TTFT | Decode tok/s (client) | Server Eval Time | Draft Accept (server) | Finding |
|---|---|---:|---:|---:|---:|---|
| Greedy ($T=0.0$) | Backend sampling (`--spec-draft-backend-sampling`) | 11.87 s | **33.17 tok/s** | 30.55 ms/tok | **66.67% (42/63)** | Greedy baseline; accurate draft alignment. |
| Stochastic ($T=0.7$) | Backend sampling (`--spec-draft-backend-sampling`) | 11.56 s | **19.88 tok/s** | 51.04 ms/tok | **32.29% (31/96)** | GPU sampler chain mismatch drops accept by >50%. |
| Stochastic ($T=0.7$) | CPU sampler (`--no-spec-draft-backend-sampling`) | 12.45 s | **34.84 tok/s** | 29.10 ms/tok | **72.88% (43/59)** | CPU sampler aligns with target sampling; +75% speedup vs backend sampling. |

**Key Finding:** When serving stochastic requests ($T > 0$), GPU backend draft sampling causes severe rejection penalties (acceptance drops to 32.3%). Disabling backend draft sampling with `--no-spec-draft-backend-sampling` restores high acceptance (72.9%) and lifts decode throughput to **34.84 tok/s** (+75% speedup). Evidence: `raw/mtp-sampler/`.

## Do not

- Do not use `--spec-draft-n-max 8` (PR: slower than AR; rollback-slot cost).
- Do not expect QSA indexer wiring alone to raise accept — measured identical to
  dense at 4k and 64k.
- Do not mix with `ngram-mod` copy-learning numbers.
- The upstream llama.cpp MTP PR remains closed/unmerged; this track is local-only
  on `llama.cpp-qwen4exp-kmtp`.
- Do not run kmtp at 64k with `-ub 512` until the hybrid-memory ubatch interaction is fixed.

## Parked post-gap work (not implemented)

From the 2026-08-29 TTFT-gap plan; **not shipped**:

1. **Zero-copy PLE gather** (main trunk) — see [`summary.md`](summary.md) §
   Parked post-gap work. Required for further honest prefill at depth without
   abandoning lazy CPU PLE.
2. **MTP draft-layer / accept chasing** — QSA indexer wiring into `graph_mtp`
   did not move accept at 4k/64k; further draft-graph edits are parked.
