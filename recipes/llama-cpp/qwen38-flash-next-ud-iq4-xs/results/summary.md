# Local UD-IQ4_XS benchmark summary

> Unpatched recipe measurements on one DGX Spark. Status remains `draft`.
> Experimental QSA kernel numbers are in [`qsa-kernels.md`](qsa-kernels.md)
> and must not be mixed with the ~25 tok/s short-prompt figure below.

Measured on 2026-08-27 using one NVIDIA DGX Spark (GB10), llama.cpp PR #27742 commit `250b61446`, CUDA 13.0.2, driver 580.159.03, and Unsloth `UD-IQ4_XS` at revision `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f`.

## Artifact integrity

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `00001-of-00003` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `00002-of-00003` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `00003-of-00003` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

Total: **93,682,584,224 bytes**. Shard 1 starts with `47 47 55 46` (`GGUF`).

## Selected configuration

```text
per_layer_token_embd=CPU
load-mode=mmap
tensor-read-lazy=on
all normal layers on CUDA
threads=12
batch=512
ubatch=128
flash attention=on
KV=f16
parallel=1
prompt cache=off
speculation=off
```

The 51B n-gram embedding table is still used. Pinning `per_layer_token_embd` to CPU with lazy mmap changes how its rows are fetched; it does not disable PLE.

## Safety result

Every load ran under `tools/spark_guard.py` with an 80 GiB start requirement, 36 GiB soft-stop floor, 28 GiB hard-kill floor, and 1 GiB maximum swap growth.

The complete 262,144-token allocation and 229,874-token prompt finished successfully. During that run:

- Preflight `MemAvailable`: **112.48 GiB**
- Minimum `MemAvailable`: **39.53 GiB**
- Maximum swap used: **0.66 GiB**
- Guard stop reason: operator interrupt after completion

This leaves only 3.53 GiB above the soft-stop floor. The native context is a demonstrated boundary, not a recommended default. The launcher therefore defaults to 4,096 tokens.

## Configuration sweep at short prompt

Prompt cache was disabled. Values are medians of five measured requests after one warmup.

| Configuration | TTFT | Decode tok/s | Finding |
|---|---:|---:|---|
| fit auto, t12, b512, ub64 | 0.679 s | 24.969 | baseline |
| all GPU, t12, b4096, ub64 | 0.687 s | 22.726 | larger logical batch hurt decode |
| all GPU, t12, b512, ub64 | 0.691 s | **25.114** | best decode median |
| all GPU, t20, b512, ub64 | 0.695 s | 24.831 | extra CPU threads hurt |
| all GPU, t12, b512, ub128 | **0.551 s** | 24.942 | selected; ~20% lower TTFT |

## Context allocation sweep with a short prompt

These rows measure the cost of allocating a larger context while processing the same 76-token prompt. They do **not** represent decode at that depth.

| Allocated context | Median TTFT | Median decode tok/s |
|---:|---:|---:|
| 1,024 | 0.551 s | 24.942 |
| 4,096 | 0.550 s | 25.060 |
| 16,384 | 0.542 s | 25.010 |
| 65,536 | 0.575 s | 22.676 |
| 131,072 | 0.561 s | 24.880 |
| 262,144 | 0.555 s | 25.226 |

The 65K dip was not monotonic and should not be treated as a context-scaling curve.

## Actual prompt-depth sweep

Deterministic prompts were generated with seed `380051` and measured through the server tokenizer. Each row is one cold, prompt-cache-disabled request with 64 output tokens. See `tools/generate_context_prompts.py` and `raw/depth-sweep-f16.jsonl`.

| Prompt tokens | TTFT / prompt completion | Decode tok/s |
|---:|---:|---:|
| 898 | 2.91 s | 21.05 |
| 3,970 | 12.38 s | 22.69 |
| 16,258 | 41.92 s | 20.73 |
| 32,642 | 86.85 s | 17.62 |
| 65,409 | 193.09 s | 11.35 |
| 130,946 | 538.32 s | 8.16 |
| 229,874 | 1,218.85 s | **5.60** |

For the 229,874-token row, llama-server reported 188.85 prompt tok/s and 5.60 generation tok/s. This is the honest long-context curve; short-prompt decode near 25 tok/s is not sustained at deep context.

## Varied task-shape baseline, speculation off

The literal `@` in each prompt is replaced with a different deterministic identifier for every run. This prevents repeated-request output learning from masquerading as speed. Values are medians of three measured requests after one warmup.

| Task | Prompt tokens | TTFT | Decode tok/s |
|---|---:|---:|---:|
| Reproduce edited module | 429 | 1.496 s | 24.039 |
| Focused function fix | 185 | 0.912 s | 24.124 |
| Add a new function | 257 | 1.081 s | 24.285 |
| Original technical prose | 130 | 0.674 s | 23.639 |

This supports an honest general-purpose headline of approximately **24–25 tok/s**, not the copy-only speculative headline from another repository.

## Parallel-two concurrency probe

Configuration: `--parallel 2`, total context 8,192, approximately 4,096
tokens per slot, two synchronized fixed prompts, 64 output tokens each,
speculation off.

The first launch was deliberately stopped before model readiness when swap
growth crossed an overly strict 0.5 GiB limit. No model error occurred. The
second launch used the same 45/38 GiB memory floors with a 1 GiB swap-growth
limit and completed three measured concurrent batches.

| Metric | Request alpha | Request beta | Pair aggregate |
|---|---:|---:|---:|
| First-batch TTFT | 2.444 s | 2.443 s | — |
| First-batch decode | 17.741 tok/s | 17.737 tok/s | 21.349 output tok/s |
| Median TTFT, 3 batches | 0.853 s | 0.853 s | — |
| Median decode, 3 batches | 20.691 tok/s | 20.678 tok/s | **32.824 output tok/s** |
| Maximum start skew | — | — | 0.288 ms |

Both requests succeeded in all three batches with no API errors. Each request
produced one stable, nonempty output hash across all repetitions:

- alpha: `912ec93d10435ab11a26dabbeaea9259d94ef82cca443043c3736715cd16ff43`
  (413 output characters)
- beta: `048906166cf180fdd67392ad058e6653bd7029eed6934305462fd7c4f2c9623d`
  (371 output characters)

Server logs showed both slots processing together and all six requests
finishing with 64 generated tokens. There were no QSA/indexer assertions,
server errors, new NVIDIA OOM/Xid/NV_ERR kernel entries, soft stops, or hard
kills. Memory evidence:

- Preflight `MemAvailable`: 112.98 GiB
- Minimum `MemAvailable`: 50.61 GiB
- Maximum swap used: 0.57 GiB

A separate non-streaming verification client stalled locally and did not reach
the server; it is excluded. The synchronized streaming probe and server logs
are the accepted evidence. Concurrency 2 is therefore proven for this bounded
4K-per-slot case. Higher concurrency remains untested.


## PLE lookup research

The “51B n-gram” path is a learned **per-layer token embedding** table, not
llama.cpp speculative decoding. Exact GGUF metadata:

```text
tensor: per_layer_token_embd.weight
shape: [160, 320001536]
type: IQ4_NL
size: 28,800,138,240 bytes (26.82 GiB)
row: 90 bytes
lookups per token: 16 (8 bigram + 8 trigram hash partitions)
```

Each token hashes prior tokens with 16 independent 64-bit multipliers, reduces
modulo ~20M rows per partition, then runs 16 CPU `ggml_get_rows` gathers.
Useful compressed data is about 1.44 KiB/token, but rows are scattered across
the 26.82 GiB shard. A 4 KiB major fault for a 90-byte row is up to ~45× read
amplification.

PR #27742 already marks the tensor lazy, skips `MAP_POPULATE`, applies
`POSIX_MADV_RANDOM` to the PLE range, and leaves 66 pages resident after load.

### Lazy mmap advice A/B — rejected

An isolated clone at commit `250b61446` added a reversible
`LLAMA_MMAP_LAZY_ADVICE={random,normal,sequential}` selector
(`patches/ple-lazy-advice.patch`). Same binary, same short prompt, PLE pages
dropped before each arm. Temperature-0 output hash
`cb7904d8097240a2bc32c77e27c03a924fcb972212566d14487d20d2aa687601` matched all
three arms. No NVIDIA OOM/Xid/`NV_ERR` entries. Guard floors stayed above
50 GiB available; swap stayed under 0.60 GiB.

| Advice | Cold TTFT | Cold decode | PLE pages after load | PLE pages after cold | PLE GiB after cold |
|---|---:|---:|---:|---:|---:|
| `RANDOM` (current) | 1.911 s | 22.168 tok/s | 66 | 2,677 | 0.010 |
| `NORMAL` | 2.169 s | 23.261 tok/s | 4,097 | 164,336 | 0.627 |
| `SEQUENTIAL` | 2.296 s | 22.262 tok/s | 1 | 37,653 | 0.144 |

`NORMAL` readahead pulled ~61× more PLE pages and slowed TTFT. `SEQUENTIAL`
pulled ~14× more pages and slowed TTFT further. The single-run `NORMAL` decode
bump is inside noise and disappears against the earlier unpatched RANDOM
baseline (24.13 tok/s cold / 26.19 tok/s steady). Keep `POSIX_MADV_RANDOM`.
Do not ship the env override.

Remaining untested directions, in the same safety order: fix whole-file
`posix_fadvise(..., POSIX_FADV_SEQUENTIAL)` on lazy GGUF files; thresholded
multithreaded `GET_ROWS` for large prefill gathers; page-sorted/deduplicated
row gathers. Explicit I/O, hot-row caches, and layout/quant redesign stay last.

## Invalid and rejected experiments

### Repeated-prompt `ngram-mod`

An early copy-heavy test reused an identical prompt. Its measured runs rose from 29.26 to 39.12 to 45.34 tok/s as the speculative cache learned the prior output. Those numbers are contaminated and excluded from release claims. A varied spec-on rerun was stopped after the host reboot described below; no spec-on release result exists yet.

### Qwen4Exp graph reuse patch

A current-head adaptation of the MIT patch from `0xBakeer/qwen38-flash-next-spark` reached `graphs reused = 127` during the warmup, then the server segfaulted on the second request. Kernel logs also recorded an NVIDIA `NV_ERR_NO_MEMORY` at the time of this experiment. The patch was removed and must not ship.

### Host reboot boundary

The Spark later rebooted without a clean shutdown. The previous boot contained multiple NVIDIA OOM events from several workloads, including one temporally associated with the graph-reuse experiment. Causality cannot be assigned solely to this recipe, but no additional model loads should be run until the host is treated as recovered and the unsafe patch remains excluded.

## Comparison-repository lessons

Research clone: `0xBakeer/qwen38-flash-next-spark` commit `4c6fc3af429bff5c472511cf965751eac6b7caf2`.

- Its credible free-form llama.cpp result is approximately 27.8 tok/s on a different `UD-Q4_K_XL` quant.
- Its 52–88 tok/s figures are copy-heavy `ngram-mod` workloads and are not general model speed.
- Its own later measurements contradict the earlier whole-PLE prewarming claim; this recipe does not prewarm the 26.8 GiB table.
- The old graph-reuse patch does not safely apply to the current QSA layout.
- Prompt cache, repeated prompts, speculative-cache learning, and short-prompt context allocation must be labeled separately.

The comparison repository is MIT licensed, Copyright (c) 2026 0xBakeer. Its methodology informed the varied task suite and rejected graph-reuse experiment; no third-party code is shipped here.

## Raw evidence

- `raw/ctx1024-nocache-v1.jsonl`: honest short-prompt baseline
- `raw/ctx1024-opt*.jsonl`: runtime tuning sweep
- `raw/ctx*.jsonl`: context-allocation sweep
- `raw/depth-sweep-f16.jsonl`: actual prompt-depth sweep
- `raw/tasks-varied-spec-off.jsonl`: task-shape baseline with per-run prompt variation
- `raw/tasks-ngram-mod.jsonl`: contaminated repeated-prompt pilot; excluded

- `raw/concurrency-np2-ctx8192.jsonl`: three parallel-two request batches
- `raw/concurrency-np2-ctx8192-guard-summary.json`: memory, swap, stop, and kernel-fault evidence
- `raw/ple-baseline-profile.json`: unpatched RANDOM PLE fault/residency profile
- `raw/ple-advice-random.jsonl`: unpatched RANDOM cold+steady timings
- `raw/ple-advice-ab.json`: isolated mmap-advice A/B decision record
- `raw/ple-advice-prototype-random.jsonl`: patched RANDOM arm
- `raw/ple-advice-prototype-normal.jsonl`: patched NORMAL arm
- `raw/ple-advice-prototype-sequential.jsonl`: patched SEQUENTIAL arm
- `patches/ple-lazy-advice.patch`: rejected env-selector prototype
Raw JSONL in this directory is the unpatched recipe evidence. Kernel-track
timings are documented in [`qsa-kernels.md`](qsa-kernels.md), not these files.
