# Local UD-IQ4_XS benchmark summary

> Unpatched recipe measurements on one DGX Spark. Status remains `draft`.
> Experimental QSA kernel numbers are a sibling config:
> [`../../qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md`](../../qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md).

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
batch=2048
ubatch=512
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
| all GPU, t12, b512, ub128 | **0.551 s** | 24.942 | former default; ~20% lower TTFT than b512/ub64 |

## Batch-size default change and microbatch sweep (2026-08-29)

`run.sh` now defaults to `-b 2048 -ub 512`. On the 76-token short prompt (one warmup,
five measured runs, unpatched tree), median steady-state decode was **28.597 tok/s**
versus **28.302 tok/s** at the previous `-b 512 -ub 128` default (within 10%; decode
regression check passed). Median TTFT on those measured runs was **0.149 s** (warm
server after load).

### Microbatch scaling comparison (`-b 2048`)

| Microbatch (`-ub`) | Short decode (76 tok) | Short warm TTFT | Cold 4k TTFT (3,913 tok) | 4k prefill tok/s | 4k decode tok/s |
|---|---:|---:|---:|---:|---:|
| `-ub 256` | 25.45 tok/s | 0.152 s | 12.47 s | 316.8 tok/s | 22.77 tok/s |
| `-ub 512` (default) | 28.60 tok/s | 0.149 s | 12.65 s | 318.1 tok/s | 25.01 tok/s |
| `-ub 1024` | **28.74 tok/s** | 0.152 s | **8.22 s** | **481.3 tok/s** | **25.18 tok/s** |

`-ub 1024` yields a **35% reduction in cold 4k TTFT** (8.22 s vs 12.65 s) by lifting prompt prefill from ~318 to **~481 tok/s** on GB10, while keeping short decode at **~28.7 tok/s**. Evidence: `raw/ubatch-sweep/`. **Caveat (2026-08-30):** the clean-tree revalidation (`raw/ple-residency/`) stopped at the pre-decided hash gate — at `-ub 1024` the 4k output hash is `06124a4b`, not the `-ub 512`-era reference `99a15d5b` (ubatch-split numerics divergence, not binary drift — the same binary matched `99a15d5b`/`b641e2eb` at `-ub 512` minutes earlier). The 8.22 s figure therefore stays an era-2026-08-29 result, not a clean-tree validated one; establish a per-ubatch hash baseline before reusing the digest as a cross-ubatch control gate.


### Deep-context microbatch scaling (`-ub 1024` at 64k / 128k)

Measured 2026-08-29 on fresh unpatched servers per depth with per-depth context allocation (`-c $((CTX + 256))`), `max_tokens=64`, prompt family `ctx*.txt` (seed `380051`).

| Target ctx | Prompt tokens | `-ub 512` TTFT | `-ub 512` prefill tok/s | `-ub 1024` TTFT | `-ub 1024` prefill tok/s | `-ub 1024` decode tok/s | TTFT Delta | Prefill Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 3,955 | 12.65 s | 318.1 tok/s | **8.22 s** | **481.3 tok/s** | 25.18 tok/s | **-35.0%** | **+51.3%** |
| 65,536 | 65,395 | 162.46 s | 404.18 tok/s | **163.53 s** | **~400 tok/s** | 19.96 tok/s | +0.7% | -1.0% |
| 131,072 | 130,931 | 538.32 s* | ~243 tok/s* | **386.77 s** | **339.47 tok/s** | 16.32 tok/s | **-28.1%** | **+39.7%** |

\*128k comparison point from the 2026-08-27 `b512`/`ub128` baseline sweep (538.32 s TTFT / 8.16 tok/s decode). At 64k, TTFT is essentially flat (+0.7%) as PLE memory/IO bottlenecks dominate over chunk launch batching. At 128k, `-ub 1024` cuts TTFT by 151.6 s (**28.1% reduction**) and lifts prefill from ~243 to **339.5 tok/s**. Evidence: `raw/deep-ub1024/`. Subject to the same 2026-08-30 hash-gate caveat above.

### PLE residency Gate 0 — cold-prefill disk reads (2026-08-30, lazy on)

True-cold arms (120 GiB page-cache eviction before each load; verified by ~51 GiB load-window `pgpgin`), request-window disk reads by three estimators:

| Depth | vmstat `bi` | iostat `rkB/s` | pgpgin | pgmajfault | Hash |
|---:|---:|---:|---:|---:|---|
| 4k | 443 MB | 443 MB | 405 MB | 55,669 | `99a15d5b` ✓ |
| 64k | 1,098 MB | 1,085 MB | 2,152 MB | 458,895 | `b641e2eb` ✓ |

Both depths clear the pre-decided 100/500 MB materiality gates: Grok's disk-bound cold prefill **reproduces on GB10** — the lazy PLE table demand-faults a material fraction of its amplified unique-row set from NVMe during prefill (4k TTFT 10.79 s / 372.5 tok/s; 64k 170.7 s / 384.7 tok/s; guard floors held, min ≥ 43 GiB). The `--tensor-read-lazy off` / surgical-PLE-populate A/B (Step 3) was originally halted by Step 2's hash gate — **the Step 3 reopen (below) has since run and closed the axis**. Evidence: `raw/ple-residency/`.

### PLE residency Step 3 reopen — `--tensor-read-lazy off` A/B (2026-08-30, `no-win`)

True-cold A/B at both depths, flags identical to Gate 0 except `--tensor-read-lazy off` (whole-GGUF `MAP_POPULATE` at load — load-window reads 87.5/117.8 GB, load→health +31–70 s). Outputs byte-identical (`99a15d5b`/`b641e2eb`), guard floors held (min 47.26/46.05 GiB, swap ≤ 0.23 GiB, no hard kill):

| Arm | TTFT | vs Gate 0 | prompt-eval | request-window reads (bi / rkB/s / pgpgin) |
|---|---:|---:|---:|---|
| `lazyoff-4k` | 22.674 s | **+110%** | 176.0 tok/s (was 372.5) | **~10.2 GB** all three (was 0.40–0.44 GB) |
| `lazyoff-64k` | 168.323 s | −1.4% (sub-gate) | 390.2 tok/s (was 384.7) | **~21.9 GB** all three (was 1.1–2.2 GB) |

**Verdict `no-win` — axis closed.** MAP_POPULATE populated the GGUF at load, but GB10
unified-memory reclaim evicted the clean mapped file pages once the request's
KV/activation working set landed, so prefill re-faulted the weights from NVMe at **25×
(4k) / 10× (64k)** the lazy-on disk traffic — 4k TTFT doubled. Lazy-on +
`POSIX_MADV_RANDOM` (fault only accessed PLE rows) is the correct design on this host;
whole-file populate must not be pursued (mlock out of scope). Step 2' (`-ub 1024`
revival) and the composition run were not run — gated on a 4k Step 3 win, which did not
occur — so the 2026-08-29 `8.22 s / 481 tok/s` `-ub 1024` figures **remain
era-2026-08-29** (per-ubatch hash baseline `06124a4b` established, but the revival arm
was never needed). Evidence: `raw/ple-residency/`.

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

## Cold depth curve (`b2048` / `ub512`, 2026-08-29)

Deterministic prompts from `/tmp/qwen38-context-prompts/ctx{4096,16384,32768,65536}.txt`
(seed `380051`). **One fresh unpatched server per depth**, one cold request each,
`max_tokens=64`, `warmup-count=0`, `repetitions=1`. Client metrics from
`stream_benchmark.py`; prefill tok/s from server `prompt eval time` on that cold
request.

| Target ctx | Prompt tokens | TTFT | Prefill tok/s (server) | Decode tok/s (client) |
|---:|---:|---:|---:|---:|
| 4,096 | 3,955 | 12.65 s | 318.13 | 25.01 |
| 16,384 | 16,243 | 37.06 s | 442.51 | 23.84 |
| 32,768 | 32,627 | 73.42 s | 446.64 | 21.97 |
| 65,536 | 65,395 | 162.46 s | 404.18 | 19.99 |

**4k verification rerun (2026-08-29):** fresh
`llama-server.unpatched-250b61446`, same cold protocol; the locked row is this
rerun (**318.13 tok/s** server prefill, **12.65 s** TTFT, **~3.2 ms/token**
wall-clock). An earlier same-day pass reported **343.8 tok/s** / **11.73 s**
TTFT. Both sit **below** the plan bar (≥380 tok/s prefill; session case G on
the same binary was **~381 tok/s** at 4k, `/tmp/prefill-G-server.log`).
**TTFT did not materially improve** vs the 2026-08-27 `b512`/`ub128` cold depth
point (**12.38 s** at ~3,970 tokens, **~3.2 ms/token** wall-clock). The
`b2048`/`ub512` default change was validated on the **76-token** short
regression (decode + warm TTFT); it did not materially move cold 4k wall-clock
TTFT. Short-prompt decode near **29 tok/s** (`b2048`/`ub512`, 76-token prompt)
is not sustained here (**~25 tok/s** in the locked 4k row).

### Earlier full-depth sweep (`b512` / `ub128`, 2026-08-27)

See `raw/depth-sweep-f16.jsonl` for the original token-ladder through 229,874
tokens. The 229,874-token row remains **5.60 tok/s** decode and **1,218.85 s**
TTFT; that long tail is unchanged by the batch default.

## Varied task-shape baseline, speculation off (`b512` / `ub128`, 2026-08-27)

The literal `@` in each prompt is replaced with a different deterministic identifier for every run. This prevents repeated-request output learning from masquerading as speed. Values are medians of three measured requests after one warmup. **Not remeasured** after the `b2048`/`ub512` default change.

| Task | Prompt tokens | TTFT | Decode tok/s |
|---|---:|---:|---:|
| Reproduce edited module | 429 | 1.496 s | 24.039 |
| Focused function fix | 185 | 0.912 s | 24.124 |
| Add a new function | 257 | 1.081 s | 24.285 |
| Original technical prose | 130 | 0.674 s | 23.639 |

**Do not blend with current defaults:** these rows are **`b512`/`ub128`,
2026-08-27 only** — still **~24 tok/s** decode on varied tasks. The
**2026-08-29** `b2048`/`ub512` change was measured on the **76-token**
`prompts/short.txt` regression only (**~29 tok/s** decode, **0.15 s** warm
TTFT); task-shape prompts were **not remeasured**. Copy-heavy `draft-mtp` +
`ngram-mod` on reproduce-module reached **~80 tok/s** decode (three varied
runs, 2026-08-29); label separately in [`results/mtp-draft.md`](mtp-draft.md).

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
4K-per-slot case.

## Parallel-four concurrency probe (2026-08-29)

Configuration: `--parallel 4`, total context 16,384 (~4,096 tokens per slot),
four synchronized fixed prompts, 128 output tokens each, speculation off,
under `spark_guard.py` (80/36/28 GiB).

| Metric | 4-way Aggregate |
|---|---:|
| Batch wall-clock | **11.42 s** |
| Successful requests | **4 / 4** |
| Aggregate decode throughput | **44.85 tok/s** |
| Per-slot median decode | **13.60 tok/s** |
| Per-slot median TTFT | **1.945 s** |
| Minimum `MemAvailable` | **47.5 GiB** (well above 36 GiB floor) |
| Maximum swap used | **0.65 GiB** (under 1.0 GiB limit) |

All four slots launched and processed concurrently without assertions, server
errors, or guard warnings. Aggregate decode throughput scales to **~45 tok/s**
(vs 32.8 tok/s at `np=2` and ~29 tok/s single-stream). Evidence: `raw/concurrency-np4/`.

## Parallel-eight concurrency probe (2026-08-29)

Configuration: `--parallel 8` (`-np 8`), total context 32,768 (~4,096 tokens per slot), `-b 2048 -ub 512`, eight synchronized fixed prompts, 128 output tokens each, speculation off, under `spark_guard.py` (80/36/28 GiB).

| Metric | 8-way Aggregate |
|---|---:|
| Batch wall-clock | **15.36 s** |
| Successful requests | **8 / 8** |
| Aggregate decode throughput | **66.67 tok/s** |
| Per-slot median decode | **10.39 tok/s** |
| Per-slot median TTFT | **3.034 s** |
| Minimum `MemAvailable` | **45.39 GiB** (well above 36 GiB floor) |
| Maximum swap used | **0.67 GiB** (under 1.0 GiB limit) |

All eight slots launched and processed concurrently without assertions, server errors, or guard warnings. Aggregate decode throughput scales to **~66.7 tok/s** (vs 44.85 tok/s at `np=4`, 32.82 tok/s at `np=2`, and ~29 tok/s single-stream) with per-slot decode remaining at **~10.4 tok/s**. Evidence: `raw/concurrency-np8/`.

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
### Multithreaded PLE index computation in `set_input` (2026-08-29)

During prompt evaluation (`n_tokens > 1`), computing the PLE row indices via n-gram multiplier mixing in `llm_graph_input_ple::set_input` on a single thread incurs measurable CPU serialization latency. A patch (`patches/ple-multithreaded-set-input.patch`) parallelizes the per-token PLE n-gram mixing loop across 12 worker threads (`std::thread` pool) during prefill ubatches while keeping single-token decode strictly single-threaded without synchronization overhead. (Note: this accelerates host-side index calculation in `set_input`; the backend tensor `ggml_get_rows` gather in `ggml-cpu` already partitions rows across the ggml threadpool).

Acceptance gates and verification on GB10 under `spark_guard.py` (80/36/28 GiB):

1. **Exact Output Hash:** Greedy temperature-0 short prompt (`prompts/short.txt`, 76 tokens, max 128 tokens) preserved the exact reference SHA-256 hash `cb7904d8097240a2bc32c77e27c03a924fcb972212566d14487d20d2aa687601`.
2. **Cold 4k TTFT Improvement at `-ub 512`:** At `-b 2048 -ub 512`, cold 4k TTFT dropped from **12.65 s to 11.68 s** (**7.7% TTFT reduction** / 0.97 s faster), lifting server prefill throughput from **318.1 to 344.2 tok/s** (+8.2%). At `-ub 1024`, TTFT was **10.785 s** (374.1 tok/s prefill); while faster than `-ub 512`, this does not beat the locked unpatched `-ub 1024` baseline (**8.22 s** / 481.3 tok/s), indicating that thread pool dispatch in `set_input` adds scheduling overhead when GPU microbatch launches are already large.
3. **Cold 64k TTFT:** At 64k depth, TTFT was **167.54 s** (391.55 tok/s server prefill, 19.40 tok/s decode). At 64k, disk/memory faults across 26.8 GiB DRAM dominate over index computation.
4. **Memory Guard:** All guard runs passed with minimum `MemAvailable` > 44.2 GiB.

| Configuration | Prompt tokens | TTFT | Server Prefill tok/s | Decode tok/s | Output SHA-256 |
|---|---:|---:|---:|---:|---|
| Short (76 tok) | 76 | 1.361 s | 60.42 | 26.12 | `cb7904d8...` (match) |
| Cold 4k (`-ub 512`) | 3,955 | **11.680 s** | **344.22** | 24.52 | `c64973d8...` |
| Cold 4k (`-ub 1024`) | 3,955 | 10.785 s | 374.11 | 25.62 | `c64973d8...` |
| Cold 64k (`-ub 512`) | 65,395 | 167.539 s | 391.55 | 19.40 | `52733803...` |

Evidence: `raw/ple-mt/`, patch in `patches/ple-multithreaded-set-input.patch`.


### Page-sorted row index gathering in `ggml-cpu` — rejected (2026-08-29)

A prototype in `ggml_compute_forward_get_rows_q` (`ggml/src/ggml-cpu/ops.cpp`) sorted `(row_idx, dst_offset)` pairs in ascending memory order per thread before dequantization, testing whether monotonic DRAM streaming could reduce mmap fault latency across the 26.8 GiB PLE table.

Measured on GB10 under `spark_guard.py` (80/36/28 GiB):

- **Short Hash:** Verified exact reference SHA-256 `cb7904d8097240a2bc32c77e27c03a924fcb972212566d14487d20d2aa687601`.
- **Cold 4k TTFT:** **12.039 s** (335.5 tok/s prefill) — **slower** than the PLE-MT tree baseline (**11.680 s** / 344.2 tok/s, a +3.1% slowdown).
- **Cold 64k TTFT:** **168.465 s** — also regressed vs the PLE-MT tree (**167.539 s**) and unpatched baseline (**162.46 s**).

Per-thread `O(K \log K)` sorting adds CPU overhead without overcoming Linux page-cache/DRAM latency floors. **Rejected and reverted from tree.** Evidence: `raw/ple-pagesort/`.
Remaining untested directions, in the same safety order: fix whole-file
`posix_fadvise(..., POSIX_FADV_SEQUENTIAL)` on lazy GGUF files; thresholded
multithreaded `GET_ROWS` for large prefill gathers; page-sorted/deduplicated
row gathers. Explicit I/O, hot-row caches, and layout/quant redesign stay last.


### nsys profile of cold 4k prefill — decision experiment (2026-08-30)

Full nsys attribution of the **11.785 s** cold 4k prompt eval (PLE-MT build,
`llama-server.ple-mt`, `-b 2048 -ub 512`) over the exact request window:
GPU kernel busy **4.545 s (38.5%)**; **6.62 s (56.1%)** in 13 CPU-side idle gaps
with zero CUDA activity between ubatch bursts; H2D of gathered PLE activations
**7 ms (0.06%)**. Neither optimization branch fires: fused gated residual
(needs GPU >70%, measured 38.5%; elementwise chains are ~9% of TTFT) and
zero-copy PLE (needs PLE gather+H2D >15%, measured ~0.1%) are both **not
indicated**. The dominant cost is per-ubatch CPU serialization from broken
CUDA-graph reuse: the QSA patch tree reports `graphs reused = 0` (rebuild
every ubatch) vs `graphs reused = 7` on unpatched `250b61446` (prompt eval
10.381 s at identical flags). Page-cache warmth accounts for only ~1.3 s of
the gap total (shuffled same-length prompt on a warm server: 8.636 s vs
9.969 s cold). New prioritized axis: QSA-native CUDA-graph-reuse fix.
Evidence: `raw/nsys-4k/`.

### QSA graph-reuse fix — `can_reuse` overrides (2026-08-30)

Added `can_reuse` shape-stability overrides to `llm_graph_input_qsa` and
`llm_graph_input_ple` (they inherited the unconditional-false base default, so
every ubatch on this arch rebuilt the compute graph: `graphs reused = 0`). The
overrides also refresh the per-batch `mctx` before reuse — without it the first
reused step dereferences a destroyed batch's memory context (GDB-captured SIGSEGV;
this and the missing shape checks explain the earlier hard-rejected 0xBakeer
patch's crash). Verified: short hash `cb7904d8` exact; repeat identical request
survives (0xBakeer scenario) at 28.5 tok/s; `graphs reused = 127` per 128-token
decode; **decode ~38.4-41.5 ms/tok vs 43.4-45.2 control (~10-12% faster)**; 4k
output byte-identical to the no-override control. Prefill unchanged by design
(padded n_kv growth correctly forces rebuild each ubatch). Evidence:
`raw/graph-reuse/`, patch `patches/qwen4exp-can-reuse.patch`.
## Parked post-gap work (not implemented)

Documented from the 2026-08-29 TTFT-gap plan; **not shipped** in this recipe:

1. **Zero-copy PLE gather** — GPU reads host-resident PLE rows over NVLink-C2C
   ATS instead of per-row mmap faults on the 16 CPU `ggml_get_rows` path per
   token. Motivation: long prefill stays PLE-bound — 229,874-token depth
   **~5.3 ms/token** wall-clock (1,218.85 s TTFT); QSA kernels roughly **double**
   decode at that depth (5.60 → 11.55 tok/s) with **TTFT effectively flat**
   (1,218.85 → 1,198.88 s). Target: closer honest prefill without abandoning
   lazy CPU PLE.
2. **MTP draft-layer changes** — further draft-graph / indexer work beyond the
   QSA-wired `graph_mtp` verification in [`mtp-draft.md`](mtp-draft.md)
   (accept unchanged at 4k/64k). Parked after measured identical accept.

Closing the remaining 4k prefill gap (locked **318** vs case G **~381**) and
long-context prefill likely needs (1) or a separate serving lane (e.g.
vLLM-class prefill), not batch-size tuning alone.

## Invalid and rejected experiments

### Repeated-prompt `ngram-mod`

An early copy-heavy test reused an identical prompt. Its measured runs rose from 29.26 to 39.12 to 45.34 tok/s as the speculative cache learned the prior output. Those numbers are contaminated and excluded from release claims. A varied spec-on rerun was stopped after the host reboot described below; no spec-on release result exists yet.

### Qwen4Exp graph reuse patch

A current-head adaptation of the MIT patch from `0xBakeer/qwen38-flash-next-spark` reached `graphs reused = 127` during the warmup, then the server segfaulted on the second request. Kernel logs also recorded an NVIDIA `NV_ERR_NO_MEMORY` at the time of this experiment. The patch was removed and must not ship.

### Host reboot boundary

The Spark later rebooted without a clean shutdown. The previous boot contained multiple NVIDIA OOM events from several workloads, including one temporally associated with the graph-reuse experiment. Causality cannot be assigned solely to this recipe, but no additional model loads should be run until the host is treated as recovered and the unsafe patch remains excluded.


### Layer redundancy and cosine similarity probe — rejected (2026-08-29)

A structural probe measured inter-layer weight cosine similarities across all 48 layers of Qwen3.8-Flash-Next (including linear attention QKV, full attention Q/K, and shared MLP gates) to test whether middle blocks (layers 12–36) exhibit high redundancy ($\ge 0.90$) suitable for layer bypassing or pruning.

Findings:
- Linear attention QKV inter-layer similarity: Mean = **0.6455**, Max = **0.6514**, Min = **0.6424**
- Full attention QKV inter-layer similarity: Mean = **0.6559**, Max = **0.6572**, Min = **0.6541**
- Shared MLP gate/down inter-layer similarity: Mean = **0.6489 / 0.6556**, Max = **0.6549 / 0.6582**
- Peak observed inter-layer cosine similarity in middle blocks: **0.6582** (far below the $\ge 0.90$ redundancy threshold).

Unlike uniform dense transformers, Qwen3.8-Flash-Next's hybrid architecture (`3× linear + 1× full attention`) with hyper-connection mixing maintains highly orthogonal, specialized layer representations throughout the network. **Layer pruning / block bypassing is rejected.** Evidence: `raw/layer-similarity/`.
## Comparison-repository lessons

Research clone: `0xBakeer/qwen38-flash-next-spark` commit `4c6fc3af429bff5c472511cf965751eac6b7caf2`.

- Its credible free-form llama.cpp result is approximately 27.8 tok/s on a different `UD-Q4_K_XL` quant.
- Its 52–88 tok/s figures are copy-heavy `ngram-mod` workloads and are not general model speed.
- Its own later measurements contradict the earlier whole-PLE prewarming claim; this recipe does not prewarm the 26.8 GiB table.
- The old graph-reuse patch does not safely apply to the current QSA layout.
- Prompt cache, repeated prompts, speculative-cache learning, and short-prompt context allocation must be labeled separately.

The comparison repository is MIT licensed, Copyright (c) 2026 0xBakeer. Its methodology informed the varied task suite and rejected graph-reuse experiment; no third-party code is shipped here.

## Ten optimization tests (2026-08-30/31)

Campaign after the PLE-residency axis closed (`no-win`, commit `ee73c38`). Full
table and per-patch ANALYSIS under [`raw/ten-tests/`](raw/ten-tests/). Headline:

| Test | Verdict | Headline |
|---|---|---|
| T1 GET_ROWS `n_tasks = n_threads` | **win (prefill)** / decode −6.8% | 4k TTFT **6.806 s** (−37% vs 10.791), hash `99a15d5b` exact |
| T2 PLE `posix_madvise(WILLNEED)` | rejected | request-window reads unchanged (410 / 2018 MB) |
| T3 `-ub 1024` revival | **win** | 4k **9.199 s** (`06124a4b`); 64k 160.99 s (`a81283e2`) |
| T4 KV `q8_0` at 262k | **enabling win** | f16 262k guard-breach (35.77 GiB); q8_0 runs (37.97 GiB, 901.65 s) |
| T5 can-reuse × MTP | no-win | 26.54 tok/s vs 40.5 MTP-alone; main-tree draft load failed |
| T6 spec-on varied tasks | **release row** | +10–45% decode vs spec-off; accept 78.3% |
| T7 PLE row cache in `get_rows_q` | rejected | CPU `get_rows_q` never runs; PLE gather is CUDA |
| T8 mixed-lane `-np 2` | gate-fail | decode lane 12.87 tok/s < 15 (`--no-cache-prompt`) |
| T9 kmtp QSA×can-reuse | **integration proven** | 4k hash `c64973d8` byte-stable; 64k 20.44 tok/s; 230k 12.94 tok/s |
| T10 `PLE_MT_THREADS` at ub1024 | no-win | best warm-first 9.156 s (pool 6); cold 10.793 s |

T1 is the largest prefill move since PLE-MT. T7 showed it is **not** the PLE
IQ4_NL dequant loop (that is CUDA `getrows.cu`) — it is the remaining CPU
GET_ROWS sites. Short-hash `cb7904d8` held on T1 and T10.

## Raw evidence

- `raw/ctx1024-nocache-v1.jsonl`: honest short-prompt baseline
- `raw/ctx1024-opt*.jsonl`: runtime tuning sweep
- `raw/ctx*.jsonl`: context-allocation sweep
- `raw/depth-sweep-f16.jsonl`: actual prompt-depth sweep (`b512`/`ub128`, 2026-08-27)
- `raw/ttft-gap/`: full 2026-08-29 TTFT-gap sequence — decode `b512` vs `b2048` regression, unpatched + kmtp (QSA-wired `draft-mtp`) depth curves 4k–64k, `ngram` copy-heavy combo, and the 4k cold verification rerun (`rerun-unpatched-4k/`); runner: `tools/ttft_gap_benchmark.sh`, all 12 guard logs passing
- `raw/tasks-varied-spec-off.jsonl`: task-shape baseline with per-run prompt variation
- `raw/tasks-ngram-mod.jsonl`: contaminated repeated-prompt pilot; excluded

- `raw/concurrency-np2-ctx8192.jsonl`: three parallel-two request batches
- `raw/concurrency-np2-ctx8192-guard-summary.json`: memory, swap, stop, and kernel-fault evidence
- `raw/concurrency-np4/`: 4-way parallel-four continuous batching probe (`-np 4 -c 16384`) — 44.85 tok/s aggregate, 13.60 tok/s/req, 0 memory violations
- `raw/concurrency-np8/`: 8-way parallel-eight continuous batching probe (`-np 8 -c 32768`) — 66.67 tok/s aggregate, 10.39 tok/s/req, 0 memory violations
- `raw/ubatch-sweep/`: microbatch sweep (`-ub 256` vs `-ub 1024`) — `-ub 1024` achieved 481.3 tok/s prefill and 8.22 s 4k TTFT (35% TTFT reduction)
- `raw/deep-ub1024/`: deep-context `-ub 1024` sweep (64k and 128k cold depth benchmarks, all guard logs passing with min available >42 GiB)
- `raw/ple-baseline-profile.json`: unpatched RANDOM PLE fault/residency profile
- `raw/ple-mt/`: multithreaded PLE `set_input` index computation verification — short hash check (`cb7904d8`), cold 4k TTFT scaling (11.68 s at ub512, 10.78 s at ub1024), and cold 64k benchmark
- `patches/ple-multithreaded-set-input.patch`: patch parallelizing PLE n-gram index computation in `llm_graph_input_ple::set_input` across worker threads
- `raw/ple-advice-random.jsonl`: unpatched RANDOM cold+steady timings
- `raw/ple-residency/`: PLE residency Gate 0 + `-ub 1024` clean-tree revalidation + Step 3 reopen `--tensor-read-lazy off` A/B — true-cold request-window disk reads 4k ≈405–443 MB / 64k ≈1,085–2,152 MB with lazy on (Grok's disk-bound prefill reproduces on GB10); lazy-off reopen **no-win** (4k TTFT 22.674 s vs 10.791 s, request-window reads rose 25×/10× — unified-memory reclaim evicts the MAP_POPULATE'd pages; axis closed, keep `--tensor-read-lazy on`); Step 2 stopped at the pre-decided hash gate (`06124a4b` ≠ `99a15d5b` at `-ub 1024`), TTFT not compared, Step 2'/composition not run
- `raw/ple-pagesort/`: page-sorted PLE row gathering verification — short hash check (`cb7904d8`), cold 4k TTFT scaling (12.04 s), and cold 64k benchmark
- `raw/ple-advice-ab.json`: isolated mmap-advice A/B decision record
- `raw/ple-advice-prototype-random.jsonl`: patched RANDOM arm
- `raw/layer-similarity/`: inter-layer cosine similarity probe across all 48 layers of Qwen3.8-Flash-Next (peak similarity 0.6582 confirms absence of layer redundancy)
- `raw/ple-advice-prototype-normal.jsonl`: patched NORMAL arm
- `raw/ple-advice-prototype-sequential.jsonl`: patched SEQUENTIAL arm
- `patches/ple-lazy-advice.patch`: rejected env-selector prototype
- `raw/nsys-4k/`: nsys cold 4k prefill decision experiment — session kernel/memops reports, request benchmark, guard log, and window analysis (GPU 38.5% busy, ~6.6 s CPU-side gaps, graphs-reuse breakage identified); neither fused-gated-residual nor zero-copy-PLE indicated
- `raw/graph-reuse/`: QSA graph-reuse fix verification — can_reuse overrides, GDB crash backtrace of the dangling-mctx bug, final gate logs (short hash match, repeat-request survival, 127 reuses/128-tok decode, ~12% decode speedup), no-override 4k control hash
- `patches/qwen4exp-can-reuse.patch`: can_reuse overrides for llm_graph_input_qsa/llm_graph_input_ple with per-batch mctx refresh (also carries the restored PLE-MT content)
- `raw/rmsnorm-fusion/`: fused zero-centered RMSNorm investigation — axis closed as not indicated: (1+w) already converter-folded, GEMM alpha structurally scalar, existing {RMS_NORM,MUL} fuser covers legal sites, qwen4exp sites fail fusion gates structurally, prize ~0.5-1% decode
- `raw/deep-prefill-levers/`: deep-context prefill sizing + prefill graph-rebuild direct measurement — prefill-stable QSA graph CLOSED (build+alloc is 3.6-5.3 ms/ubatch, ~0.3-0.4% of prompt eval; nsys gaps are the host PLE gather path, not rebuild); decode graph-reuse win at 64k (71.88 vs 73.27 ms/tok, ~1.9%); ATS microbench named as deep-lane gate (failed, see ats-microbench)
- `raw/ats-microbench/`: zero-copy host PLE first gate — ATS sustained 8.11 GB/s (1 GiB) / 6.36 GB/s (8 GiB cache-defeated) at 90-byte random granularity, failing the ≥20 GB/s plan gate; reframe: PLE demand ~1.4 MB/s/ubatch is 4,500x under the measured floor, so deep prefill is not interconnect-bound (sources + analysis)
- `raw/ten-tests/`: 2026-08-30/31 ten-test campaign (T1–T10) — jsonl, server logs (`git add -f`), vmstat snapshots, per-patch ANALYSIS, campaign rollup
- `patches/tt10-t1.patch` / `tt10-t2.patch` / `tt10-t7.patch` / `tt10-t10.patch`: T1 GET_ROWS fan-out (measured win), T2 prefetch (rejected), T7 row cache (rejected, path absent), T10 PLE_MT env (no-win)
Raw JSONL in this directory is the unpatched recipe evidence. Kernel-track
timings are the sibling config
[`../../qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md`](../../qwen38-flash-next-ud-iq4-xs-qsa/results/qsa-kernels.md).
