# PLE residency on GB10 — Gate 0, Step 2 hash-gate stop, Step 3 reopen (lazy-off A/B, no-win)

Date: 2026-08-30 (reopen arms 20:21–20:36 UTC). Binary:
`/home/djl/llama.cpp-qwen4exp/build/bin/llama-server` (14:47 graph-reuse rebuild, tree
`250b61446` + 7-file QSA patch, sha256 prefix `c6a3ace0` — verified unchanged at reopen
time, so all hash references carry forward). No llama.cpp source edits in this
experiment.

Bench: `/tmp/stream_benchmark.py`, temperature 0, `--reasoning off` server-side, one cold
request per arm, `--warmup-count 0 --repetitions 1`. Every load under `spark_guard.py`
(80 start / 36 soft / 28 hard GiB, swap growth ≤ 1 GiB). Every cold arm preceded by the
120 GiB DeepSeek-V4 page-cache eviction; true-cold verified by load-window reads (see
below). `pgpgin` is in **KB** on this kernel (6.17.0-1026-nvidia) — pgpgin→MB uses no ×4.
Request window = after-`/health` snapshot → after-request snapshot; sampler row indexing
per-arm from the run-log timestamps (samplers start at load_t0).

Arms executed (all flags per plan: `-b 2048 -t 12 -fa on -lm mmap
-ot per_layer_token_embd=CPU -ngl all -fit off`, port 28900, `-c/-ub/--tensor-read-lazy`
varied):

| Arm | CTX | UB | Lazy | Evict |
|---|---:|---:|---|---|
| `gate0-4k` | 4096 | 512 | on | yes |
| `gate0-64k` | 65536 | 512 | on | yes |
| `ub1024-4k` | 4096 | 1024 | on | yes |
| `lazyoff-4k` | 4096 | 512 | **off** | yes |
| `lazyoff-64k` | 65536 | 512 | **off** | yes |

## Prior results carried forward (Gate 0 + Step 2 stop, same binary)

Gate 0 (lazy on) — disk-read during cold prefill, three estimators:

| Arm | MB_req vmstat `bi` | MB_req iostat `rkB/s` | MB_req pgpgin (KB) | pgmajfault | Threshold | Classification |
|---|---:|---:|---:|---:|---:|---|
| `gate0-4k` | 443.4 | 443.1 | 405.1 | 55,669 | 100 MB | **material** |
| `gate0-64k` | 1,098.2 | 1,085.1 | 2,152.1 | 458,895 | 500 MB | **material** |

vmstat `bi` and iostat `rkB/s` agree to <2%; `pgpgin` runs higher at 64k (counts
readahead). Classification unambiguous under all three estimators at both depths: Grok's
disk-bound cold prefill **reproduces on GB10** with lazy on.

Gate 0 per-arm detail (hashes vs the clean-tree 2026-08-30 `-ub 512` references):

| Arm | load→health | TTFT | prompt eval | decode | hash `[:8]` | guard min MemAvail | swap growth |
|---|---:|---:|---|---:|---|---:|---:|
| `gate0-4k` | 167 s | 10.791 s | 10,616.43 ms / 3,955 tok = **372.54 tok/s** | 24.60 tok/s | `99a15d5b` ✓ | 47.81 GiB | 0.222 GiB |
| `gate0-64k` | 140 s | 170.663 s | 169,972.84 ms / 65,395 tok = **384.74 tok/s** | 14.53 tok/s | `b641e2eb` ✓ | 43.27 GiB | 0.218 GiB |
| `ub1024-4k` | 122 s | 10.043 s (not compared) | 9,829.38 ms / 3,955 tok = 402.37 tok/s | 25.09 tok/s | `06124a4b` | 47.68 GiB | 0.212 GiB |

Step 2 stop: `ub1024-4k` tripped the pre-decided hash gate (`06124a4b` ≠ `99a15d5b`);
TTFT not compared, Step 3 not run that session. The mismatch is attributable to
`-ub 1024` ubatch-split numerics diverging from the `-ub 512`-era reference digest (the
same binary matched both Gate 0 hashes minutes earlier), not binary drift. The reopen
plan therefore gated the `-ub 512` arms on `99a15d5b`/`b641e2eb` and would have gated any
`-ub 1024` arm on `06124a4b` (per-ubatch baseline).

## Step 3 reopen — `--tensor-read-lazy off` A/B (2026-08-30, 20:21–20:36 UTC)

Mechanism under test: `llama-mmap.cpp:479` sets `MAP_POPULATE` on the whole GGUF mapping
only when `prefetch && lazy_ranges.empty()` — i.e. exactly when `--tensor-read-lazy off`
(`llama-model-loader.cpp:1290-1299` adds the `TENSOR_READ_LAZY` tensor to
`lazy_tensor_ranges` only when mode ≠ OFF). Both arms ran true-cold with flags identical
to the same-depth Gate 0 arm except `--tensor-read-lazy off`. The load-watch gate never
fired (no guard `hard_kill`; server survived MAP_POPULATE of the 88 GiB GGUF), so the
surgical-populate Arm 3b was not needed.

Measured arms:

| Arm | load→health | TTFT | prompt eval (slot) | decode | hash `[:8]` | guard min MemAvail | swap growth | MB_req vmstat `bi` | MB_req iostat `rkB/s` | MB_req pgpgin |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| `lazyoff-4k` | 198 s | **22.674 s** | 22,472.91 ms / 3,955 tok = **175.99 tok/s** | 21.39 tok/s | `99a15d5b` ✓ | 47.26 GiB | 0.226 GiB | 10,217 MB | 10,204 MB | 10,454 MB |
| `lazyoff-64k` | 210 s | **168.323 s** | 167,582.37 ms / 65,395 tok = **390.23 tok/s** | 14.36 tok/s | `b641e2eb` ✓ | 46.05 GiB | 0.220 GiB | 22,376 MB | 22,364 MB | 21,904 MB |

Load-window reads confirm the flag did what it says on the load path: `lazyoff-4k`
load-window `bi` ≈ **87.5 GB** and `lazyoff-64k` ≈ **117.8 GB** (whole-GGUF populate plus
context), vs ~51 GiB pgpgin per lazy-on load in Gate 0. Guard floors held in every arm
(min MemAvailable 46.05–47.26 GiB vs 36 GiB soft floor; hard kill never fired; swap
growth ≤ 0.23 GiB vs 1 GiB cap). Both hashes match the `-ub 512` references exactly —
outputs are byte-identical to Gate 0, so TTFT comparison is valid.

### Gates

- **Hash gate:** PASS both arms (`99a15d5b` 4k, `b641e2eb` 64k).
- **Win gate (TTFT ≤ 0.95× Gate 0):** 4k gate 10.25 s — measured 22.674 s, **2.10× the
  Gate 0 TTFT (+110%)**. FAIL. 64k gate 162.13 s — measured 168.323 s (−1.37% vs
  170.663 s). FAIL (sub-gate).
- **Mechanism gate (MB_req < ~50 MB at 4k / ~250 MB at 64k):** FAIL, and inverted —
  request-window reads **rose 25×** at 4k (405–443 MB → ~10.2 GB) and **10×** at 64k
  (1,085–2,152 MB → ~21.9 GB). All three estimators agree at both depths.
- **Guard floors:** PASS (no hard kill; min ≥ 46 GiB; swap ≤ 0.23 GiB).

### Why MAP_POPULATE made residency worse, not better

`--tensor-read-lazy off` did populate the whole GGUF into the page cache at load
(load-window reads 87.5–117.8 GB, +31–70 s load→health). But on GB10's 128 GiB unified
memory the request itself allocates the KV cache and activation working set on top of the
model, and kernel reclaim evicts the clean, mapped file pages first. By request time the
populated weights were gone from RAM: prefill re-faulted them from NVMe at full force —
~10.2 GB read during the 4k request (vs 0.4 GB lazy-on, where only the PLE
demand-fault rows were read) and ~21.9 GB at 64k. The 4k prompt-eval rate halved
(372.5 → 176.0 tok/s) and TTFT doubled; at 64k prefill compute is large enough that the
extra paging mostly overlapped, leaving TTFT essentially flat (−1.4%, sub-gate).
Notably the request-window `pgmajfault` count did **not** scale with the bytes (4k:
45,699 vs Gate 0's 55,669; 64k: 119,340 vs 458,895) — the re-fault of evicted
once-populated mapping brings readaround/readahead amplification with it, which is why
`bi`/`pgpgin` bytes explode while fault counts stay comparable. Total disk traffic per
cold request went **up**, not down (4k: ~51.4 → ~97.7 GB; 64k: ~53 → ~140 GB).

The lazy-on design is the correct one for this workload: `--tensor-read-lazy on` +
`POSIX_MADV_RANDOM` faults in only the accessed PLE rows (~0.4 GB at 4k) and leaves the
rest of the 26.8 GiB table on disk, exactly as the shipped recipe default intends. There
is no residency win available from whole-file populate under unified-memory reclaim
pressure — the axis is closed without `mlock` (per plan, mlock is out of scope and not
added).

### Skipped arms (pre-decided by the plan)

- **Arm 3b (surgical PLE populate):** not needed — the load-watch gate never fired; Arm A
  completed both depths healthy.
- **Step 2' (`-ub 1024` revival) and composition:** not run — the plan gates them on a
  Step 3 win at 4k, which did not occur. The 2026-08-29 `8.22 s / 481 tok/s` `-ub 1024`
  figures remain era-2026-08-29, not clean-tree validated.

## Verdict

**`no-win`** — `--tensor-read-lazy off` loses at 4k (TTFT 22.674 s vs 10.791 s, +110%)
and does not clear the 5% win gate at 64k (168.323 s vs 170.663 s, −1.4%), with
byte-identical outputs and held guard floors. The mechanism check is decisive: residency
did not improve — request-window disk reads rose 25×/10× (all three estimators agree), so
whole-file MAP_POPULATE on GB10 unified memory **worsens** cold-prefill paging by loading
pages the kernel then reclaims before the request. The PLE residency axis is **closed**:
keep `--tensor-read-lazy on` (recipe default); do not pursue whole-file populate. mlock
remains out of scope per plan. Serving defaults unchanged (`-ub 512`, lazy `on`,
`-ot per_layer_token_embd=CPU`).

## Files

- `ple-res-<arm>.jsonl` — benchmark rows (run + aggregate)
- `ple-res-<arm>-guard.jsonl` — guard lifecycle/samples
- `ple-res-<arm>-vmstat.log`, `-iostat.log` — 1 s samplers
- `ple-res-<arm>-vmstat-{before,after-load,after-req}.txt` — `/proc/vmstat` snapshots
- `ple-res-<arm>-server.log` — server logs (`slot print_timing` lines)
- `ple-res-<arm>-run.log`, `-runner.out` — arm runner logs
- `ple-res-runner.sh` — arm runner (procedure as executed)

Sampler window indexing (health→after-req): 4k +198→+224 s (26 rows), 64k +210→+383 s
(173 rows). Prior-session windows for reference: Gate 0 health at +167/+140/+122 s,
after-req at +195/+181 s; 64k request window +140→+195 s.
