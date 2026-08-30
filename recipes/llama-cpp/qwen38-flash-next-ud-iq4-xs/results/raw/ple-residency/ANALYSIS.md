# PLE residency on GB10 — Gate 0 disk-read measurement, Step 2 hash-gate stop

Date: 2026-08-30. Binary: `/home/djl/llama.cpp-qwen4exp/build/bin/llama-server`
(14:47 graph-reuse rebuild, tree `250b61446` + 7-file QSA patch — `git rev-parse` confirmed
`250b61446` before launch). No llama.cpp source edits in this experiment.

Bench: `/tmp/stream_benchmark.py`, temperature 0, `--reasoning off` server-side, one cold
request per arm, `--warmup-count 0 --repetitions 1`. Every load under `spark_guard.py`
(80 start / 36 soft / 28 hard GiB, swap growth ≤ 1 GiB). Every cold arm preceded by the
120 GiB DeepSeek-V4 page-cache eviction; true-cold verified by load-window `pgpgin`
(51–53 GiB read per load, ≥ the 20 GiB check). `pgpgin` is in **KB** on this kernel
(6.17.0-1026-nvidia): 53.7 M units ≈ 51.2 GiB at load — so pgpgin→MB uses no ×4.

Arms executed (all flags per plan: `-b 2048 -t 12 -fa on -lm mmap
-ot per_layer_token_embd=CPU -ngl all -fit off`, port 28900, `-c/-ub/--tensor-read-lazy`
varied):

| Arm | CTX | UB | Lazy | Evict |
|---|---:|---:|---|---|
| `gate0-4k` | 4096 | 512 | on | yes |
| `gate0-64k` | 65536 | 512 | on | yes |
| `ub1024-4k` | 4096 | 1024 | on | yes |

## Gate 0 — disk-read during cold prefill (lazy on)

Request window = after-`/health` snapshot → after-request snapshot. Three estimators:

| Arm | MB_req vmstat `bi` | MB_req iostat `rkB/s` | MB_req pgpgin (KB) | pgmajfault | Threshold | Classification |
|---|---:|---:|---:|---:|---:|---|
| `gate0-4k` | 443.4 | 443.1 | 405.1 | 55,669 | 100 MB | **material** |
| `gate0-64k` | 1,098.2 | 1,085.1 | 2,152.1 | 458,895 | 500 MB | **material** |

vmstat `bi` and iostat `rkB/s` agree to <2% (device-level reads). `pgpgin` runs higher at
64k (2.15 GB vs 1.1 GB) — it counts all pages read including readahead; the classification
is unambiguous under any of the three estimators at both depths. Grok's disk-bound cold
prefill **reproduces on GB10**: 4k reads ~0.4 GB (4× the 100 MB gate, ~1/16 of Grok's
647 MB at 8k given the smaller unique-row set), 64k reads ~1.1–2.2 GB — a material
fraction of the ~1.8 GB amplified unique-row set.

Per-arm detail (all hashes vs the clean-tree 2026-08-30 references):

| Arm | load→health | TTFT | prompt eval | decode | hash `[:8]` | guard min MemAvail | swap growth |
|---|---:|---:|---|---:|---|---:|---:|
| `gate0-4k` | 167 s | 10.791 s | 10,616.43 ms / 3,955 tok = **372.54 tok/s** | 24.60 tok/s | `99a15d5b` ✓ | 47.81 GiB | 0.222 GiB |
| `gate0-64k` | 140 s | 170.663 s | 169,972.84 ms / 65,395 tok = **384.74 tok/s** | 14.53 tok/s | `b641e2eb` ✓ | 43.27 GiB | 0.218 GiB |
| `ub1024-4k` | 122 s | 10.043 s (not compared — see below) | 9,829.38 ms / 3,955 tok = 402.37 tok/s | 25.09 tok/s | `06124a4b` ✗ | 47.68 GiB | 0.212 GiB |

Guard floors held in every arm (min MemAvailable ≥ 43 GiB vs 36 GiB soft floor; hard kill
never fired; swap growth ≤ 0.23 GiB vs 1 GiB cap).

## Step 2 — `-ub 1024` revalidation on the clean tree

`ub1024-4k` ran true-cold after eviction (load pgpgin ≈ 51 GiB) with flags identical to
`gate0-4k` except `-ub 1024`. Result: **hash `06124a4b` ≠ reference `99a15d5b`**.

Per the plan's pre-decided stop rule ("mismatch on this control-like run → binary
drifted; stop the whole experiment"), TTFT was **not** compared and **Step 3 was not
run**. The 10.043 s TTFT / 402.37 tok/s figures are recorded for completeness only.

Context recorded alongside the stop (does not override it): the same binary produced
hash-exact outputs minutes earlier in the Gate 0 arms at both depths (`99a15d5b`,
`b641e2eb`), so the mismatch is attributable to `-ub 1024` itself altering ubatch-split
numerics relative to the `-ub 512`-era reference digest, not to binary drift between
sessions. The reference `99a15d5b` was produced at `-ub 512`; the 2026-08-29 ledger also
records `ple-mt` at `-ub 1024` with a different 4k hash (`10.79 s` row) — consistent with
per-ubatch output divergence. Consequence for any reopen: establish a per-ubatch hash
baseline before using the hash as a control gate.

## Step 3 — not run

Gate 0 classified both depths material (4k ≥ 100 MB, 64k ≥ 500 MB), which would have
triggered the `--tensor-read-lazy off` A/B at 4k and 64k. The Step 2 hash-gate stop
halted the experiment first, per the plan's stop rule. No lazy-off arm, no surgical
PLE populate arm (`dd` range documented in the plan), no composition run.

## Verdict

**`rejected`** — Step 2's pre-decided hash gate fired (`06124a4b` ≠ `99a15d5b`); the
experiment stopped before Step 3. Gate 0 stands as measured: cold-prefill disk reads are
material on GB10 at 4k (~0.4 GB) and 64k (~1.1–2.2 GB) with lazy on, so the residency
axis is **open**, not closed — but untested (`--tensor-read-lazy off` and surgical PLE
populate remain unmeasured here). The 2026-08-29 `8.22 s / 481 tok/s` `-ub 1024` figures
remain era-2026-08-29 and are not clean-tree validated.

## Files

- `ple-res-<arm>.jsonl` — benchmark rows (run + aggregate)
- `ple-res-<arm>-guard.jsonl` — guard lifecycle/samples
- `ple-res-<arm>-vmstat.log`, `-iostat.log` — 1 s samplers
- `ple-res-<arm>-vmstat-{before,after-load,after-req}.txt` — `/proc/vmstat` snapshots
- `ple-res-<arm>-server.log` — server logs (`slot print_timing` lines)
- `ple-res-<arm>-run.log`, `-runner.out` — arm runner logs
- `ple-res-runner.sh` — arm runner (procedure as executed)

Sampler window indexing: vmstat/iostat samplers started at load_t0; health at +167 s
(4k), +140 s (64k), +122 s (ub1024); request windows end at the after-req snapshots
(+195 s at 64k, +181 s at 4k).
