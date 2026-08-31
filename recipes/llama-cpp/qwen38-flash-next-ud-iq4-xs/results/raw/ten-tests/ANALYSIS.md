# Ten optimization tests — campaign rollup (2026-08-30/31)

True-cold arms via `/tmp/tt10-runner.sh` (env-overridable `SERVER_BIN`/`LIB_DIR`/`KVQ`).
Hashes: `99a15d5b` (4k ub512), `b641e2eb` (64k ub512), `06124a4b` (4k ub1024),
`cb7904d8` (short). Guard 80/36/28 unless noted.

| Test | Verdict | Headline number |
|---|---|---|
| T1 GET_ROWS fan-out | **win (prefill) / decode-regression flagged** | 4k TTFT **6.806 s** (−37%), hash exact; short decode −6.8% |
| T2 PLE prefetch | rejected (MB_req) | pgpgin 410 / 2018 MB, no ≥25% drop |
| T3 ub1024 revival | **win** | 4k **9.199 s** (`06124a4b`); 64k 160.99 s (`a81283e2`) |
| T4 KV q8_0 | **enabling win at 262k** | f16 262k guard-breach (35.77 GiB); q8_0 runs (37.97 GiB, 901.65 s) |
| T5 MTP composition | no-win | 26.54 tok/s vs 40.5 gate; main-tree draft load failed |
| T6 spec-on tasks | **release row** | +10–45% decode vs spec-off; accept 78.3% |
| T7 PLE row cache | rejected (path absent) | `get_rows_q` never runs; PLE gather is CUDA |
| T8 mixed-lane np2 | gate-fail | decode lane 12.87 tok/s < 15; `--no-cache-prompt` required |
| T9 kmtp integration | **proven (hash-drift)** | 4k `c64973d8` byte-stable; 64k 20.44 tok/s; 230k 12.94 tok/s |
| T10 PLE_MT pool | no-win | best warm-first 9.156 s (pool 6); cold 10.793 s, no beat of 10.043 |

Patches: `patches/tt10-t{1,2,7,10}.patch`. Clones left on Spark:
`/home/djl/llama.cpp-tt10-t1`, `/home/djl/llama.cpp-tt10-t10`.
