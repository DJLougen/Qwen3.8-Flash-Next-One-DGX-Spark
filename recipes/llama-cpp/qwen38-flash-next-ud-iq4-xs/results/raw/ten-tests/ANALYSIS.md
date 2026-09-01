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

## By context length

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
