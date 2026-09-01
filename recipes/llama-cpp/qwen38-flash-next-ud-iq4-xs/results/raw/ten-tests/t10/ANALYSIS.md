# T10 — PLE-MT thread-pool sizing at `-ub 1024`

Date: 2026-08-31. Tree: `/home/djl/llama.cpp-tt10-t10` (clone of main dirty tree
at `250b61446`, no T1/T2/T7). Three-line change in `qwen4exp.cpp` `set_input`:
`n_threads = min(12, n_tokens)` → `min(PLE_MT_THREADS env, n_tokens)` default 12.
Patch: `patches/tt10-t10.patch`. `-t 12` launch flag fixed across arms.

## Warm 4k ub1024 sweep (warmup 1 + 3 measured; prompt-cache hits after warmup)

| PLE_MT_THREADS | warmup TTFT (first req after load) | measured median TTFT | decode median | hash |
|---:|---:|---:|---:|---|
| 12 | 10.832 s | 0.169 s | 26.61 tok/s | `06124a4b` exact |
| **6** | **9.156 s** | 0.186 s | 27.03 tok/s | `06124a4b` exact |
| 4 | 9.773 s | 0.178 s | 26.76 tok/s | `06124a4b` exact |

Measured-rep TTFTs (~0.17 s) are prompt-cached and not comparable to the 10.043 s
clean-tree ub1024 figure. The warmup column isolates dispatch overhead.

## Cold (evict, pool=6 = best warmup)

| Arm | Hash | TTFT | prompt-eval | vs refs |
|---|---|---:|---:|---|
| `t10-cold-4k-1024` | `06124a4b` exact | 10.793 s | 372.2 tok/s | Gate 0 ub512 10.791 s; T3 same-day ub1024 **9.199 s**; era 10.043 s |
| `t10-cold-4k-512` | `99a15d5b` exact | 11.038 s | 365.2 tok/s | Gate 0 10.791 s |
| `t10-short3` | `cb7904d8` exact | 0.140 s | — | decode 27.82 tok/s |

## Verdict

**No-win.** Pool=6 is the best first-request warmup (9.156 s) but cold ub1024 at
that pool is 10.793 s — does not beat 10.043 s or T3's 9.199 s on the reference
binary. Hashes exact (pool size does not change outputs). Keep the hardcoded 12;
do not ship an env knob. The T10 clone's cmake was a fresh Release+CUDA configure
(not a copy of the reference `CMakeCache`), so absolute TTFT is not a same-binary
A/B vs T3 — the within-T10 pool curve is the valid comparison, and it is flat
within ~1.7 s warmup / ~0.3 s cold.

Evidence: this directory.
