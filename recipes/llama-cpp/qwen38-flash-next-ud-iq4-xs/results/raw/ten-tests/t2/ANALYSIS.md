# T2 — Server-side PLE row prefetch

Date: 2026-08-31. Stacked on the T1 tree. Helper `prefetch_ple_rows` in
`tools/server/server-context.cpp` replicates `llm_graph_input_ple::set_input`
n-gram hashing over the tokenized prompt and `posix_madvise(WILLNEED)` each
computed row (page-aligned). Patch: `patches/tt10-t2.patch`. No `--no-ple-prefetch`
flag (env A/B not needed — rejected on the MB_req gate).

T1+T2 vs T1-only via separate `libllama-server-impl.so` (T2 lives in the server
impl; T1 lives in `libggml-cpu`).

## Results (true-cold, ub512, lazy on; hashes exact)

| Arm | Hash | TTFT | prompt-eval | pgpgin_MB req | vs T1 |
|---|---|---:|---:|---:|---|
| `t2-4k` | `99a15d5b` | 6.560 s | 651.6 tok/s | **409.6** | T1 6.806 s / 437.8 MB |
| `t2-64k` | `b641e2eb` | 131.24 s | 515.3 tok/s | **2018.3** | T1 131.94 s / 2023.4 MB |

Success required MB_req drop ≥ 25% **and** TTFT win. MB_req is unchanged vs T1
and vs Gate 0 (405–443 / 1,085–2,152). Prefetch issues `WILLNEED` *inside* the
request window (after tokenize, before the task loop), so the faults still land
in the measured window — they just move earlier in it. 4k `pgmajfault` dropped
(59,830 T1 → 5,980 T2) which is consistent with madvise converting some major
faults, but pgpgin (KB actually read) did not drop.

TTFT deltas vs T1 are noise (+3.6% / +0.5%).

## Verdict

**Rejected (MB_req gate).** Mechanism does not move request-window NVMe reads
out of the request. Do not ship. Keep T1's GET_ROWS fan-out without prefetch.

Evidence: this directory.
