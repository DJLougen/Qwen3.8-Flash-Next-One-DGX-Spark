# T1 — GET_ROWS CPU thread fan-out

Date: 2026-08-30/31. Tree: `/home/djl/llama.cpp-tt10-t1` (clone of `llama.cpp-qwen4exp` at
`250b61446` + dirty QSA/can-reuse/PLE-MT diff, then one-line patch). Binary:
`llama.cpp-tt10-t1/build/bin/llama-server` with `libggml-cpu` from that build.

Patch (`patches/tt10-t1.patch`): in `ggml/src/ggml-cpu/ggml-cpu.c`,
`case GGML_OP_GET_ROWS:` `n_tasks = 1;` → `n_tasks = n_threads;` (`SET_ROWS`
stays 1). The FIXME comment said get_rows *can* use more threads.

## Results vs Gate 0 (same protocol, true-cold, ub512, lazy on)

| Arm | Hash | TTFT | prompt-eval | decode | vs Gate 0 TTFT |
|---|---|---:|---:|---:|---|
| `t1-4k` | `99a15d5b` **exact** | **6.806 s** | 599.9 tok/s | 23.42 tok/s | 10.791 s → **−37.0%** (win ≤ 10.25 s) |
| `t1-64k` | `b641e2eb` **exact** | **131.94 s** | 498.1 tok/s | 13.96 tok/s | 170.66 s → **−22.7%** (win ≤ 162.13 s) |
| `t1-short3` (warm, 3-rep) | `cb7904d8` **exact** | 0.144 s median | — | **26.66 tok/s** | 28.6 tok/s → **−6.8%** (decode-regression gate ≤ 1% **fails**) |

Request-window pgpgin: 4k 437.8 MB / 64k 2023.4 MB — same band as Gate 0
(405–443 / 1,085–2,152). Guard floors held (4k min 50.86 GiB). `graphs reused = 63`
(prefill rebuilds, unchanged).

## Verdict

**Win (prefill) / decode-regression flagged.** Hashes prove the thread fan-out does
not change outputs. Prefill TTFT clears both depth gates by a wide margin. Short
warm decode −6.8% exceeds the 1% gate — CPU-thread launch cost leaking into
single-token decode, consistent with the original FIXME.

T7 later showed `ggml_compute_forward_get_rows_q` never runs for the PLE table on
this binary (CUDA `getrows.cu` handles `GGML_OP_GET_ROWS` for the IQ4_NL table).
T1's measured win is therefore the remaining **CPU** GET_ROWS sites (QSA
member/top_k gathers and any others pinned to CPU), not the PLE IQ4_NL dequant
loop itself. The win is still real and hash-exact.

Evidence: this directory (`ple-res-t1-*.jsonl`, server logs, vmstat snapshots,
short-prompt jsonl).
