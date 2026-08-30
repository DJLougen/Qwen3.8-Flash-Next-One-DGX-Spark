# Experiment A — nsys profile of cold 4k prefill (2026-08-30)

Decision experiment: attribute the ~11.68 s cold 4k TTFT (PLE-MT build) across GPU
kernels, CPU PLE gather, and sync gaps; decides Experiment B vs C.

## Setup

- Binary: `build/bin/llama-server.ple-mt` (Spark tree `250b61446` + QSA patch + PLE-MT).
- Flags: `-c 4096 -np 1 -b 2048 -ub 512 -t 12 -fa on -lm mmap --tensor-read-lazy on
  -ot per_layer_token_embd=CPU -ngl all -fit off --host 127.0.0.1 --port 28900`.
- Profiler: `nsys profile --trace=cuda,nvtx --output=/tmp/nsys-4k` under
  `/tmp/spark_guard.py` (80/36/28 GiB floors, 1 GiB swap; guard log `guard.jsonl`).
- One cold request: `ctx4096.txt`, 64 max tokens, 0 warmups, 1 repetition
  (`benchmark.jsonl`): TTFT **12.015 s** (nsys overhead vs 11.68 s baseline ≈ +2.9%),
  prompt eval 11.785 s, decode 19.8 tok/s. Guard floors held (min available ≈ 52 GiB).
- Finalization quirk: SIGINT must go to the nsys CLI PID only, not the guard child
  group — the guard's group SIGKILL after grace truncates the qdstrm (first attempt
  produced an incomplete `/tmp/nsys-report-a9c0.qdstrm`). Signaling only nsys lets the
  report finalize while the server is later killed separately (server teardown hangs in
  `futex_do_wait`; SIGKILL is the established teardown for all runs on this tree).
- Session-wide reports: `nsys-4k-stats.txt` (cuda_gpu_kern_sum, cuda_gpu_mem_time_sum).
  Window analysis below was extracted from `/tmp/nsys-4k.sqlite` over the exact
  prompt-eval window 121.8–133.6 s (server-log `print_timing` correlation).

## Attribution of the 11.785 s prompt eval

| Component | Time | Share | Evidence |
|---|---|---|---|
| GPU kernel busy (union of kernel intervals) | **4.545 s** | **38.5%** | 44,442 kernels in window; interval union |
| CPU-side gaps with zero CUDA activity | **6.62 s** (13 gaps > 50 ms; 7 gaps of 350–1200 ms between ubatch bursts) | **56.1%** | gap interval analysis |
| — of which: one-time CUDA JIT (`cuLibraryLoadData`) | 0.571 s | 4.8% | 4 calls inside gaps |
| — of which: `cudaStreamSynchronize` inside gaps | 0.060 s | 0.5% | 411 calls |
| — remainder: pure non-CUDA CPU per-ubatch work | **~5.99 s** | **50.8%** | no runtime/memcpy/osrt rows in gaps |
| H2D of gathered PLE activations | **0.007 s** | 0.06% | 4,443 H2D memcpys total in window |

## Top-10 GPU kernels in the prompt-eval window

| Total (ms) | Instances | Kernel |
|---|---|---|
| 2159.3 | 5531 | mul_mat_q |
| 766.8 | 9337 | k_bin_bcast (elementwise mul/add/repeat chains) |
| 346.3 | 310 | gated_delta_net_cuda |
| 277.1 | 1242 | mm_ids_helper |
| 230.7 | 2082 | unary_gated_op_kernel |
| 158.5 | 1589 | rms_norm_f32 |
| 143.1 | 5532 | quantize_mmq_q8_1 |
| 142.0 | 1970 | cutlass::Kernel2 |
| 73.8 | 319 | concat_non_cont |
| 49.6 | 104 | flash_attn_ext_f16 |

## Classification per plan

- **Experiment C gate (>15% CPU PLE gather + H2D): NOT MET.** H2D of gathered
  activations is 7 ms (0.06%). The earlier PLE row-access profile already bounded
  unique-row page-fault cost well under 15%; the gaps are not PLE-staging driven.
  Supporting measurements: second identical request TTFT 0.244 s (KV-cache hit); a
  shuffled same-length prompt on the same server TTFT 8.636 s vs 9.969 s cold —
  page-fault warmth accounts for only ~1.3 s of the gap total.
- **Experiment B gate (>70% GPU kernel time with many small elementwise kernels):
  NOT MET.** GPU busy is 38.5%. Elementwise (`k_bin_bcast` + `unary_gated_op_kernel` +
  `concat_non_cont`) is ~1.07 s of 4.55 s kernel time (~9% of TTFT) — a full fusion
  could recover at most ~3–4% TTFT, below the ≥3% gate once realized.
- **Neither branch fires. Both B and C are recorded as not-indicated by profile.**

## Actual dominant cost: per-ubatch CPU serialization from broken CUDA-graph reuse

- QSA build (this binary): `graphs reused = 0` — every ubatch rebuilds the compute
  graph on CPU. Unpatched `250b61446` at identical flags/protocol:
  `graphs reused = 7`, prompt eval **10.381 s** (380.99 tok/s) —
  ~1.4 s faster with 0.6 s more GPU-visible work efficiency from not rebuilding.
- Gap structure: ~500 ms of pure CPU work (no CUDA API, no osrt syscalls) between each
  512-token ubatch GPU burst (~7 bursts + initial graph build). Pattern: graph burst →
  `cudaGraphLaunch`/`cudaStreamSynchronize` → 300–1200 ms silence → ~30 tiny
  PLE-row H2D memcpys (≤0.3 ms) → next burst.
- The QSA patch's dynamic per-ubatch graph shape (hybrid-index tensors, `get_v_ntrans`
  view, `mm_ids` mapping) defeats the graph cache, forcing per-ubatch rebuild +
  re-instantiation on the CPU critical path.
- Cross-check: ledger row "0xBakeer CUDA graph-reuse patch" already hard-rejected
  (segfault) — the generic graph-reuse approach is a dead end on this tree; a QSA-native
  fix (ubatch-stable graph shape or per-ubatch graph cache keyed on QSA block count)
  is the untested lever this profile points to.

## Follow-up recorded in ledger

"Fused Gated Residual + Projection" (B) and "Zero-Copy Host PLE" (C) → both updated to
**not indicated** by this profile; new prioritized axis: **QSA CUDA-graph-reuse fix**
(High).
