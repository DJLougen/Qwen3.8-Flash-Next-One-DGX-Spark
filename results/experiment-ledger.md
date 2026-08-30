# Qwen3.8-Flash-Next Exploration & Experiment Ledger

Authoritative reference of tested, rejected, parked, and untested exploration axes for **Qwen3.8-Flash-Next** on **NVIDIA DGX Spark** (1× GB10, 128 GB unified memory).

---

## 1. What Works & Why (Proven / Shipped / Measured Baseline)

| Axis / Configuration | Measured Result (1× GB10) | Why It Works | Production Decision / Status |
|---|---|---|---|
| **Batch default (`-b 2048 -ub 512`)** | **~29 tok/s** short decode (vs ~25 baseline); **0.15 s** warm TTFT | Larger microbatch amortizes CPU/GPU launch overhead on short requests without hurting single-stream decode. | **Shipped as recipe default** in `run.sh` (validated 2026-08-29; decode regression passed within 1.0%). |
| **Lazy PLE CPU MMAP (`--tensor-read-lazy on -lm mmap -ot per_layer_token_embd=CPU`)** | Enables 262k context allocation; keeps table resident footprint $<1\text{ GiB}$ | Leaves the 26.8 GiB `IQ4_NL` n-gram table on host DRAM; faults in only accessed rows rather than prefaulting 27 GB into HBM. | **Shipped as recipe default**; keeps memory under 36 GiB guard floor. |
| **`POSIX_MADV_RANDOM` on PLE table** | **1.911 s** cold TTFT on short prompt; 66 resident pages after load | Prevents Linux kernel readahead from pulling unneeded 4 KiB neighboring pages on scattered n-gram misses. | **Shipped as recipe default**; prevents page cache explosion. |
| **QSA CUDA kernels (`qsa-lightning-working.patch`)** | **18.73 tok/s** at 64k (vs 11.35 AR); **11.55 tok/s** at 229k (vs 5.60 AR, **2.06×**) | Fused mean+RMS weighting (`r=4`), `__ldg` half2/float4 loads on lightning WMMA, and compact FA gather accelerate per-token decode. | **Verified in sibling recipe** (`qsa-kernels.md`); unmerged patch track. |
| **`draft-mtp` ($N=3$) on short prompts** | **~40.5 tok/s** (75.6% accept) on memorized/short sequence vs ~25 AR (**1.6×**) | MTP draft head accurately predicts high-probability sequential continuations; verification overhead is lower than draft gain. | **Verified on isolated tree** (`llama.cpp-qwen4exp-mtp`); draft GGUF `q8_0` (3.9 GB). |
| **`draft-mtp` + `ngram-mod` on copy task** | **79.86 tok/s** decode (67–100% accept) on `reproduce-module.txt` | N-gram exact matching catches repeated tokens while MTP predicts transitions; highly synergistic for copy-heavy workflows. | **Verified; labeled separately** as copy-heavy benchmark only. |
| **Microbatch scaling (`-b 2048 -ub 1024`)** | **~28.74 tok/s** short decode; **8.22 s** 4k TTFT (**35% TTFT drop** vs 12.65 s at ub512) | Larger microbatch achieves 481 tok/s prefill (2.08 ms/token vs 3.16 ms at ub256) on GB10 without degrading decode. | **Proven accelerator for prefill**; keeps decode intact. |
| **Parallel-2 concurrency (`--parallel 2 -c 8192`)** | **20.68 tok/s/req** (**32.82 tok/s** aggregate), 0.853 s TTFT | Continuous batching fills GB10 SM capacity while staying within the 36 GiB memory guard (28.15 GiB headroom). | **Proven up to 2 concurrent streams**. |
| **Parallel-4 concurrency (`--parallel 4 -c 16384`)** | **13.60 tok/s/req** (**44.85 tok/s** aggregate), 1.945 s TTFT | 4-way continuous batching lifts aggregate throughput to ~45 tok/s with 47.5 GiB available memory and 0 memory violations. | **Proven safe up to 4 concurrent streams** (16k total context). |
| **Deep-context chunked prefill (`-b 2048 -ub 1024`)** | **163.53 s** 64k TTFT (19.96 tok/s); **386.77 s** 128k TTFT (**339.5 tok/s prefill**, **28.1% TTFT reduction** vs 538 s baseline) | Larger microbatch sustains high prefill throughput (339–481 tok/s) at extreme context without memory violations (min available >42 GiB). Flat at 64k where PLE IO dominates. | **Verified at 64k/128k**; documented in `summary.md` (`raw/deep-ub1024/`). |
| **Parallel-8 concurrency (`--parallel 8 -c 32768`)** | **10.39 tok/s/req** (**66.67 tok/s** aggregate), 3.034 s TTFT | 8-way continuous batching lifts aggregate throughput to ~66.7 tok/s with 45.39 GiB available memory and 0 memory violations. | **Proven safe up to 8 concurrent streams** (32k total context). |
| **Multithreaded PLE index hashing in `set_input` (`src/models/qwen4exp.cpp`)** | **11.68 s** cold 4k TTFT at `-ub 512` (**344.2 tok/s prefill**, **7.7% TTFT reduction** vs 12.65 s); **cb7904d8** output hash preserved | Parallelizes PLE n-gram hash mixing across 12 threads during prefill; reduces CPU index calculation latency without touching single-token decode. Does not beat the 8.22 s `-ub 1024` baseline (10.79 s due to thread scheduling overhead at larger ubatch). | **Verified and validated** (`raw/ple-mt/`). |
| **MTP Host Sampler for Stochastic Serving (`--no-spec-draft-backend-sampling` at $T=0.7$)** | **34.84 tok/s** decode at $T=0.7$ (**72.88% accept**, 29.10 ms/tok eval time) vs 19.88 tok/s (32.29% accept) with backend sampling (**+75% speedup**) | GPU backend draft sampler (`top-k=10`) diverges from stochastic target distributions; host `common_sampler` aligns draft proposals with target sampling to prevent rollback collapse and achieve peak throughput. | **Proven default for non-zero temperature serving** (`raw/mtp-sampler/`). |

---

## 2. What Doesn't Work & Why (Tested, Rejected, or Parked)

| Tested Axis / Experiment | Observed Result / Failure Mode | Why It Doesn't Work (Root Cause) | Action / Do Not Repeat |
|---|---|---|---|
| **0xBakeer CUDA graph-reuse patch** | Reached 127 reused graphs, then **segfaulted / caused `NV_ERR_NO_MEMORY`** | Does not handle QSA dynamic memory layout past warmup; corrupts execution graph on token boundary. | **Hard-rejected.** Removed from repo; never re-apply to Qwen4Exp. |
| **PLE Lazy advice selector (`NORMAL`/`SEQUENTIAL`)** | Cold TTFT regressed (**2.17 s / 2.30 s** vs 1.91 s); residency grew **14–61×** | Kernel readahead triggers page-cache thrashing for pseudo-random 3-token hash lookups across 26.8 GiB. | **Hard-rejected.** Keep `POSIX_MADV_RANDOM`; do not add env selector. |
| **Repeated-prompt `ngram-mod` as general speed** | Numbers climbed from 29.2 $\to$ 45.3 tok/s on identical prompt | Speculative cache memorizes exact prior output; masquerades as speed. | **Rejected as invalid measurement.** Must use varied `@` placeholders. |
| **MTP at long context ($N=3$ at 229k tokens)** | **10.2 tok/s** (43.2% accept) — **slower** than kernel AR (11.55 tok/s) | Draft KV cache overhead + full-context gather per draft step exceeds the time saved by low acceptance (~43%). | **Parked.** Do not use MTP for deep context ($>64\text{k}$) on single-GPU GB10. |
| **MTP draft KV cache at $f16$ on deep context** | Breaches 36 GiB guard floor during load at context $>229\text{k}$ | $f16$ KV for both trunk and draft requires $>16\text{ KB/token}$, exhausting GB10 unified memory. | **Rejected.** Must use `-ctkd q8_0 -ctvd q8_0` if testing deep context. |
| **`--spec-draft-n-max 8`** | Slower than autoregressive baseline | Rollback-slot synchronization cost and high draft rejection penalty exceed draft gains. | **Do not use.** Keep $N \le 3$. |
| **Wiring draft QSA indexer into `graph_mtp` to fix accept** | Acceptance stayed **identical** (54.9% at 4k, 33.3% at 64k) | MTP draft acceptance is content/entropy bound, not driven by sparse-vs-dense draft attention mismatch. | **Parked.** Do not chase further draft-graph indexer rewrites. |
| **kmtp (QSA-wired draft) at 64k with `-b 2048 -ub 512`** | **11 tokens, 3.7% accept, 8.97 tok/s** — target output diverged from the 64-token AR baseline | Same binary/draft/prompt at `-b 512 -ub 128` produced the correct 64 tokens at 33.3% accept. The larger ubatch corrupts target KV state in the QSA-wired draft's hybrid memory at 64k (`find_slot: non-consecutive token position` warnings, then early stop). Speculative decode must never change target output — this is a correctness bug, not a tuning result. | **Do not run kmtp at 64k with `-ub 512`.** Root-cause the hybrid-memory ubatch interaction before any MTP depth work. |
| **4-head lightning inner-loop / IQ4_XS 8-warp MMVQ** | No speedup or regressed 64k decode (17.13 vs 18.73 tok/s) | Divergent warp execution on non-power-of-two head counts; memory read amplification. | **Reverted in QSA patch.** |
| **Whole-table PLE prewarming in VRAM (~27 GiB)** | Exceeds 36 GiB available memory threshold | Deprives 262k KV cache of essential allocation headroom. | **Rejected for general serving.** |
| **Entropy-Gated Draft Length (`--spec-draft-p-min >= 0.60`)** | Halts drafting on arbitrary text ($p < 0.60$); emits only 3 tokens vs 64 at $p_{\min}=0.0$ (**28.38 tok/s**, 66.7% accept) | Natural text has low top-1 probability ($p < 0.60$), causing the gate to suppress drafting on nearly all steps. | **Keep `--spec-draft-p-min 0.0`** (disabled) for general serving. |
| **Page-Sorted Row Index Gathering (`ggml-cpu/ops.cpp`)** | Cold 4k TTFT regressed (**12.04 s vs 11.68 s** on PLE-MT tree); 64k regressed (**168.5 s vs 167.5 s**) | Per-thread `O(K \log K)` index sorting adds CPU overhead without overcoming Linux page-fault latency floors. | **Rejected and reverted.** Do not sort row indices in ggml-cpu. |
| **Layer Redundancy / Cosine Pruning (Layers 12–36)** | Peak inter-layer cosine similarity is only **0.6582** (mean **0.645–0.656** across linear/full attention & shared MLP); far below $\ge 0.90$ redundancy threshold | Hybrid architecture (3× linear + 1× full attention) with hyper-connections maintains distinct, specialized state updates across all 48 layers. | **Hard-rejected.** Do not prune or bypass middle transformer blocks. |

---

## 3. Untested Exploration Axes (Prioritized Backlog)

| Domain | Untested Axis / Experiment | Concrete Hypothesis / Mechanism to Test | Priority / Feasibility |
|---|---|---|---|
| **Layer Surgery** | **Fused Zero-Centered RMSNorm** | Fuse $(1 + w)$ weight offset directly into GEMM input scaling; eliminates 1 separate kernel launch per transformer block. | **High** (CUDA / GGML kernel edit) |
| **Layer Surgery** | **Fused Gated Residual + Projection** | Fuse $\text{output} = x + \text{gate} \odot \text{proj}(\text{act})$ into a single elementwise kernel pass. | **Medium** (Kernel fusion in `ggml-cuda`) |
| **MTP Knobs** | **Tree / Multi-Branch Speculation** | Evaluate a small 2-branch tree draft ($1 \to 2$) verified in a single masked attention pass vs linear chains. | **Medium** (Requires engine support in `llama-graph`) |
| **PLE & I/O** | **Zero-Copy Host PLE over NVLink-C2C** | Write a CUDA kernel that directly dereferences the host-mapped PLE table over GB10 ATS hardware coherency instead of CPU staging. | **High** (Major long-context prefill TTFT accelerator) |
