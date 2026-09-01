# Deep-context prefill levers — sizing and disposition (2026-08-30)

## Question

After the decode graph-reuse fix shipped (see `../graph-reuse/`), what remains for
prefill, especially at deep context?

## Evidence assembled

1. **4k prefill anatomy** (`../nsys-4k/`): GPU busy 38.5% (4.545 s of 11.785 s);
   ~6.6 s of CPU-side gaps between ubatch bursts; PLE H2D 7 ms. Gap cause at 4k is
   dominated by per-ubatch compute-graph rebuild + scheduling (padded n_kv grows
   512/ubatch → `can_reuse` correctly false → full rebuild), plus one-time CUDA JIT
   (~0.57 s) and sync overhead.
2. **Prefill rebuilds are architecturally forced** for the current graph shapes: the
   QSA tensors (`cell_blk`, `blk_cells`, `blk_pos`, `bias`) and the kq_mask carry
   n_kv/n_blocks dims. The open variant — fixed max-shape tensors + `-inf` bias
   masking (`n_blocks_max` machinery already exists in `llama-memory-hybrid-idx.cpp`)
   — would make prefill graphs reusable too.
3. **Empirical upper bound of that variant**: the vanished-library era (Aug 29,
   graph-reuse code active in libs) measured 4k prompt eval **10.38 s** vs today's
   clean-tree 11.17-13.18 s (page-cache dependent). That bounds the prefill-stable
   variant at **~1.4-2.4 s (~12-20%) at 4k**, and likely less, since that era's
   libs also differed in other ways.
4. **Depth scaling kills it at 64k+**: prior deep-context sweep showed 64k TTFT
   flat vs `-ub 1024` (+0.7%) because "PLE memory/IO bottlenecks dominate over
   chunk launch batching" — at depth, prefill is page-fault/IO-bound across the
   26.8 GiB PLE table, not orchestration-bound. The decode fix showed the same
   physics: its win shrank from ~12% (4k) to ~1.9% (64k) as per-token GPU cost
   grew. A prefill-stable graph would shrink the same way: at 64k the ~500
   ms/ubatch rebuild is amortized over far more work per ubatch and the PLE fault
   floor dominates.

## Disposition

- The **prefill-stable graph variant** (fixed max-shape QSA tensors + bias masking)
  is a **medium-effort, depth-decaying** lever: ~12-20% cap at 4k, shrinking toward
  ~0-2% at 64k+ where PLE IO dominates. Recorded in the ledger as open (Medium)
  for shallow/mid-context serving lanes only; not pursued further this session —
  at the depths this deployment actually serves (4k default, 262k max), the 4k
  lane's best remaining prefill lever is already captured by `-ub 1024`
  (481 tok/s, 35% TTFT cut, shipped), and the deep lane is IO-bound.
- **Deep-context prefill itself is PLE-IO-bound** (prior session's flat 64k result
  + this session's 2.72 ms/tok at 64k): the only lever that moves it is the
  zero-copy host PLE over NVLink-C2C ATS axis — parked in the ledger for 4k, but
  it remains *the* deep-context prefill lever. Its precondition microbenchmark
  (ATS bandwidth at 90-byte row granularity, ≥20 GB/s / ≤2 µs per row) is the
  cheap first gate and is recorded as the concrete next step on that axis.
