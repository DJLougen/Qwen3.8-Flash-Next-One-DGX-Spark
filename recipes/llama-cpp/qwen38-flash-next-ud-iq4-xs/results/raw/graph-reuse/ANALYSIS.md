# QSA graph-reuse fix — `can_reuse` overrides for `llm_graph_input_qsa` / `llm_graph_input_ple` (2026-08-30)

Patch: `qwen4exp-can-reuse.patch` (applies to `src/models/qwen4exp.cpp` on tree
`250b61446` + 7-file QSA diff; includes the PLE-MT multithreaded `set_input` content,
which had been lost from the Spark tree and was restored + verified in this session).

## What was done

`llm_graph_input_qsa` and `llm_graph_input_ple` inherit the base
`llm_graph_input_i::can_reuse` which returns **false unconditionally**
(`src/llama-graph.h:113-119`), so every ubatch on the qwen4exp arch rebuilds the full
compute graph (`graphs reused = 0`). The patch adds shape-stability overrides:

- **QSA**: all five input tensors (`k_idxs`, `cell_blk`, `blk_cells`, `blk_pos`,
  `bias`) compared against the shapes a rebuild would produce from live padded
  `n_kv` / `n_blocks = ceil(n_kv/ratio)` / `n_tps` / `n_stream`. Passes for decode
  steps inside one n_kv pad bucket; a bucket crossing correctly rebuilds — identical
  semantics to the dense path's `can_reuse_kq_mask`.
- **PLE**: `rows->ne[0] == ple_n_heads * n_tokens` (ubatch geometry only).
- **Both**: refresh the per-batch `mctx` member from `params.mctx` before returning —
  the stored context belongs to a destroyed batch; without the refresh the first
  reused decode step dereferences a dangling pointer (SIGSEGV, captured under GDB in
  `crash-backtrace-gdb.log`: `set_input_k_idxs` ← `llm_graph_input_qsa::set_input`).
  This refresh mirrors `llm_graph_input_attn_kv::can_reuse` /
  `llm_graph_input_mem_hybrid::can_reuse`.

## Verified gates (final build, 2026-08-30, guard 80/36/28 GiB all passing)

| Gate | Result |
|---|---|
| Short-prompt output hash (temp 0, 128 tok) | `cb7904d8097240a2bc32c77e27c03a924fcb972212566d14487d20d2aa687601`, 746 chars — **exact reference match** |
| Repeat identical request (0xBakeer segfault scenario) | **survived**; `graphs reused = 253` cumulative; decode 35.08 ms/tok (28.5 tok/s) |
| Short 128-tok decode | `graphs reused = 127` (every decode step after the first); 38.4-40.1 ms/tok |
| Cold 4k TTFT | 9.25-11.59 s (vs 13.18 s same-day no-override control cold); 315 cumulative reuses |
| 4k decode | 41.3-41.5 ms/tok |
| 4k output hash | `99a15d5b2d01b1e980b84f70...` (318 chars) — **byte-identical to the no-override control** run on the same tree (`control-nooverride-4k.jsonl`); the historical `c64973d8...` reference came from a now-vanished experimental library state (see below) and is not the clean-tree output |
| Guard floors | held throughout (min available ≈ 55 GiB) |

## Performance summary

- Decode: **~38.4-41.5 ms/tok vs 43.4-45.2 ms/tok** without the fix (~10-12% faster
  decode; short-repeat 28.5 tok/s vs 23-23.3 baseline on the same day).
- Warm short TTFT: 0.367 s (cached prompt, graph reuse active).
- Prefill is unchanged by design (padded n_kv grows each ubatch → correct rebuild),
  so cold 4k TTFT moves only with page-cache state, not with this patch.


## 64k depth verification (2026-08-30)

| Build | 64k TTFT | 64k decode | reuses |
|---|---:|---:|---:|
| Override build | 178.7 s (cold page cache) | **71.88 ms/tok** | 63 |
| No-override control (same day) | 180.3 s | 73.27 ms/tok | 0 |

Decode win at 64k is only **~1.9%**: at deep context the decode step is dominated by
the 64k KV gather (GPU work), not per-token graph rebuild + launch orchestration.
The fix's win scales inversely with per-token GPU cost — ~12% at short/4k
(38.4-41.5 vs 43.4-45.2 ms/tok), ~2% at 64k. Prefill unchanged by design at both
depths (2.72 vs 2.75 ms/tok server, within page-cache noise).

## Session archaeology (why prior numbers disagreed)

- The Aug-29 "PLE-MT" gate runs (`graphs reused = 127/63`, decode 38-40 ms/tok, 4k
  hash `c64973d8`) executed against a **now-vanished library state**: `llama-server`
  binaries are 72 KiB launchers that dlopen `libllama.so.0`/`libggml-cuda.so` from
  the build dir RUNPATH, so a copied binary runs whatever libs are current. On Aug 29
  17:34-19:24 the build-dir libs still carried 0xBakeer-era graph-reuse code; the
  tree was cleaned (`ggml-cuda.cu` restored 20:33, libs rebuilt 20:34) *after* those
  runs. All of today's runs on the clean tree — unpatched, PLE-MT, and override
  builds — show `graphs reused = 0` before this patch.
- The PLE-MT content itself had also been lost from `qwen4exp.cpp` on Spark; it was
  restored in this session and re-verified (short hash `cb7904d8` passes with and
  without the overrides).
- The repo's stored `patches/ple-multithreaded-set-input.patch` was corrupt
  (hunk-header arithmetic wrong, non-applicable); it was regenerated from the
  verified tree state.

## Ledger disposition

Verified win: decode-phase graph reuse for the qwen4exp hybrid arch, without the
0xBakeer patch's dynamic-layout memory corruption. Distinct from the hard-rejected
generic patch: this fix reuses only when input shapes are byte-stable and refreshes
per-batch contexts, which is why the repeat-request segfault scenario passes.
