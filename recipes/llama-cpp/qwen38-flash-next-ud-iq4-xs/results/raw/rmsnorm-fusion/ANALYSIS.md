# Fused Zero-Centered RMSNorm — investigated, not indicated (2026-08-30)

Ledger axis: "Fuse $(1+w)$ weight offset directly into GEMM input scaling;
eliminates 1 separate kernel launch per transformer block."

## Verdict

**Not indicated.** The axis rests on two premises, both false on this tree, and the
realistic remaining prize is under the ≥3% gate.

## Evidence (three-source, file:line verified)

1. **$(1+w)$ is already folded.** No runtime fold exists anywhere;
   `qwen4exp.cpp:210-211`: "the converter folded each gamma to $(1 + w)$" — the
   checkpoint tensors already carry $(1+\gamma)$. Every site applies the weight
   raw via `ggml_mul`.

2. **GEMM-alpha fusion is structurally impossible.** The norm weights are
   per-channel vectors ($[hc\_dim]$, $[n\_embd]$, $[idx\_dim]$, $[n\_embd\_head]$);
   cublas GEMM alpha is a compile-time scalar `1.0f` (`ggml-cuda.cu:1373/1387/1401`,
   used at `:1507/:1522`), and `mul_mat_q` has no per-row/per-channel input scaling.
   Folding into the GEMM B matrix offline would break quantization.

3. **The tree already fuses norm+mul where legal.** A graph-level fuser (on by
   default; `GGML_CUDA_DISABLE_FUSION` kills it) recognizes `{RMS_NORM, MUL}`,
   `{RMS_NORM, MUL, ADD}`, `{RMS_NORM, MUL, ROPE(,VIEW,SET_ROWS)}` chains
   (`ggml-cuda.cu:3962-3980`) and routes them into
   `rms_norm_f32<block, do_multiply[, do_add]>` — one launch. The separate
   `k_bin_bcast<op_mul>` kernels in the nsys capture are the sites that FAIL the
   fusion gates, and qwen4exp's sites fail them **structurally**, not fixably:
   - hc_mix + PLE grouped norms (97 + 3·n_ple sites/token): a `RESHAPE` sits
     between `rms_norm` and `mul` (`qwen4exp.cpp:213→214`, `:1123-1124`) and is
     *required* — the $[hc\_dim]$ weight cannot broadcast against the
     $[n\_embd, hc, T]$ norm output without it.
   - attn q/k + indexer q/k norms (48 sites/token): inputs are strided views
     (`:670-672`), failing the row-contiguity gate (`ggml-cuda.cu:5175-5178`).
   - ssm_norm (36 sites/token): a second gate-mul follows the weight-mul
     (`:383-386`), which the two-op fusion pattern cannot absorb.
   Additionally, hc_mix's scaled output has three consumers (two GEMMs + the
   gate mul, `:217/:222/:236`), so even a relaxed fuser covering one consumer
   would not eliminate the value's recompute.

4. **Prize sizing.** ~181 + 3·n_ple norm-weight mul launches per decode token at
   ~3-6 µs each ≈ 0.2-0.4 ms of the ~38-41 ms/tok decode budget (**~0.5-1%**).
   Prefill: `rms_norm_f32` 158.5 ms + the fusable share of the 767 ms
   `k_bin_bcast` family in the 4k window ≈ ≤2.6% of the 11.8 s prompt eval,
   with realistic recovery ~1-2% — **under the ≥3% gate**.

5. **Side-idea rejected.** Offline-folding `ple_norm_conv` into `ple_conv1d`
   (`w[c]*wk[k,c]` at load time) is exact depthwise-conv math and free at
   runtime, but changes fp32 rounding order → hash-gate risk for ~0.1%.

## Sources

- `src/models/qwen4exp.cpp` norm sites: `:212-214` (hc_mix), `:383-386` (ssm_norm),
  `:511/:520` (indexer), `:674/:684` (attn q/k), `:830-831` (l2_norm, unweighted),
  `:1119-1148` (PLE grouped norms).
- `src/llama-graph.cpp:1580-1606` (`build_norm`: rms_norm then separate mul at `:1602`).
- `ggml/src/ggml.c:3155-3167` (`ggml_rms_norm_impl`: unary, no weight operand) and
  `:7691-7746` (`ggml_can_fuse_subgraph_ext` gates).
- `ggml/src/ggml-cuda/norm.cu` (`rms_norm_f32<block, do_multiply, do_add>` fused
  kernel; plain launcher takes no weight), `ggml-cuda.cu:3962-3980` (fusion
  dispatch), `:5175-5182` (contiguity gate), `:1373/1387/1401` (scalar alpha).
- nsys 4k window: `rms_norm_f32` 158.5 ms / 1589 instances; `k_bin_bcast` family
  766.8 ms / 9337 instances (`raw/nsys-4k/`).
