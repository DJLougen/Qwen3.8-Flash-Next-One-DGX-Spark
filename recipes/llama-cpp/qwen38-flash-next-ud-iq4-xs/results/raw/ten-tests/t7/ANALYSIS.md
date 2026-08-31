# T7 — PLE row cache across requests

Date: 2026-08-31. Stacked on the T1 tree (`#ifdef GGML_PLE_CACHE` FIFO of
dequantized f32 rows in `ggml_compute_forward_get_rows_q`, keyed
`(src0->data, row)`, shape gate `type == IQ4_NL && ne00 == 160`). Patch:
`patches/tt10-t7.patch`. Built as `build-t7cache` with
`-DCMAKE_CXX_FLAGS=-DGGML_PLE_CACHE` (define **did** compile in — `strings` on
`libggml-cpu` contains `PLE cache engaged`).

## What was measured

Two warm domain-serving sequences (prose `@`-variants ×3 then ctx4096 once):

- `t7-cache` — first attempt, **wrong lib**: `llama-server` from `build/bin`
  resolved `libggml-cpu.so.0` via RUNPATH, so the cache-on lib never loaded.
  Numbers are a T1-equivalent control (prose median decode 25.9 tok/s, ctx4096
  TTFT 6.629 s, hash `99a15d5b`).
- `t7-cache2` / `t7-diag` — rerun from `build-t7cache/bin/llama-server`
  (RUNPATH pins the cache-on dir). Still **zero** `PLE cache engaged` lines.
  Unconditional first-call dump inside `ggml_compute_forward_get_rows_q`
  (`tt10-t7-diag`) also never printed, despite the server serving real requests
  (27 `print_timing` lines).

GGUF confirms the PLE tensor is exactly the planned signature:
`per_layer_token_embd.weight` `ne=(160, 320001536)` `type_id=20` (`GGML_TYPE_IQ4_NL`).
So the shape key is not why it never logged. The function **never runs**.

## Root cause

On this binary the PLE gather is `ggml_get_rows(ctx0, model.per_layer_tok_embd, rows)`
and CUDA `ggml_cuda_op_get_rows` (`ggml/src/ggml-cuda/getrows.cu`) claims
`GGML_OP_GET_ROWS`. The CPU `get_rows_q` dequant path — T7's cache site — is
not on the PLE hot path. (This also reframes T1: that win is other CPU GET_ROWS
sites, not PLE IQ4_NL dequant.)

## Verdict

**Rejected (`get_rows_q` path absent / tensor-identification-ambiguous).** Plan
contingency: do not ship a heuristic that could cache a non-PLE tensor; the
shape key matched but the intercept site is wrong. A PLE row cache would have
to live in the CUDA get_rows kernel or a server-side host cache in front of
it — out of scope.

Evidence: this directory (`t7-cache*`, `t7-diag*`, patch).
