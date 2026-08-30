# Prefill graph-rebuild share — direct measurement (2026-08-30)

Closes the "Prefill-stable QSA graph" axis with a direct measurement instead of the
vanished-era estimate.

## Method

`llama_context::process_ubatch` instrumented (env-gated `LLAMA_GRAPH_BUILD_TIME`,
`fprintf(stderr)`): wall time of `model.build_graph` + `ggml_backend_sched_alloc_graph`
per rebuild ubatch, and of `res->set_inputs`. Shipped fix build (`can_reuse` overrides
active; decode shows `graphs reused = 63`), 4k cold protocol, guard passing. Full lines:
`gbt-timing-lines.log`; run summary: `gbt-run-summary.txt`.

## Result

Prefill (8 ubatches: 512×7 + 325 + warmup shapes):

| n_tokens | graph build+alloc | set_inputs |
|---:|---:|---:|
| 512 (each of 7) | **3.6-5.3 ms** | 4.4-10.9 ms (grows with n_kv: PLE prev-walk) |
| 325 (last) | 5.2 ms | 8.2 ms |

**Total prefill rebuild cost: ~35 ms of the 9.2-11.2 s prompt eval = ~0.3-0.4%.**

Decode (n_tokens=1, first step builds): 3.6 ms build+alloc, then ~0.4 ms set_inputs
per reused step — consistent with the measured ~12% decode win coming from
orchestration collapse, not graph build alone (build is 3.6 ms; the reuse path also
eliminates sched resets and re-allocation).

## Conclusion

The nsys ~500 ms/ubatch CPU gaps were **never graph rebuild** — direct timing shows
build+alloc is 3.6-5.3 ms. The gaps are the PLE gather + host-side hash/dequant work
that runs inside and around the ubatch pipeline (outside the measured windows), plus
`sched` compute staging. Therefore:

- **Prefill-stable QSA graph (max-shape tensors + bias masking): CLOSED — the
  addressed cost does not exist.** Its vanished-era "cap" (~12-20% at 4k) was a
  page-cache artifact, same confound as the decode archaeology.
- The real prefill gap content (PLE host-side gather path at ~0.5 s scale) is already
  bounded: PLE staging is not H2D-bound (7 ms), not ATS-bandwidth-bound (microbench),
  and its hash half is already parallelized (PLE-MT). The residual is the host
  dequant/gather compute itself — the same floor the prior row-access profile
  bounded.

Prompt eval on this run: 9.22 s / 429 tok/s (warm-ish page cache), reuses 63.
