# ATS bandwidth microbenchmark — zero-copy host PLE first gate (2026-08-30)

Plan (Experiment C, step 1): allocate a host buffer, read 90-byte rows at 16
pseudo-random indices per iteration from a CUDA kernel over GB10 ATS; gate
**sustained ≥ 20 GB/s** and **≤ 2 µs/row**. Runners: `ats_bandwidth.cu` (1 GiB
table) and `ats_bandwidth_8g.cu` (8 GiB table, cache-defeating), built
`nvcc -arch=sm_121a -O3`.

## Results

| Table | Time | Sustained | Per-row (amortized) | GB/s gate (≥20) | µs/row gate (≤2) |
|---|---:|---:|---:|---|---|
| 1 GiB | 0.0331 s | **8.11 GB/s** | 0.010 µs | FAIL | pass |
| 8 GiB | 0.0421 s | **6.36 GB/s** | 0.013 µs | FAIL | pass |

## Verdict per plan text: NEGATIVE — do not implement the ATS gather kernel

The 20 GB/s sustained gate fails at both table sizes (6.4-8.1 GB/s measured at
90-byte random granularity over `cudaHostRegisterMapped` + ATS).

## Reframe (why the gate's calibration matters more than the failure)

The 20 GB/s gate assumed the PLE gather is bandwidth-bound. It is not:

- PLE demand per 512-token ubatch: 16 lookups/token × 512 = 8,192 rows × 90 B
  ≈ **720 KiB per ubatch**, spread over a ~500 ms ubatch → ~**1.4 MB/s** demand.
- Even the measured 6.36 GB/s ATS floor is **~4,500× the demand**. Raw ATS
  bandwidth cannot be the deep-context prefill bottleneck.
- The per-row latency gate passes with 150× margin (amortized), so scattered-row
  latency is also not the wall.

Therefore the deep-context prefill floor (2.72 ms/tok at 64k, TTFT flat vs
`-ub 1024`) is **not** explained by ATS gather bandwidth either — pointing
instead at the surrounding per-row CPU work (hash/staging/dequant launch path)
or the small-transfer regime of the existing CPU-side gather, not the interconnect.
The zero-copy axis stays parked: its bandwidth premise is disproven, and any
reopen would need to target the CPU-side per-row orchestration cost, not ATS.
