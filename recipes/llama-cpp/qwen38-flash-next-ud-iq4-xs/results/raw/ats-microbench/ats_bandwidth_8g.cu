// ATS bandwidth microbenchmark for zero-copy host PLE gather (2026-08-30).
// Mimics PLE access: read 90-byte rows from a host-mmap'd buffer at 16
// pseudo-random indices per iteration, from a CUDA kernel over GB10 ATS.
// Gate (from the optimization plan): sustained >= 20 GB/s and <= 2 us/row
// to consider GPU-side gather of the 26.8 GiB IQ4_NL PLE table viable.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>

#define MiB (1024.0*1024.0)
#define GiB (1024.0*1024.0*1024.0)

__global__ void ats_gather_kernel(const uint8_t * __restrict__ table,
                                  const int64_t * __restrict__ idx,
                                  float * __restrict__ sink,
                                  int iters, int rows_per_iter,
                                  int64_t table_rows) {
    float acc = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < rows_per_iter; ++r) {
            const int64_t i = idx[it * rows_per_iter + r];
            // 90-byte row, read as 22.5 floats worth of bytes -> read 90 bytes via
            // 3x 30-byte strided uint4-ish accesses; simplest: sum 90 uint8 loads
            // via reinterpret to keep the access pattern honest (scattered rows).
            const uint8_t * row = table + i * 90;
            #pragma unroll
            for (int b = 0; b < 90; b += 16) {
                // 16-byte vector load where aligned enough is not guaranteed;
                // fall back to byte-wise sum to avoid misalignment faults.
                float4 v;
                memcpy(&v, row + b, 16); // compiler -> ld.global.v4 when aligned; else byte path
                acc += v.x + v.y + v.z + v.w;
            }
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) { sink[0] = acc; } // keep the loads alive
}

int main() {
    const int64_t table_rows = (int64_t)(8.0 * GiB / 90); // 1 GiB table
    const size_t table_bytes = (size_t)table_rows * 90;
    uint8_t * table = (uint8_t *)aligned_alloc(4096, table_bytes);
    memset(table, 1, table_bytes);

    // register the host buffer for UVA/ATS access
    void * dev_ptr = nullptr;
    cudaError_t err = cudaHostRegister(table, table_bytes, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&dev_ptr, table, 0);
    if (!dev_ptr) { fprintf(stderr, "no device pointer\n"); return 1; }

    // pseudo-random row indices: 16 per iteration, LCG for reproducibility
    const int rows_per_iter = 16;
    const int iters = 200000;
    size_t idx_bytes = (size_t)iters * rows_per_iter * sizeof(int64_t);
    int64_t * hidx = (int64_t *)malloc(idx_bytes);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < iters * rows_per_iter; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        hidx[i] = (int64_t)(s % (uint64_t)table_rows);
    }
    int64_t * didx = nullptr;
    cudaMalloc(&didx, idx_bytes);
    cudaMemcpy(didx, hidx, idx_bytes, cudaMemcpyHostToDevice);
    float * dsink = nullptr;
    cudaMalloc(&dsink, 4);

    const int total_rows = iters * rows_per_iter;
    const double bytes_read = (double)total_rows * 90.0;

    // warmup (ATS fault-in)
    ats_gather_kernel<<<1, 128>>>((const uint8_t *)dev_ptr, didx, dsink, 1000, rows_per_iter, table_rows);
    cudaDeviceSynchronize();

    // timed run: many blocks to saturate the GPU
    const int blocks = 128;
    auto t0 = std::chrono::high_resolution_clock::now();
    ats_gather_kernel<<<blocks, 128>>>((const uint8_t *)dev_ptr, didx, dsink,
                                       iters / blocks, rows_per_iter, table_rows);
    err = cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err)); return 1; }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double gbps = bytes_read / secs / GiB;
    double us_per_row = secs * 1e6 / (double)(iters / blocks * blocks * rows_per_iter);
    printf("ATS gather microbenchmark\n");
    printf("table: %.2f GiB, row: 90 B, %d rows/iter, %d total rows read\n",
           table_bytes / GiB, rows_per_iter, iters / blocks * blocks * rows_per_iter);
    printf("time: %.4f s\n", secs);
    printf("sustained: %.2f GB/s (gate: >= 20)\n", gbps);
    printf("per-row: %.3f us (gate: <= 2.0)\n", us_per_row);
    printf("VERDICT: %s\n", (gbps >= 20.0 && us_per_row <= 2.0) ? "PASS - zero-copy ATS viable" : "FAIL - ATS not viable at 90B granularity");

    cudaFree(didx); cudaFree(dsink);
    cudaHostUnregister(table);
    free(table); free(hidx);
    return 0;
}
