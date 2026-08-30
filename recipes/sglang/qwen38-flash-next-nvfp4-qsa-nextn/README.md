# Qwen3.8-Flash-Next NVFP4 (QSA + NVMe PLE + NEXTN MTP)

> **Status: verified** on 2026-08-29, 1x DGX Spark GB10. The engine is an
> unmerged fork, pinned to an immutable commit as the lane checklist permits.
> All figures below were produced by the author on one machine and have not been
> reproduced by a second party.

## Summary

Serves `RadixArk/Qwen3.8-Flash-Next-NVFP4` at the full 262144-token context on one
DGX Spark (GB10, 128 GB unified memory), with NEXTN speculative decoding.

This is a different checkpoint and a different approach from the external NVFP4
SGLang reference linked in the [lane README](../README.md): that one is a W4A16
`sm121` build; this one is a ModelOpt NVFP4 checkpoint served through a fork that
streams the PLE table off NVMe.

## Why it fits in unified memory

The checkpoint's PLE n-gram table is 51.2B parameters -- 47.7 GiB in FP8 across 10
`model-plefp8-*.safetensors` shards. Held resident it does not leave room for a
long context beside the backbone. This recipe streams it: 16 row gathers per
token straight out of the model directory, dequantised `fp8 -> bf16 *
weight_scale` on the fly.

The working set is far smaller than the table. At 160 bytes per row and 16 rows
per token, decode touches ~2.5 KiB/token -- about 66 KiB/s at 26 tok/s. The
gather costs roughly 150 us against a 37.9 ms decode step (~0.4%), so PLE access
is not on the critical path; the backbone is.

This is consistent with the llama.cpp lane's
[`results/experiment-ledger.md`](../../../results/experiment-ledger.md), which
rejected whole-table PLE prewarming for depriving the 262k KV cache of headroom.
The same argument applies here in the other direction: on unified memory the free
GPU memory *is* the free host memory, so caching PLE rows in RAM spends KV
headroom on kilobytes-per-second of traffic. Left streaming deliberately.

## Runtime identity

| Component | Pin |
| --- | --- |
| Container image | `docker.io/scitrera/dgx-spark-sglang:0.5.17` |
| Image digest | `sha256:cc1cec4d023a88b6452b2766c2f47626ea9aedf6dced6e3f98a5f3bdf348b4d0` |
| Engine | `jzinno/sglang`, branch `feat/qwen4-nvme-ple` |
| Engine commit | `d4477bd298aef3edae611eb7b2e533d5526e324b` |
| flashinfer | 0.6.15.post1 (as shipped in the image) |
| Entrypoint | `python3 -m sglang.launch_server` |

The fork is mounted as a full-tree `PYTHONPATH` overlay rather than installed, so
the image stays unmodified and the engine can be swapped by repointing
`FORK_PATH`.

Upstream context: the PLE streaming backend is sglang PR #36567; the `qwen4_exp`
model support it builds on is PR #36585. Both were unmerged when this was written,
which is why `model.revision` is pinned to a commit but the engine is a branch.

## Model

- Repository: `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- Revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
- Quantization: ModelOpt NVFP4 (`nvidia-modelopt` 0.46.0, commit `87c9f8cf`),
  per the checkpoint's own `conversion_environment.json`
- Served with `--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass`
- Chat template: the checkpoint's own `chat_template.jinja`; no
  `--trust-remote-code` needed

## API shape

OpenAI-compatible, bound to `127.0.0.1:30000` by default.

- `/v1/chat/completions`, streaming and non-streaming
- `/health`, `/get_model_info`
- Tool calling verified through `tool_calls` with `finish_reason: tool_calls`
- Reasoning splits into `reasoning_content` (see *Parsers*)

## Reproducing

```bash
cp env.example .env    # edit MODEL_PATH and FORK_PATH
./run.sh
```

Cold start is dominated by weight loading: `load_weight=516 s` over 206 shards,
about 600 s to a healthy `/health`. A supervisor with a short start timeout will
kill it mid-load; budget 20 minutes.

## Benchmarks

Client: `results/raw/verify.py` (stdlib only), driving `/v1/chat/completions`
over HTTP with streaming. Every prompt carries a unique `Run marker` prefix so
the radix cache cannot serve a prior prefill. TTFT counts the first token of
*any* kind, reasoning included. Greedy (`temperature=0`) unless stated. One
warmup request precedes each phase. Raw logs in `results/raw/`, machine-readable
summary in `results/results.json`.

### Concurrency

Each stream sends a ~1.2k-token prompt and generates 192 tokens.

| Streams | Aggregate | Per-stream | TTFT p50 | TTFT p95 | Accept | Success | Free mem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.5 tok/s | 25.7 tok/s | 1.51 s | 1.51 s | 2.81 | 1/1 | 17.3 GB |
| 2 | 29.9 tok/s | 18.0 tok/s | 1.37 s | 2.33 s | 2.56 | 2/2 | 17.3 GB |
| 4 | 48.3 tok/s | 16.3 tok/s | 4.15 s | 4.15 s | 2.69 | 4/4 | 17.3 GB |
| 8 | 57.2 tok/s | 15.7 tok/s | 3.46 s | 16.73 s | 2.80 | 8/8 | 17.3 GB |

2.7x aggregate throughput from 1 to 8 streams with no failures. The cost is tail
latency: TTFT p95 reaches 16.7 s at 8 streams while p50 stays at 3.5 s. Pick 8
for batch throughput, fewer for interactive use.

### Context depth

Cold prefill, unique prefix per run.

| Prompt tokens | TTFT | Prefill | Decode | Accept |
| ---: | ---: | ---: | ---: | ---: |
| 7,806 | 3.73 s | 2,094 tok/s | 29.5 tok/s | 2.38 |
| 31,702 | 15.01 s | 2,112 tok/s | 27.4 tok/s | 3.02 |
| 63,892 | 30.88 s | 2,069 tok/s | 28.7 tok/s | 2.70 |
| 127,895 | 64.97 s | 1,968 tok/s | 25.9 tok/s | 2.73 |
| 199,931 | 137.42 s | 1,455 tok/s | 22.9 tok/s | 2.88 |

Prefill holds near 2,100 tok/s out to 128k and degrades only at 200k. Decode
drifts 29.5 -> 22.9 tok/s across the full range.

**Speculation does not degrade with depth on this runtime.** Accept length stays
between 2.38 and 3.02 from 8k to 200k, and decode falls by less than 25% across
a 25x increase in context. This is worth recording because
`results/experiment-ledger.md` parks MTP above 64k for the llama.cpp lane, where
draft KV overhead and full-context gather per draft step made speculation a net
loss at 229k. That does not reproduce here.

### Memory boundary

Ramping concurrent ~32k-token streams produced no request failures up to 16
(the scheduler queues beyond `--max-running-requests`, so residency is bounded
at 8 regardless).

Host memory is the real constraint, not the KV pool. Sampling `free` during 8
concurrent 32k streams:

| | |
| --- | --- |
| Peak host memory used | 120 GB of 121 GB |
| Minimum available | **1 GB** |
| KV pool utilisation at that moment | 0.59 |
| Request success | 8/8 |

The KV pool was only 59% full while the machine came within 1 GB of exhaustion.
The gap is PLE residency: the fork's `mmap` backend does not bound how much of
the 47.7 GiB table stays mapped (its only knobs are `PATH`, `BACKEND`,
`CACHE_PAGES`, `QUEUE_DEPTH`, `MAX_BATCH_PAGES`, `LOG_INTERVAL` -- there is no
RSS budget), and on Linux 6.x folio faulting inflates residency as described
under *mmap RSS growth* below.

Practical consequence: **concurrency x prompt length is the limit, not
concurrency alone.** Eight concurrent short prompts left 17.3 GB free; eight
concurrent 32k prompts left 1 GB. Treat 8 x 32k as the edge of the envelope on a
128 GB Spark and reduce `--max-running-requests` if serving long prompts
concurrently.

### Output validation

| Check | Result |
| --- | --- |
| GSM8K, first 40 of the official test split | **39/40 (97.5%)** |
| Needle retrieval at 0.1% / 40% / 90% depth, 106k haystack | **3/3** exact |
| Reasoning split | `17*23` -> content `391`, scratchpad in `reasoning_content`, no `<think>` leak |
| Tool calls | structured `tool_calls`, `finish_reason: tool_calls`, streaming and not |
| `enable_thinking: false` | content returned, `reasoning_content: null` |

The single GSM8K miss is item 12 (0-indexed); per-item records are in
`results/results.json`. GSM8K matches the 39/40 measured without speculation on
the same engine commit, so NEXTN is lossless here.

### Two measurement traps

**TTFT must count reasoning tokens.** With `--reasoning-parser qwen3` the think
block streams as `reasoning_content` deltas before any `content` delta appears.
Measuring time-to-first-`content` reports 5.4 s where the true figure is 173 ms.
Any TTFT number for a thinking model is wrong unless it counts
`reasoning_content`.

**The radix cache invalidates repeated prompts.** Re-sending an identical 54,660
token prompt returned TTFT of 0.39 s against 23.94 s cold -- a prefix cache hit,
not a measurement. Every prompt in this harness carries a unique prefix.

## Relationship to the llama.cpp lane

This is not a tuned variant of the llama.cpp recipes -- it shares no runtime,
checkpoint, quantization, or PLE strategy with them, so it is best read as an
independent path to the same goal rather than a comparison point.

| | llama.cpp lane | This recipe |
| --- | --- | --- |
| Runtime | llama.cpp (PR #27742) | SGLang (fork of PRs #36567 / #36585) |
| Checkpoint | UD-IQ4-XS | ModelOpt NVFP4 |
| PLE table | 26.8 GiB IQ4_NL on host DRAM, lazy mmap + `POSIX_MADV_RANDOM` | 47.7 GiB FP8 streamed from NVMe |
| PLE placement | `-ot per_layer_token_embd=CPU`, <1 GiB resident | mmap row gathers, RSS bounded by sweep |
| Speculation | draft-MTP tree | NEXTN chain |

The repository README warns against ranking figures across rows that differ in
binary or benchmark protocol, and that warning applies here with force: the
quantization differs, the prompts differ, and the depths measured do not line up.
Nothing below should be read as a speedup claim for one runtime over the other.

With that said, the shape of the tradeoff looks different enough to be worth
recording:

- **Shallow decode favours the llama.cpp lane.** 29 tok/s unpatched and 40.5
  tok/s with draft-MTP at 4k, against 26.4 tok/s here.
- **Deep-context prefill is where this configuration is strongest.** ~1850 tok/s
  sustained over a 110k prompt, against 339.5 tok/s at 128k for the tuned
  `-ub 1024` deep-context result in `results/experiment-ledger.md`. Depths differ
  by ~18k and prefill throughput falls with depth, so treat the ratio as
  indicative only -- but the gap is larger than that mismatch accounts for.
- **Speculation behaves differently at depth.** The ledger records draft-MTP on
  QSA kernels at 229,859 depth running *slower* than kernel AR (10.2 tok/s at 43%
  accept vs 11.55 tok/s). NEXTN here still contributes at 110k (~21 tok/s), with
  accept length holding in the 2.3-2.8 band. That divergence may be a property of
  the draft topology (tree vs chain) rather than of either runtime, and is not
  something this recipe has isolated.

A genuine head-to-head would need the same checkpoint, the same prompt set, and
matched depths on one machine. That has not been done, and none of the figures
above substitute for it.

## Output validation

- `17*23` returns `391` in `content`, with the scratchpad in `reasoning_content`
  and no `<think>` tag leaking into `content`
- Streaming and non-streaming tool calls parse into structured `tool_calls`
- `chat_template_kwargs={"enable_thinking": false}` returns content with
  `reasoning_content: null`
- Regression risk: changing the engine commit, `--mem-fraction-static`, or either
  parser flag invalidates these

## Spark observations

### Parsers

`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are both required.
Omitting them leaves the model's scratchpad in `message.content` with a stray
closing `</think>` and no opening tag.

Use plain `qwen3`, **not** `qwen3-thinking`: the latter hardcodes
`force_reasoning=True`, which swallows all output as `reasoning_content` when a
client sends `chat_template_kwargs={"enable_thinking": false}`, since that path
emits no `<think>` tags at all. Plain `qwen3` picks `force_reasoning` up from the
template's own toggle.

There is no `qwen3` *tool-call* parser -- argparse rejects it at startup. This
checkpoint's template emits the XML form
(`<tool_call><function=name><parameter=arg>`), which is `Qwen3CoderDetector`, not
the Qwen2.5 JSON-in-`<tool_call>` form that `qwen`/`qwen25` expect.
`--tool-call-parser auto --reasoning-parser auto` also resolves correctly against
this template.

### Version gate

The sgl-kernel gate asserts flashinfer 0.6.17; the image ships 0.6.15.post1.
Running with `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` was clean -- no API
problems in the QSA, NEXTN or PLE paths -- so the floor appears looser than the
gate suggests. Remove the override once an image ships 0.6.17.

### mmap RSS growth

The fork documents the `mmap` PLE backend as correctness-only, but it is
serviceable with one caveat. On Linux 6.x with large page-cache folios, each
random row fault maps a whole folio, so process RSS climbs toward the full
47.7 GiB even though only 16 rows per token are touched. `MADV_RANDOM` does not
prevent this -- it limits readahead I/O, not mapping-in of already-cached folios.
On unified memory the inflated RSS skews the free-memory reads used for KV sizing.

This extends rather than contradicts the llama.cpp lane's `POSIX_MADV_RANDOM`
finding: that advice still correctly suppresses readahead, but does not bound
residency once folios are cached. Bounding RSS needs an explicit
`MADV_DONTNEED` sweep above a budget; pages stay in page cache, so hot rows
re-fault at minor-fault cost.

### Podman and CDI

Tested under rootful podman rather than Docker. The NVIDIA CDI spec lands in
`/var/run/cdi`, not `/etc/cdi`, and `--device nvidia.com/gpu=all` resolves against
it. Switching `PLE_BACKEND` to `io_uring` requires a relaxed seccomp profile under
podman's default, supplied via `SECCOMP_PROFILE`; Docker's default profile differs
here.

### Monitoring

`nvidia-smi` reports `Not Supported` for memory fields under UMA on this box. The
free-memory figures above come from the server's own
`available_gpu_mem` line and from `free -g`.

### Long-document testing

Raw `/generate` on a large non-chat text returns an instant EOS. Use the chat
template for long-context tests or the result is misleading.

## Not yet covered

- **Multi-node.** Single-GPU only; no tensor-parallel or disaggregated layout.
- **OOM recovery.** The memory boundary is characterised but no recovery
  procedure is documented, and no buffer-cache flush was needed or tested.
- **Sustained load.** Phases are short bursts; there is no multi-hour soak, so
  drift in PLE residency over long uptime is unmeasured.
- **io_uring backend.** Only `mmap` was benchmarked. `io_uring` with
  `SGLANG_QWEN4_PLE_NVME_CACHE_PAGES` is untested here and needs a relaxed
  seccomp profile under podman.
- **Temperature sweep is indicative only.** Two samples per temperature at
  T=0.0/0.7/1.0 showed accept length moving non-monotonically (2.05-2.75),
  consistent with content variance rather than sampler divergence. It rules out a
  large effect, not a small one.
- **Second-party reproduction.** All figures come from one machine and one
  operator.

## Identifiers

- Recipe ID: `sglang/qwen38-flash-next-nvfp4-qsa-nextn`
- Runtime ID: `sglang`
