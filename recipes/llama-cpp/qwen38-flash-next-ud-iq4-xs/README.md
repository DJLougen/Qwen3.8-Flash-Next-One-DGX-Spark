# Qwen3.8 Flash Next UD-IQ4_XS on DGX Spark

> **Draft, measured on one DGX Spark.** This is the **public default** for the
> UD-IQ4_XS GGUF: unpatched llama.cpp [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
> commit `250b61446`. Status stays `draft` because that architecture PR is still
> open.
>
> Overnight QSA kernels are a **second config**, not this `run.sh`:
> [`../qwen38-flash-next-ud-iq4-xs-qsa/`](../qwen38-flash-next-ud-iq4-xs-qsa/).

## Scope

This lane targets the three-shard `UD-IQ4_XS` GGUF from
[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
at immutable revision `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f`.

The model's “51B n-gram” feature is its 20-million-row PLE embedding table
(`per_layer_token_embd.weight`). It is unrelated to llama.cpp's optional
`--spec-type ngram-mod` speculative decoder.

This recipe keeps the PLE table on CPU-backed mmap storage with lazy row reads:

```text
-lm mmap
--tensor-read-lazy on
-ot per_layer_token_embd=CPU
```

That still uses every requested PLE row while avoiding roughly 27 GiB of
always-resident table memory.

## Pinned runtime

- llama.cpp PR: [`ggml-org/llama.cpp#27742`](https://github.com/ggml-org/llama.cpp/pull/27742)
- Tested commit: `250b61446efc91e3a179c8677956f2667c8fbda0`
- Build: Release, `GGML_CUDA=ON`, CUDA architecture `121a-real`
- CUDA toolkit: 13.0.2
- NVIDIA driver: 580.159.03
- Device: NVIDIA GB10

Do not use the older `/home/djl/llama.cpp` build; it does not support
`qwen4exp`. The tested Spark tree is `/home/djl/llama.cpp-qwen4exp`.

## Artifact verification

Expected files under
`/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/`:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

Total: `93,682,584,224` bytes. Verify before launching:

```bash
sha256sum "$MODEL_DIR"/*.gguf
```

## Safe launch

Copy this recipe directory to Spark or set `SPARK_GUARD` to the deployed guard,
then review `env.example`:

```bash
set -a
source env.example
set +a
./run.sh
```

Defaults:

- loopback bind only
- context 4,096
- one server slot
- prompt cache disabled for honest TTFT
- F16 KV
- all normal layers on CUDA
- PLE lazy mmap on CPU
- no speculative decoding
- 80 GiB minimum `MemAvailable` before start
- SIGTERM below 36 GiB; SIGKILL below 28 GiB
- soft stop if swap grows by more than 1 GiB

Binding outside loopback requires `API_KEY`.

`SPEC_TYPE=ngram-mod` is accepted for explicitly labeled copy/edit experiments,
but it is not the default and must never be reported as general model speed.

## Measured performance

The selected short-prompt configuration reached:

- **Short prompt only** (`prompts/short.txt`, 76 tokens, `b2048`/`ub512`, post-warmup): **~0.15 s** TTFT, **~29 tok/s** decode (2026-08-29 regression)
- **Varied task-shape prompts** (`b512`/`ub128`, 2026-08-27): still **~24 tok/s** — not remeasured at `b2048`/`ub512` (see `results/summary.md`)
- cold 4k-target depth (~3,955 tokens, verification rerun): **12.65 s TTFT**, **~318 tok/s prefill** (below plan bar ≥380; TTFT ~flat vs 12.38 s at `b512`)
- native 262,144-token allocation: successful
- 229,874-token prompt: **1,218.85-second TTFT**, **5.60 tok/s decode**
- minimum available memory in the full-depth run: **39.53 GiB**
- parallel 2 at 8,192 total context: **0.853-second median TTFT per request**, **20.68 tok/s per request**, **32.82 aggregate output tok/s**, with 50.61 GiB minimum available memory

See [`results/summary.md`](results/summary.md) for configuration sweeps,
context-depth results, task-shape data, raw-file provenance, and rejected
experiments.

Experimental QSA CUDA kernels are documented in the sibling recipe
[`../qwen38-flash-next-ud-iq4-xs-qsa/`](../qwen38-flash-next-ud-iq4-xs-qsa/)
(patch, locked hashes, separate prompt protocol).

NVFP4 on 2× GB10 (SGLang) is documented separately; see
[`results/nvfp4-sglang-comparison.md`](../../../results/nvfp4-sglang-comparison.md).

Experimental MTP draft (isolated tree, not `run.sh`): ctx 4096 **~40.5 tok/s**
decode, 75.6% accept, n-max 3. See [`results/mtp-draft.md`](results/mtp-draft.md).

## Parked post-gap work (not in `run.sh`)

From the 2026-08-29 TTFT-gap plan; **documented only**, not implemented:

1. **Zero-copy PLE gather** — GPU reads host-resident PLE rows over NVLink-C2C
   ATS instead of per-row mmap faults on the 16 CPU gathers per token; long
   prefill remains bound (229k **~5.3 ms/token**; QSA decode **~2×** with TTFT
   **~flat**). See [`results/summary.md`](results/summary.md).
2. **MTP draft-layer follow-on** — beyond QSA-wired `graph_mtp` (accept unchanged
   at 4k/64k); see [`results/mtp-draft.md`](results/mtp-draft.md).

Details: [`results/summary.md`](results/summary.md) § Parked post-gap work.

## Reproduce benchmark inputs

Start the server at the desired context, then generate deterministic prompt
files through its tokenizer:

```bash
python3 tools/generate_context_prompts.py \
  --base-url http://127.0.0.1:8081 \
  --output-dir /tmp/qwen38-context-prompts \
  --targets 1024,4096,16384,32768,65536,131072,230000
```

Measure streaming TTFT and post-first-token decode:

```bash
python3 tools/stream_benchmark.py \
  --base-url http://127.0.0.1:8081 \
  --model qwen38-ud-iq4-xs \
  --prompt-file prompts/short.txt \
  --max-tokens 128 \
  --context-label ctx4096-short \
  --warmup-count 1 \
  --repetitions 5 \
  --timeout 300 \
  --jsonl-out results/raw/local-run.jsonl
```

Reproduce the proven two-slot probe only with the stricter guard floors:

```bash
PARALLEL=2 \
CONTEXT_SIZE=8192 \
MIN_START_MEM_GIB=100 \
SOFT_STOP_MEM_GIB=45 \
HARD_KILL_MEM_GIB=38 \
./run.sh

python3 tools/concurrent_benchmark.py \
  --base-url http://127.0.0.1:8081 \
  --model qwen38-ud-iq4-xs \
  --prompt alpha=prompts/short.txt \
  --prompt beta=prompts/concurrency-b.txt \
  --max-tokens 64 \
  --repetitions 3 \
  --jsonl-out results/raw/concurrency-np2-ctx8192.jsonl
```

Task-shape experiments use `--variation-placeholder @` so repeated requests
cannot train a speculative cache on an identical output.

## Context length (ten-tests 2026-08-30/31)

True-cold unless noted. Evidence: [`results/raw/ten-tests/`](results/raw/ten-tests/).

| Ctx | Config | TTFT (s) | Prefill tok/s | Decode tok/s | Hash | Guard min GiB |
|---|---|---:|---:|---:|---|---:|
| ~76 (warm) | Gate 0 ub512 | 0.149 | — | 28.6 | `cb7904d8` | — |
| ~76 (warm) | T1 GET_ROWS | 0.144 | — | 26.66 | `cb7904d8` | — |
| ~76 (warm) | T5 kmtp+MTP | 0.322 | — | 26.54 | — | — |
| **4k** | Gate 0 ub512 | 10.791 | 372.5 | 24.6 | `99a15d5b` | — |
| **4k** | T1 GET_ROWS ub512 | **6.806** | **599.9** | 23.42 | `99a15d5b` | 50.86 |
| **4k** | T3 ub1024 | **9.199** | 438.9 | 23.79 | `06124a4b` | — |
| **4k** | T9 kmtp ub512 | 12.011 | 335.0 | 24.95 | `c64973d8` | 50.86 |
| **64k** | Gate 0 ub512 | 170.663 | 384.7 | 14.5 | `b641e2eb` | — |
| **64k** | T1 GET_ROWS ub512 | **131.94** | **498.1** | 13.96 | `b641e2eb` | — |
| **64k** | T3 ub1024 | **160.99** | 408.4 | 14.35 | `a81283e2` | — |
| **64k** | T9 kmtp ub512 | 166.57 | 393.9 | **20.44** | `b0ea9f23` | 47.71 |
| **128k** | era f16 ub1024 | 386.77 | 339.5 | — | — | — |
| **128k** | T4 kvq8 ub1024 | 397.5 | 330.4 | 9.78 | `9b622db0` | 44.2 |
| **230k / 262k** | T4 kvf16 | — | — | — | — | **35.77 breach** |
| **230k / 262k** | T4 kvq8 ub1024 | 901.65 | 255.4 | 6.20 | `1cda86a2` | 37.97 |
| **230k** | T9 kmtp ub1024 | 922.76 | 249.6 | **12.94** | `e2875202` | 36.12 |

T1 owns 4k/64k prefill. T9 owns 64k/230k decode (QSA). T4 `q8_0` KV is the only
config that loads 262k under the 36 GiB floor. Recipe default remains F16 KV at
4k; use `-ctk q8_0 -ctv q8_0` only for 262k.

## Rejected experiments and host safety

- A port of the comparison repository's Qwen4Exp graph-reuse patch reached
  `graphs reused = 127`, then segfaulted on the second request. It was removed.
- Kernel logs showed NVIDIA `NV_ERR_NO_MEMORY` during that experiment.
- The host later rebooted without a clean shutdown. Multiple NVIDIA OOM events
  from other workloads were also present, so sole causality is unproven.
- Whole-table PLE prewarming is excluded: it consumes tens of GiB and the
  comparison repository's later measurements show no prose benefit.
- Switching PLE lazy-range advice from `POSIX_MADV_RANDOM` to `NORMAL` or
  `SEQUENTIAL` preserved output hashes but grew table residency (0.63 GiB and
  0.14 GiB vs 0.01 GiB) and worsened cold TTFT. Keep `RANDOM`.
- Quantized KV is **not** the 4k default (F16). T4 showed `-ctk q8_0 -ctv q8_0`
  **enables 262k** under the 36 GiB floor (f16 breaches at 35.77 GiB during
  load; q8_0 holds 37.97 GiB). 128k q8_0 is not a TTFT win (397.5 vs 386.77 s).

After a reboot or NVIDIA OOM, do not immediately relaunch. Confirm host uptime,
GPU process state, `nvidia-smi`, disk headroom, and `MemAvailable` first.

## Provenance and credit

- Qwen model weights: Qwen Community License
- GGUF conversion: Unsloth
- Runtime architecture support: llama.cpp PR #27742
- Comparison methodology reviewed from
  [`0xBakeer/qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark)
  commit `4c6fc3af429bff5c472511cf965751eac6b7caf2`

That comparison repository is MIT licensed, Copyright (c) 2026 0xBakeer. Its
methodology informed the varied task suite and the now-rejected graph-reuse
experiment; this recipe does not ship its patch or tools.

## Status

- Recipe ID: `llama-cpp/qwen38-flash-next-ud-iq4-xs`
- Runtime ID: `llama-cpp`
- Manifest status: `draft`
- Publication status: in-tree draft; SGLang/vLLM lanes remain empty
