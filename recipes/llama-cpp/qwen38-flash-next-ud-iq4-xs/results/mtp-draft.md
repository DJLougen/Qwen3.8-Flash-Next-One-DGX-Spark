# Experimental MTP draft (`draft-mtp`, n-max 3)

> **Not the recipe default.** `run.sh` still launches unpatched AR with
> `SPEC_TYPE=none`. This used an isolated tree at
> `/home/djl/llama.cpp-qwen4exp-mtp` (commit `250b61446` plus a local port of
> closed llama.cpp [PR #27842](https://github.com/ggml-org/llama.cpp/pull/27842)).
> That PR is **not merged**.

## What was converted

`--mtp --outtype q8_0` from the local
`/home/djl/models/Qwen3.8-Flash-Next-FP8` checkpoint (architectures
`Qwen4ExpForConditionalGeneration`). Converter logs showed the 31 `mtp.*`
tensors as `torch.bfloat16` even though the repo is tagged FP8.

Output:

```text
/home/djl/models/Qwen3.8-Flash-Next-UD-IQ4_XS/mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf
```

34 tensors, **3.9 GiB**. Pairs with the existing Unsloth `UD-IQ4_XS` main GGUF;
the main weights were not requantized.

## Launch (measured)

```text
llama-server \
  -m UD-IQ4_XS-00001-of-00003.gguf \
  -c 4096 -np 1 -b 512 -ub 128 -t 12 \
  -fa on -lm mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU -ngl all -fit off \
  --spec-type draft-mtp \
  -md mtp-Qwen3.8-Flash-Next-FP8-Q8_0.gguf -ngld 99 \
  --spec-draft-n-max 3
```

Spark guard: 80 / 36 / 28 GiB, swap growth 1 GiB.

## Protocol

- Host: one DGX Spark GB10, CUDA 13.0.2, driver 580.159.03
- Chat completions, `temperature=0`, `max_tokens=51`, `thinking=false`
- Prompt: `Continue this sequence exactly:` plus `1 2 3 … 20`
- This is **not** the kernel-track hash protocol (`2689367b205c16ce`)

## Results (2026-08-28)

Server `slot print_timing` decode (`eval time`, 51 completion tokens):

| Run | Decode tok/s | ms/token | Draft accept | Mean draft len |
|---|---:|---:|---:|---:|
| warmup | 35.00 | 28.57 | 75.556% (34/45) | 3.27 |
| 1 | 37.14 | 26.93 | 75.556% (34/45) | 3.27 |
| 2 | **40.86** | 24.47 | 75.556% (34/45) | 3.27 |
| 3 | 40.48 | 24.71 | 75.556% (34/45) | 3.27 |

Steady median **~40.5 tok/s**. Unpatched AR on this GGUF is **~25 tok/s** on the
recipe short-prompt protocol → about **1.6×**, in line with the PR’s Strix Halo
median 1.63× at n-max 3. Do not paste this onto the 64k QSA-kernel 18.73 tok/s
figure; context, prompt, and binary differ.

Client wall-clock (includes TTFT) was ~28 tok/s on the same three runs.

## Do not

- Do not use `--spec-draft-n-max 8` (PR: slower than AR; rollback-slot cost).
- Do not treat this as `verified`. The llama.cpp PR is closed/unmerged, the
  converter was run against the FP8 tree, and only ctx 4096 was measured.
- Do not mix with `ngram-mod` copy-learning numbers.
