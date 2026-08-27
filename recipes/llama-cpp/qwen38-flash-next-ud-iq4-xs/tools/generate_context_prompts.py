#!/usr/bin/env python3
"""Generate deterministic, token-counted context prompts through llama-server."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

VOCABULARY = (
    "architecture", "benchmark", "cache", "context", "decode", "deterministic",
    "embedding", "engine", "generation", "gradient", "hardware", "inference",
    "latency", "memory", "model", "operator", "parallel", "parameter",
    "performance", "pipeline", "precision", "prefill", "prompt", "quantized",
    "request", "response", "runtime", "scheduler", "sequence", "server",
    "stream", "tensor", "throughput", "token", "validation", "vector",
    "attention", "kernel", "storage", "measurement", "reproducible",
    "configuration", "system", "compute", "bandwidth", "index", "expert",
    "layer", "safety", "threshold", "random", "document", "function",
    "variable", "network", "analysis", "quality", "result", "sample",
    "iteration", "buffer", "matrix", "process", "profile",
)
SEED = 380051
SUFFIX = (
    "\n\nSummarize the benchmark methodology and identify its key safety "
    "constraint.\n"
)


def deterministic_words(count: int, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    return [VOCABULARY[rng.randrange(len(VOCABULARY))] for _ in range(count)]


def tokenize_count(base_url: str, text: str, timeout: float) -> int:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"tokenization request failed: {exc}") from exc
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list):
        raise RuntimeError("tokenization response does not contain a tokens array")
    return len(tokens)


def largest_prefix_at_most(
    words: Sequence[str],
    token_limit: int,
    count_tokens: Callable[[str], int],
) -> tuple[str, int]:
    if token_limit < 1:
        raise ValueError("token_limit must be positive")
    low, high = 1, len(words)
    best_text = ""
    best_count = 0
    while low <= high:
        middle = (low + high) // 2
        text = " ".join(words[:middle]) + SUFFIX
        count = count_tokens(text)
        if count <= token_limit:
            best_text, best_count = text, count
            low = middle + 1
        else:
            high = middle - 1
    if not best_text:
        raise ValueError("token limit is too small for the prompt suffix")
    return best_text, best_count


def parse_targets(value: str) -> list[int]:
    try:
        targets = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("targets must be comma-separated integers") from exc
    if not targets or any(target < 1 for target in targets):
        raise argparse.ArgumentTypeError("targets must contain positive integers")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=parse_targets, required=True)
    parser.add_argument("--reserve-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reserve_tokens < 0:
        print("reserve-tokens must be nonnegative", file=sys.stderr)
        return 2
    if any(target <= args.reserve_tokens for target in args.targets):
        print("every target must exceed reserve-tokens", file=sys.stderr)
        return 2

    max_words = max(args.targets) * 2
    words = deterministic_words(max_words)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for target in args.targets:
        token_limit = target - args.reserve_tokens
        text, raw_tokens = largest_prefix_at_most(
            words,
            token_limit,
            lambda candidate: tokenize_count(args.base_url, candidate, args.timeout),
        )
        path = args.output_dir / f"ctx{target}.txt"
        path.write_text(text, encoding="utf-8")
        record = {
            "target_context": target,
            "reserve_tokens": args.reserve_tokens,
            "raw_tokens": raw_tokens,
            "bytes": path.stat().st_size,
            "path": str(path),
            "seed": SEED,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
