#!/usr/bin/env python3
"""Run exactly two synchronized streaming requests against llama-server."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from stream_benchmark import (
    append_jsonl,
    build_request_metadata,
    run_streaming_request,
)

Runner = Callable[..., Any]


def parse_prompt_spec(value: str) -> tuple[str, Path]:
    request_id, separator, path = value.partition("=")
    if not separator or not request_id or not path:
        raise argparse.ArgumentTypeError("prompt must use REQUEST_ID=PATH")
    return request_id, Path(path)


def run_concurrent_batch(
    prompts: Sequence[tuple[str, Path, str]],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    runner: Runner = run_streaming_request,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(prompts) != 2:
        raise ValueError("exactly two prompts are required")

    barrier = threading.Barrier(3)

    def worker(request_id: str, prompt_path: Path, prompt_text: str) -> dict[str, Any]:
        request = build_request_metadata(
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            prompt_file=prompt_path,
            prompt_text=prompt_text,
        )
        request["concurrency"] = 2
        barrier.wait()
        started_at = time.perf_counter()
        metrics, error = runner(
            base_url=base_url,
            model=model,
            prompt=prompt_text,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        ended_at = time.perf_counter()
        return {
            "request_id": request_id,
            "success": error is None and metrics is not None,
            "metrics": metrics,
            "error": error,
            "request": request,
            "worker_wall_sec": ended_at - started_at,
            "worker_started_monotonic": started_at,
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, *prompt) for prompt in prompts]
        batch_started = time.perf_counter()
        barrier.wait()
        results = [future.result() for future in futures]
        batch_ended = time.perf_counter()

    output_tokens = sum(
        int(result["metrics"]["output_tokens"])
        for result in results
        if result["success"] and result["metrics"].get("output_tokens") is not None
    )
    batch_wall_sec = batch_ended - batch_started
    batch = {
        "batch_wall_sec": batch_wall_sec,
        "successful_requests": sum(1 for result in results if result["success"]),
        "total_output_tokens": output_tokens,
        "aggregate_output_tps": output_tokens / batch_wall_sec if batch_wall_sec > 0 else None,
        "start_skew_sec": abs(
            results[0]["worker_started_monotonic"]
            - results[1]["worker_started_monotonic"]
        ),
    }
    return results, batch


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    request_rows = [
        record
        for record in measured
        if record["record_type"] == "concurrent_request"
    ]
    request_ids = sorted({record["request_id"] for record in request_rows})
    per_request: dict[str, Any] = {}
    for request_id in request_ids:
        rows = [
            record
            for record in request_rows
            if record["request_id"] == request_id and record["success"]
        ]
        per_request[request_id] = {
            "successful_runs": len(rows),
            "ttft_median_sec": median_or_none(
                [
                    float(row["metrics"]["ttft_sec"])
                    for row in rows
                    if row["metrics"].get("ttft_sec") is not None
                ]
            ),
            "decode_median_tps": median_or_none(
                [
                    float(row["metrics"]["generation_tps"])
                    for row in rows
                    if row["metrics"].get("generation_tps") is not None
                ]
            ),
            "output_sha256": sorted(
                {
                    row["metrics"]["output_sha256"]
                    for row in rows
                    if row["metrics"].get("output_sha256")
                }
            ),
            "errors": [
                row["error"]
                for row in request_rows
                if row["request_id"] == request_id and row["error"]
            ],
        }

    batch_rows = [
        record
        for record in measured
        if record["record_type"] == "concurrent_batch"
    ]
    return {
        "record_type": "concurrent_aggregate",
        "measured_batches": len(batch_rows),
        "successful_batches": sum(
            row["successful_requests"] == 2 for row in batch_rows
        ),
        "batch_wall_median_sec": median_or_none(
            [float(row["batch_wall_sec"]) for row in batch_rows]
        ),
        "aggregate_output_median_tps": median_or_none(
            [
                float(row["aggregate_output_tps"])
                for row in batch_rows
                if row["aggregate_output_tps"] is not None
            ]
        ),
        "start_skew_max_sec": max(
            (float(row["start_skew_sec"]) for row in batch_rows),
            default=None,
        ),
        "per_request": per_request,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", type=parse_prompt_spec, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--jsonl-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.prompt) != 2:
        print("exactly two --prompt REQUEST_ID=PATH arguments are required", file=sys.stderr)
        return 2
    if args.max_tokens != 64:
        print("this concurrency probe requires --max-tokens 64", file=sys.stderr)
        return 2
    if args.warmup_count < 0 or args.repetitions < 1 or args.timeout <= 0:
        print("invalid warmup, repetition, or timeout value", file=sys.stderr)
        return 2

    prompts = []
    try:
        for request_id, path in args.prompt:
            prompts.append((request_id, path, path.read_text(encoding="utf-8")))
    except OSError as exc:
        print(f"failed to read prompt: {exc}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    total_batches = args.warmup_count + args.repetitions
    for batch_index in range(total_batches):
        warmup = batch_index < args.warmup_count
        results, batch = run_concurrent_batch(
            prompts,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        for result in results:
            result.update(
                {
                    "record_type": "concurrent_request",
                    "batch_index": batch_index,
                    "warmup": warmup,
                }
            )
            result.pop("worker_started_monotonic", None)
            records.append(result)
            metrics = result.get("metrics") or {}
            print(
                f"batch={batch_index} request={result['request_id']} success={result['success']} "
                f"ttft={metrics.get('ttft_sec')} decode_tps={metrics.get('generation_tps')} "
                f"error={result['error']}"
            )
        batch.update(
            {
                "record_type": "concurrent_batch",
                "batch_index": batch_index,
                "warmup": warmup,
            }
        )
        records.append(batch)
        print(
            f"batch={batch_index} successful={batch['successful_requests']}/2 "
            f"wall={batch['batch_wall_sec']:.3f}s aggregate_tps={batch['aggregate_output_tps']}"
        )

    aggregate = aggregate_records(records)
    records.append(aggregate)
    append_jsonl(args.jsonl_out, records)
    print(json.dumps(aggregate, sort_keys=True))
    return 0 if aggregate["successful_batches"] == args.repetitions else 1


if __name__ == "__main__":
    raise SystemExit(main())
