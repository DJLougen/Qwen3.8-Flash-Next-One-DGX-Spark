#!/usr/bin/env python3
"""OpenAI-compatible streaming benchmark client for a local llama-server."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DONE_SENTINEL = "DONE"


class SseParseError(ValueError):
    """Raised when an SSE payload cannot be parsed."""


@dataclass(frozen=True)
class TimingMarks:
    request_start: float
    first_content: float | None
    end: float


def iter_sse_lines(response: Any) -> Iterable[str]:
    """Yield decoded SSE lines as the transport delivers them (strict UTF-8)."""
    for raw_line in response:
        yield raw_line.decode("utf-8")


def feed_sse_buffer(buffer: str, chunk: str) -> tuple[str, list[str]]:
    """Append *chunk* to *buffer* and return complete SSE data payloads."""
    buffer += chunk
    payloads: list[str] = []
    while True:
        separator = buffer.find("\n\n")
        if separator < 0:
            break
        event_block = buffer[:separator]
        buffer = buffer[separator + 2 :]
        payloads.extend(_payloads_from_event_block(event_block))
    return buffer, payloads


def flush_sse_buffer(buffer: str) -> list[str]:
    """Extract payloads from a trailing partial SSE buffer at stream end."""
    if not buffer.strip():
        return []
    return _payloads_from_event_block(buffer)


def _payloads_from_event_block(event_block: str) -> list[str]:
    payloads: list[str] = []
    for line in event_block.split("\n"):
        line = line.strip("\r")
        if not line.startswith("data:"):
            continue
        data = line[5:].lstrip()
        if data:
            payloads.append(data)
    return payloads


def parse_sse_payload(payload: str) -> dict[str, Any] | str | None:
    """Parse one SSE data payload into a JSON object, DONE, or None."""
    payload = payload.strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return DONE_SENTINEL
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SseParseError(f"invalid JSON in SSE payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SseParseError("SSE payload JSON must be an object")
    return parsed


def extract_output_deltas(event: dict[str, Any]) -> list[str]:
    """Return every nonempty content/reasoning delta from one event, in order."""
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        return []
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return []
    deltas: list[str] = []
    for field in ("content", "reasoning_content"):
        value = delta.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise SseParseError(
                f"choices[0].delta.{field} must be a string when present"
            )
        if value:
            deltas.append(value)
    return deltas


def extract_content_delta(event: dict[str, Any]) -> str:
    """Return the first streamed text or reasoning delta, if present."""
    deltas = extract_output_deltas(event)
    return deltas[0] if deltas else ""


def extract_usage(event: dict[str, Any]) -> dict[str, int] | None:
    """Return usage with completion_tokens when the server supplies it."""
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        return None
    if not isinstance(completion_tokens, int):
        raise SseParseError("usage.completion_tokens must be an integer when present")
    result: dict[str, int] = {"completion_tokens": completion_tokens}
    for key in ("prompt_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result


def compute_stream_metrics(
    marks: TimingMarks,
    usage: dict[str, int] | None,
    output_text: str = "",
) -> dict[str, float | int | str | None]:
    """Compute TTFT, decode throughput, end-to-end rate, and output fingerprint.

    Decode throughput uses ``(completion_tokens - 1) / (end - first_content)``
    under the assumption that the first streamed content delta accounts for
    exactly one completion token.
    """
    total_sec = marks.end - marks.request_start
    completion_tokens = usage.get("completion_tokens") if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    total_tokens = usage.get("total_tokens") if usage else None

    metrics: dict[str, float | int | str | None] = {
        "ttft_sec": None,
        "total_sec": total_sec,
        "output_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "post_first_tokens": None,
        "generation_tps": None,
        "end_to_end_output_tps": None,
        "output_chars": len(output_text),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }

    if completion_tokens is not None and total_sec > 0:
        metrics["end_to_end_output_tps"] = completion_tokens / total_sec

    if marks.first_content is None:
        return metrics

    metrics["ttft_sec"] = marks.first_content - marks.request_start
    post_first_sec = marks.end - marks.first_content

    if completion_tokens is not None:
        post_first_tokens = completion_tokens - 1
        metrics["post_first_tokens"] = post_first_tokens
        if post_first_tokens > 0 and post_first_sec > 0:
            metrics["generation_tps"] = post_first_tokens / post_first_sec

    return metrics


def reduce_stream_events(
    events: Sequence[tuple[float, dict[str, Any] | str]],
    request_start: float,
) -> tuple[TimingMarks, dict[str, int] | None, str]:
    """Fold timestamped stream events into timing marks, usage, and output text."""
    first_content: float | None = None
    end = request_start
    usage_snapshot: dict[str, int] | None = None
    output_parts: list[str] = []

    for timestamp, event in events:
        end = timestamp
        if event == DONE_SENTINEL:
            continue
        if not isinstance(event, dict):
            raise SseParseError("stream event must be a JSON object or DONE")

        for delta in extract_output_deltas(event):
            if first_content is None:
                first_content = timestamp
            output_parts.append(delta)

        usage = extract_usage(event)
        if usage is not None:
            usage_snapshot = usage

    return TimingMarks(request_start, first_content, end), usage_snapshot, "".join(
        output_parts
    )


def aggregate_metric_values(values: Sequence[float | int]) -> dict[str, float]:
    """Return median/min/max for a non-empty numeric sequence."""
    numeric = [float(value) for value in values]
    return {
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def compute_aggregate(
    run_records: Sequence[dict[str, Any]],
    *,
    context_label: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate successful measured (non-warmup) runs."""
    measured_runs = [
        record
        for record in run_records
        if record.get("record_type") == "run" and not record.get("warmup")
    ]
    successful_runs = [record for record in measured_runs if record.get("success")]

    metrics: dict[str, Any] = {}
    for key in (
        "ttft_sec",
        "total_sec",
        "generation_tps",
        "end_to_end_output_tps",
        "output_tokens",
        "post_first_tokens",
        "prompt_tokens",
        "total_tokens",
        "output_chars",
    ):
        values = [
            record["metrics"][key]
            for record in successful_runs
            if record.get("metrics", {}).get(key) is not None
        ]
        if values:
            metrics[key] = aggregate_metric_values(values)

    return {
        "record_type": "aggregate",
        "context_label": context_label,
        "measured_runs": len(measured_runs),
        "successful_runs": len(successful_runs),
        "request": request,
        "metrics": metrics,
    }


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append machine-readable JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def vary_prompt(
    prompt: str,
    placeholder: str | None,
    run_index: int,
) -> tuple[str, str | None]:
    if placeholder is None:
        return prompt, None
    if placeholder not in prompt:
        raise ValueError("variation placeholder is not present in prompt")
    variation_value = f"variant_{run_index:04d}"
    return prompt.replace(placeholder, variation_value), variation_value


def build_request_metadata(
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    prompt_file: Path,
    prompt_text: str,
) -> dict[str, Any]:
    prompt_bytes = prompt_text.encode("utf-8")
    return {
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "concurrency": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "prompt_file": str(prompt_file),
        "prompt_bytes": len(prompt_bytes),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
    }


def build_run_record(
    *,
    context_label: str,
    warmup: bool,
    run_index: int,
    request: dict[str, Any],
    success: bool,
    metrics: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "record_type": "run",
        "context_label": context_label,
        "warmup": warmup,
        "run_index": run_index,
        "success": success,
        "request": request,
        "metrics": metrics,
        "error": error,
    }


def _process_payloads(
    payloads: Sequence[str],
    first_content_time: float | None,
    usage_snapshot: dict[str, int] | None,
    end_time: float,
    output_parts: list[str],
) -> tuple[float | None, dict[str, int] | None, float, str | None]:
    current_first = first_content_time
    current_usage = usage_snapshot
    current_end = end_time
    for payload in payloads:
        current_end = time.perf_counter()
        try:
            event = parse_sse_payload(payload)
        except SseParseError as exc:
            return current_first, current_usage, current_end, str(exc)

        if event is None or event == DONE_SENTINEL:
            continue
        try:
            deltas = extract_output_deltas(event)
        except SseParseError as exc:
            return current_first, current_usage, current_end, str(exc)

        for delta in deltas:
            if current_first is None:
                current_first = time.perf_counter()
            output_parts.append(delta)

        try:
            usage = extract_usage(event)
        except SseParseError as exc:
            return current_first, current_usage, current_end, str(exc)
        if usage is not None:
            current_usage = usage

    return current_first, current_usage, current_end, None


def run_streaming_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, float | int | str | None] | None, str | None]:
    """Execute one streaming chat completion and return metrics or an error."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_start = time.perf_counter()
    buffer = ""
    first_content_time: float | None = None
    usage_snapshot: dict[str, int] | None = None
    output_parts: list[str] = []
    end_time = request_start

    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            try:
                for line in iter_sse_lines(response):
                    end_time = time.perf_counter()
                    buffer, payloads = feed_sse_buffer(buffer, line)
                    (
                        first_content_time,
                        usage_snapshot,
                        end_time,
                        error,
                    ) = _process_payloads(
                        payloads,
                        first_content_time,
                        usage_snapshot,
                        end_time,
                        output_parts,
                    )
                    if error is not None:
                        return None, error
            except UnicodeDecodeError as exc:
                return None, f"invalid UTF-8 in stream: {exc}"

            trailing_payloads = flush_sse_buffer(buffer)
            (
                first_content_time,
                usage_snapshot,
                end_time,
                error,
            ) = _process_payloads(
                trailing_payloads,
                first_content_time,
                usage_snapshot,
                end_time,
                output_parts,
            )
            if error is not None:
                return None, error
    except urllib.error.HTTPError as exc:
        body_snippet = exc.read(2048).decode("utf-8", errors="replace")
        return None, f"HTTP {exc.code}: {body_snippet[:500]}"
    except urllib.error.URLError as exc:
        return None, f"connection error: {exc.reason}"
    except TimeoutError:
        return None, f"request timed out after {timeout}s"

    marks = TimingMarks(request_start, first_content_time, end_time)
    output_text = "".join(output_parts)
    return compute_stream_metrics(marks, usage_snapshot, output_text), None


def _format_metric(value: float | int | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _print_run_line(label: str, record: dict[str, Any]) -> None:
    if not record.get("success"):
        print(f"{label}: failed ({record.get('error')})", file=sys.stderr)
        return
    metrics = record.get("metrics") or {}
    print(
        f"{label}: ok "
        f"ttft={_format_metric(metrics.get('ttft_sec'), suffix='s')} "
        f"total={_format_metric(metrics.get('total_sec'), suffix='s')} "
        f"tokens={_format_metric(metrics.get('output_tokens'))} "
        f"decode_tps={_format_metric(metrics.get('generation_tps'))} "
        f"e2e_tps={_format_metric(metrics.get('end_to_end_output_tps'))}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OpenAI-compatible streaming chat completions against "
            "a local llama-server."
        )
    )
    parser.add_argument("--base-url", required=True, help="Server base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to the prompt text file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Requested completion token limit",
    )
    parser.add_argument(
        "--context-label",
        required=True,
        help="Label recorded with each JSONL row",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=0,
        help="Warmup repetitions excluded from aggregates",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        required=True,
        help="Measured repetitions included in aggregates",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--variation-placeholder",
        default=None,
        help=(
            "Replace this literal in the prompt with a deterministic per-run "
            "variant to prevent cross-request repetition artifacts"
        ),
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        required=True,
        help="Append-only JSONL output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.warmup_count < 0:
        print("warmup-count must be >= 0", file=sys.stderr)
        return 2
    if args.repetitions < 1:
        print("repetitions must be >= 1", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("max-tokens must be >= 1", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("timeout must be > 0", file=sys.stderr)
        return 2

    try:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"failed to read prompt file: {exc}", file=sys.stderr)
        return 2
    if args.variation_placeholder is not None and args.variation_placeholder not in prompt:
        print("variation-placeholder is not present in prompt", file=sys.stderr)
        return 2

    request_metadata = build_request_metadata(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        prompt_file=args.prompt_file,
        prompt_text=prompt,
    )
    request_metadata["variation_placeholder"] = args.variation_placeholder

    run_records: list[dict[str, Any]] = []
    total_runs = args.warmup_count + args.repetitions

    for run_index in range(total_runs):
        warmup = run_index < args.warmup_count
        label = (
            f"warmup {run_index + 1}/{args.warmup_count}"
            if warmup
            else (
                f"run {run_index - args.warmup_count + 1}/"
                f"{args.repetitions}"
            )
        )

        run_prompt, variation_value = vary_prompt(
            prompt,
            args.variation_placeholder,
            run_index,
        )
        run_request = build_request_metadata(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            prompt_file=args.prompt_file,
            prompt_text=run_prompt,
        )
        run_request["variation_placeholder"] = args.variation_placeholder
        run_request["variation_value"] = variation_value

        metrics, error = run_streaming_request(
            base_url=args.base_url,
            model=args.model,
            prompt=run_prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        success = error is None and metrics is not None
        record = build_run_record(
            context_label=args.context_label,
            warmup=warmup,
            run_index=run_index,
            request=run_request,
            success=success,
            metrics=metrics,
            error=error,
        )
        run_records.append(record)
        _print_run_line(label, record)

    aggregate = compute_aggregate(
        run_records,
        context_label=args.context_label,
        request=request_metadata,
    )
    append_jsonl(args.jsonl_out, [*run_records, aggregate])

    aggregate_metrics = aggregate.get("metrics", {})
    print(
        "aggregate: "
        f"successful={aggregate['successful_runs']}/{aggregate['measured_runs']} "
        f"ttft_median={_format_metric((aggregate_metrics.get('ttft_sec') or {}).get('median'), suffix='s')} "
        f"decode_tps_median={_format_metric((aggregate_metrics.get('generation_tps') or {}).get('median'))}"
    )

    if aggregate["successful_runs"] < 1:
        print("no measured sample succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
