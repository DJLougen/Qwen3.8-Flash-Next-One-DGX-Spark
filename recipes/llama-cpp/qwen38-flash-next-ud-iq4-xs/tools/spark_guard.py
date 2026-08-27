#!/usr/bin/env python3
"""Linux process guard for DGX Spark benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

GIB = 1024**3
DEFAULT_MEMINFO = Path("/proc/meminfo")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_PREFLIGHT = 2
EXIT_SOFT_STOP = 3
EXIT_HARD_KILL = 4
EXIT_INTERRUPT = 5
EXIT_LAUNCH_FAILED = 6


class GuardAction(str, Enum):
    CONTINUE = "continue"
    SOFT_STOP = "soft_stop"
    HARD_KILL = "hard_kill"


@dataclass(frozen=True)
class Thresholds:
    min_start_mem_bytes: int
    soft_stop_mem_bytes: int
    hard_kill_mem_bytes: int
    max_swap_growth_bytes: int
    soft_stop_grace_seconds: float
    sample_interval_seconds: float

    def as_event_dict(self) -> dict[str, float | int]:
        return {
            "min_start_mem_bytes": self.min_start_mem_bytes,
            "soft_stop_mem_bytes": self.soft_stop_mem_bytes,
            "hard_kill_mem_bytes": self.hard_kill_mem_bytes,
            "max_swap_growth_bytes": self.max_swap_growth_bytes,
            "soft_stop_grace_seconds": self.soft_stop_grace_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
        }


@dataclass(frozen=True)
class MemoryMetrics:
    mem_available_bytes: int
    swap_used_bytes: int


def gib_to_bytes(gib: float) -> int:
    return int(gib * GIB)


def parse_meminfo_kib(text: str) -> dict[str, int]:
    """Parse /proc/meminfo-style content into kB values keyed by field name."""
    required_fields = {"MemAvailable", "SwapTotal", "SwapFree"}
    values: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key not in required_fields:
            continue
        parts = rest.strip().split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ValueError(f"unsupported meminfo value in line: {raw_line!r}")
        values[key] = int(parts[0])
    return values


def metrics_from_meminfo_kib(values: dict[str, int]) -> MemoryMetrics:
    if "MemAvailable" not in values:
        raise ValueError("MemAvailable missing from meminfo")
    if "SwapTotal" not in values or "SwapFree" not in values:
        raise ValueError("SwapTotal or SwapFree missing from meminfo")

    mem_available_bytes = values["MemAvailable"] * 1024
    swap_used_kib = values["SwapTotal"] - values["SwapFree"]
    if swap_used_kib < 0:
        raise ValueError("swap used bytes would be negative")
    return MemoryMetrics(
        mem_available_bytes=mem_available_bytes,
        swap_used_bytes=swap_used_kib * 1024,
    )


def read_memory_metrics(meminfo_path: Path) -> MemoryMetrics:
    text = meminfo_path.read_text(encoding="utf-8")
    return metrics_from_meminfo_kib(parse_meminfo_kib(text))


def validate_thresholds(thresholds: Thresholds) -> str | None:
    if thresholds.min_start_mem_bytes <= 0:
        return "min_start_mem_gib must be positive"
    if thresholds.soft_stop_mem_bytes <= 0:
        return "soft_stop_mem_gib must be positive"
    if thresholds.hard_kill_mem_bytes <= 0:
        return "hard_kill_mem_gib must be positive"
    if thresholds.max_swap_growth_bytes < 0:
        return "max_swap_growth_gib must be non-negative"
    if thresholds.soft_stop_grace_seconds < 0:
        return "soft_stop_grace_seconds must be non-negative"
    if thresholds.sample_interval_seconds <= 0:
        return "sample_interval_seconds must be positive"
    if thresholds.min_start_mem_bytes <= thresholds.soft_stop_mem_bytes:
        return "min_start_mem_gib must be greater than soft_stop_mem_gib"
    if thresholds.soft_stop_mem_bytes <= thresholds.hard_kill_mem_bytes:
        return "soft_stop_mem_gib must be greater than hard_kill_mem_gib"
    return None


def swap_growth_exceeded(
    current_swap_used_bytes: int,
    initial_swap_used_bytes: int,
    max_swap_growth_bytes: int,
) -> bool:
    growth = current_swap_used_bytes - initial_swap_used_bytes
    return growth > max_swap_growth_bytes


def evaluate_metrics(
    metrics: MemoryMetrics,
    thresholds: Thresholds,
    initial_swap_used_bytes: int,
    soft_stop_started_at: float | None,
    now: float,
) -> GuardAction:
    if metrics.mem_available_bytes < thresholds.hard_kill_mem_bytes:
        return GuardAction.HARD_KILL

    soft_due_to_mem = metrics.mem_available_bytes < thresholds.soft_stop_mem_bytes
    soft_due_to_swap = swap_growth_exceeded(
        metrics.swap_used_bytes,
        initial_swap_used_bytes,
        thresholds.max_swap_growth_bytes,
    )
    if soft_due_to_mem or soft_due_to_swap:
        if soft_stop_started_at is None:
            return GuardAction.SOFT_STOP
        if now - soft_stop_started_at >= thresholds.soft_stop_grace_seconds:
            return GuardAction.HARD_KILL

    return GuardAction.CONTINUE


class JsonlLogger:
    def __init__(self, path: Path, thresholds: Thresholds) -> None:
        self._path = path
        self._thresholds = thresholds
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def write(
        self,
        event: str,
        *,
        child_pid: int | None = None,
        mem_available_bytes: int | None = None,
        swap_used_bytes: int | None = None,
        **extra: Any,
    ) -> None:
        record: dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
            "child_pid": child_pid,
            "mem_available_bytes": mem_available_bytes,
            "swap_used_bytes": swap_used_bytes,
            "thresholds": self._thresholds.as_event_dict(),
        }
        record.update(extra)
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()


def signal_process_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a command under Linux memory/swap guardrails.",
    )
    parser.add_argument(
        "--min-start-mem-gib",
        type=float,
        required=True,
        help="Minimum MemAvailable required before launch.",
    )
    parser.add_argument(
        "--soft-stop-mem-gib",
        type=float,
        required=True,
        help="MemAvailable threshold that triggers SIGTERM to the child group.",
    )
    parser.add_argument(
        "--hard-kill-mem-gib",
        type=float,
        required=True,
        help="MemAvailable threshold that triggers SIGKILL to the child group.",
    )
    parser.add_argument(
        "--max-swap-growth-gib",
        type=float,
        required=True,
        help="Maximum allowed swap growth before soft stop.",
    )
    parser.add_argument(
        "--soft-stop-grace-seconds",
        type=float,
        required=True,
        help="Grace period after soft stop before SIGKILL.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        required=True,
        help="Interval between /proc/meminfo samples.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Append-only JSONL lifecycle log path.",
    )
    parser.add_argument(
        "--meminfo-path",
        type=Path,
        default=DEFAULT_MEMINFO,
        help="Path to meminfo source (default: /proc/meminfo).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --",
    )
    return parser


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        min_start_mem_bytes=gib_to_bytes(args.min_start_mem_gib),
        soft_stop_mem_bytes=gib_to_bytes(args.soft_stop_mem_gib),
        hard_kill_mem_bytes=gib_to_bytes(args.hard_kill_mem_gib),
        max_swap_growth_bytes=gib_to_bytes(args.max_swap_growth_gib),
        soft_stop_grace_seconds=float(args.soft_stop_grace_seconds),
        sample_interval_seconds=float(args.sample_interval_seconds),
    )


def run_guard(
    thresholds: Thresholds,
    meminfo_path: Path,
    log_path: Path,
    command: list[str],
) -> int:
    if not command:
        print("spark_guard: missing command after --", file=sys.stderr)
        return EXIT_USAGE
    if command[0] == "--":
        command = command[1:]
    if not command:
        print("spark_guard: missing command after --", file=sys.stderr)
        return EXIT_USAGE

    ordering_error = validate_thresholds(thresholds)
    logger = JsonlLogger(log_path, thresholds)

    try:
        try:
            preflight_metrics = read_memory_metrics(meminfo_path)
        except (OSError, ValueError) as exc:
            logger.write(
                "preflight_failed",
                reason="metrics_unavailable",
                error=str(exc),
            )
            print(f"spark_guard: preflight metrics unavailable: {exc}", file=sys.stderr)
            return EXIT_PREFLIGHT

        preflight_ok = ordering_error is None and (
            preflight_metrics.mem_available_bytes >= thresholds.min_start_mem_bytes
        )
        logger.write(
            "preflight",
            child_pid=None,
            mem_available_bytes=preflight_metrics.mem_available_bytes,
            swap_used_bytes=preflight_metrics.swap_used_bytes,
            passed=preflight_ok,
            ordering_error=ordering_error,
        )

        if ordering_error is not None:
            print(f"spark_guard: invalid thresholds: {ordering_error}", file=sys.stderr)
            return EXIT_PREFLIGHT

        if preflight_metrics.mem_available_bytes < thresholds.min_start_mem_bytes:
            print(
                "spark_guard: MemAvailable below minimum start threshold",
                file=sys.stderr,
            )
            return EXIT_PREFLIGHT

        interrupted = False
        child: subprocess.Popen[bytes] | None = None
        pgid: int | None = None

        def handle_interrupt(_signum: int, _frame: object | None) -> None:
            nonlocal interrupted
            interrupted = True
            if pgid is not None:
                signal_process_group(pgid, signal.SIGTERM)

        previous_sigint = signal.signal(signal.SIGINT, handle_interrupt)

        try:
            try:
                child = subprocess.Popen(
                    command,
                    start_new_session=True,
                )
            except OSError as exc:
                logger.write(
                    "launch_failed",
                    reason="launch_failed",
                    error=str(exc),
                    command=command,
                )
                print(f"spark_guard: failed to launch command: {exc}", file=sys.stderr)
                return EXIT_LAUNCH_FAILED

            pgid = child.pid
            initial_swap_used = preflight_metrics.swap_used_bytes
            soft_stop_started_at: float | None = None
            stop_reason: str | None = None

            logger.write(
                "start",
                child_pid=child.pid,
                mem_available_bytes=preflight_metrics.mem_available_bytes,
                swap_used_bytes=preflight_metrics.swap_used_bytes,
                command=command,
            )

            while True:
                if interrupted:
                    if pgid is not None:
                        signal_process_group(pgid, signal.SIGTERM)
                    try:
                        child.wait(timeout=thresholds.soft_stop_grace_seconds)
                    except subprocess.TimeoutExpired:
                        if pgid is not None:
                            signal_process_group(pgid, signal.SIGKILL)
                        child.wait()
                    try:
                        metrics = read_memory_metrics(meminfo_path)
                    except (OSError, ValueError):
                        metrics = preflight_metrics
                    logger.write(
                        "interrupt",
                        child_pid=child.pid,
                        mem_available_bytes=metrics.mem_available_bytes,
                        swap_used_bytes=metrics.swap_used_bytes,
                        stop_reason="signal_interrupt",
                    )
                    return EXIT_INTERRUPT

                try:
                    child.wait(timeout=thresholds.sample_interval_seconds)
                    try:
                        final_metrics = read_memory_metrics(meminfo_path)
                    except (OSError, ValueError):
                        final_metrics = preflight_metrics
                    logger.write(
                        "exit",
                        child_pid=child.pid,
                        mem_available_bytes=final_metrics.mem_available_bytes,
                        swap_used_bytes=final_metrics.swap_used_bytes,
                        child_exit_code=child.returncode,
                        stop_reason=stop_reason,
                    )
                    if stop_reason == "soft_stop":
                        return EXIT_SOFT_STOP
                    if stop_reason == "hard_kill":
                        return EXIT_HARD_KILL
                    return child.returncode if child.returncode is not None else EXIT_OK
                except subprocess.TimeoutExpired:
                    pass

                try:
                    metrics = read_memory_metrics(meminfo_path)
                except (OSError, ValueError) as exc:
                    logger.write(
                        "preflight_failed",
                        child_pid=child.pid,
                        reason="metrics_unavailable",
                        error=str(exc),
                    )
                    if pgid is not None:
                        signal_process_group(pgid, signal.SIGKILL)
                    child.wait()
                    return EXIT_HARD_KILL

                logger.write(
                    "sample",
                    child_pid=child.pid,
                    mem_available_bytes=metrics.mem_available_bytes,
                    swap_used_bytes=metrics.swap_used_bytes,
                )

                now = time.time()
                action = evaluate_metrics(
                    metrics,
                    thresholds,
                    initial_swap_used,
                    soft_stop_started_at,
                    now,
                )

                if action is GuardAction.HARD_KILL:
                    stop_reason = "hard_kill"
                    if pgid is not None:
                        signal_process_group(pgid, signal.SIGKILL)
                    child.wait()
                    logger.write(
                        "hard_kill",
                        child_pid=child.pid,
                        mem_available_bytes=metrics.mem_available_bytes,
                        swap_used_bytes=metrics.swap_used_bytes,
                        stop_reason=stop_reason,
                    )
                    logger.write(
                        "exit",
                        child_pid=child.pid,
                        mem_available_bytes=metrics.mem_available_bytes,
                        swap_used_bytes=metrics.swap_used_bytes,
                        child_exit_code=child.returncode,
                        stop_reason=stop_reason,
                    )
                    return EXIT_HARD_KILL

                if action is GuardAction.SOFT_STOP:
                    stop_reason = "soft_stop"
                    soft_stop_started_at = now
                    if pgid is not None:
                        signal_process_group(pgid, signal.SIGTERM)
                    logger.write(
                        "soft_stop",
                        child_pid=child.pid,
                        mem_available_bytes=metrics.mem_available_bytes,
                        swap_used_bytes=metrics.swap_used_bytes,
                        stop_reason=stop_reason,
                    )
                    continue

                if soft_stop_started_at is not None:
                    if now - soft_stop_started_at >= thresholds.soft_stop_grace_seconds:
                        stop_reason = "hard_kill"
                        if pgid is not None:
                            signal_process_group(pgid, signal.SIGKILL)
                        child.wait()
                        logger.write(
                            "hard_kill",
                            child_pid=child.pid,
                            mem_available_bytes=metrics.mem_available_bytes,
                            swap_used_bytes=metrics.swap_used_bytes,
                            stop_reason="grace_expired",
                        )
                        logger.write(
                            "exit",
                            child_pid=child.pid,
                            mem_available_bytes=metrics.mem_available_bytes,
                            swap_used_bytes=metrics.swap_used_bytes,
                            child_exit_code=child.returncode,
                            stop_reason=stop_reason,
                        )
                        return EXIT_HARD_KILL
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    thresholds = thresholds_from_args(args)
    return run_guard(
        thresholds=thresholds,
        meminfo_path=args.meminfo_path,
        log_path=args.log_path,
        command=args.command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
