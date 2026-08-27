"""Unit tests for recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/tools/spark_guard.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    REPO_ROOT
    / "recipes"
    / "llama-cpp"
    / "qwen38-flash-next-ud-iq4-xs"
    / "tools"
    / "spark_guard.py"
)


def load_spark_guard_module():
    spec = importlib.util.spec_from_file_location("spark_guard", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spark_guard = load_spark_guard_module()


def meminfo_text(mem_available_kib: int, swap_total_kib: int, swap_free_kib: int) -> str:
    return (
        f"MemAvailable:   {mem_available_kib} kB\n"
        f"SwapTotal:      {swap_total_kib} kB\n"
        f"SwapFree:       {swap_free_kib} kB\n"
    )


def make_thresholds(**overrides):
    defaults = {
        "min_start_mem_bytes": spark_guard.gib_to_bytes(32),
        "soft_stop_mem_bytes": spark_guard.gib_to_bytes(16),
        "hard_kill_mem_bytes": spark_guard.gib_to_bytes(8),
        "max_swap_growth_bytes": spark_guard.gib_to_bytes(4),
        "soft_stop_grace_seconds": 1.0,
        "sample_interval_seconds": 0.1,
    }
    defaults.update(overrides)
    return spark_guard.Thresholds(**defaults)


class ParseMeminfoTests(unittest.TestCase):
    def test_metrics_from_meminfo_kib_converts_to_bytes(self):
        values = spark_guard.parse_meminfo_kib(
            meminfo_text(mem_available_kib=1024, swap_total_kib=2048, swap_free_kib=512)
        )
        metrics = spark_guard.metrics_from_meminfo_kib(values)
        self.assertEqual(metrics.mem_available_bytes, 1024 * 1024)
        self.assertEqual(metrics.swap_used_bytes, (2048 - 512) * 1024)

    def test_parse_meminfo_requires_supported_units(self):
        with self.assertRaises(ValueError):
            spark_guard.parse_meminfo_kib("MemAvailable: 1024 bytes\n")

    def test_parse_meminfo_ignores_unrelated_unitless_fields(self):
        values = spark_guard.parse_meminfo_kib(
            meminfo_text(1024, 2048, 512) + "HugePages_Total: 0\n"
        )
        self.assertEqual(values["MemAvailable"], 1024)

    def test_metrics_from_meminfo_requires_fields(self):
        with self.assertRaises(ValueError):
            spark_guard.metrics_from_meminfo_kib({"SwapTotal": 1, "SwapFree": 0})


class ThresholdValidationTests(unittest.TestCase):
    def test_validate_thresholds_rejects_invalid_ordering(self):
        thresholds = make_thresholds(
            min_start_mem_bytes=spark_guard.gib_to_bytes(8),
            soft_stop_mem_bytes=spark_guard.gib_to_bytes(16),
            hard_kill_mem_bytes=spark_guard.gib_to_bytes(4),
        )
        error = spark_guard.validate_thresholds(thresholds)
        self.assertIsNotNone(error)
        self.assertIn("min_start_mem_gib", error)

    def test_validate_thresholds_accepts_valid_ordering(self):
        thresholds = make_thresholds()
        self.assertIsNone(spark_guard.validate_thresholds(thresholds))


class EvaluateMetricsTests(unittest.TestCase):
    def test_swap_growth_exceeded(self):
        initial = 1024
        max_growth = 2048
        self.assertFalse(
            spark_guard.swap_growth_exceeded(initial + 1024, initial, max_growth)
        )
        self.assertTrue(
            spark_guard.swap_growth_exceeded(initial + 4096, initial, max_growth)
        )

    def test_evaluate_metrics_hard_kill_wins_over_soft(self):
        thresholds = make_thresholds()
        metrics = spark_guard.MemoryMetrics(
            mem_available_bytes=spark_guard.gib_to_bytes(4),
            swap_used_bytes=0,
        )
        action = spark_guard.evaluate_metrics(metrics, thresholds, 0, None, time.time())
        self.assertEqual(action, spark_guard.GuardAction.HARD_KILL)

    def test_evaluate_metrics_soft_stop_before_grace(self):
        thresholds = make_thresholds()
        metrics = spark_guard.MemoryMetrics(
            mem_available_bytes=spark_guard.gib_to_bytes(12),
            swap_used_bytes=0,
        )
        action = spark_guard.evaluate_metrics(metrics, thresholds, 0, None, time.time())
        self.assertEqual(action, spark_guard.GuardAction.SOFT_STOP)

    def test_evaluate_metrics_grace_expired_triggers_hard_kill(self):
        thresholds = make_thresholds(soft_stop_grace_seconds=1.0)
        metrics = spark_guard.MemoryMetrics(
            mem_available_bytes=spark_guard.gib_to_bytes(12),
            swap_used_bytes=0,
        )
        started = time.time() - 2.0
        action = spark_guard.evaluate_metrics(metrics, thresholds, 0, started, time.time())
        self.assertEqual(action, spark_guard.GuardAction.HARD_KILL)


class SparkGuardCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log_path = self.tmp / "events.jsonl"
        self.meminfo_path = self.tmp / "meminfo"

    def tearDown(self):
        self._tmp.cleanup()

    def write_meminfo(self, mem_available_gib: float, swap_used_gib: float = 0.0):
        total_swap_kib = 1024 * 1024
        used_swap_kib = int((swap_used_gib * spark_guard.GIB) / 1024)
        free_swap_kib = total_swap_kib - used_swap_kib
        mem_available_kib = int((mem_available_gib * spark_guard.GIB) / 1024)
        self.meminfo_path.write_text(
            meminfo_text(mem_available_kib, total_swap_kib, free_swap_kib),
            encoding="utf-8",
        )

    def base_args(self) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT_PATH),
            "--min-start-mem-gib",
            "32",
            "--soft-stop-mem-gib",
            "16",
            "--hard-kill-mem-gib",
            "8",
            "--max-swap-growth-gib",
            "4",
            "--soft-stop-grace-seconds",
            "1",
            "--sample-interval-seconds",
            "0.1",
            "--log-path",
            str(self.log_path),
            "--meminfo-path",
            str(self.meminfo_path),
        ]

    def run_guard(self, *command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.base_args(), "--", *command],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def read_events(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def wait_for_event(self, event_name: str, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if event_name in {event["event"] for event in self.read_events()}:
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {event_name!r} in JSONL log")

    def finish_guard_process(self, proc: subprocess.Popen[str]) -> int:
        proc.communicate(timeout=10)
        assert proc.returncode is not None
        return proc.returncode

    def test_preflight_fails_on_invalid_threshold_ordering(self):
        self.write_meminfo(64.0)
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--min-start-mem-gib",
                "8",
                "--soft-stop-mem-gib",
                "16",
                "--hard-kill-mem-gib",
                "4",
                "--max-swap-growth-gib",
                "4",
                "--soft-stop-grace-seconds",
                "1",
                "--sample-interval-seconds",
                "0.1",
                "--log-path",
                str(self.log_path),
                "--meminfo-path",
                str(self.meminfo_path),
                "--",
                sys.executable,
                "-c",
                "print('should-not-run')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, spark_guard.EXIT_PREFLIGHT)
        events = self.read_events()
        self.assertEqual(events[0]["event"], "preflight")
        self.assertFalse(events[0]["passed"])
        self.assertNotIn("start", {event["event"] for event in events})

    def test_preflight_fails_when_start_memory_too_low(self):
        self.write_meminfo(16.0)
        completed = self.run_guard(sys.executable, "-c", "print('blocked')")
        self.assertEqual(completed.returncode, spark_guard.EXIT_PREFLIGHT)
        events = self.read_events()
        self.assertEqual(events[0]["event"], "preflight")
        self.assertFalse(events[0]["passed"])
        self.assertNotIn("start", {event["event"] for event in events})

    def test_normal_child_exit_is_forwarded(self):
        self.write_meminfo(64.0)
        completed = self.run_guard(sys.executable, "-c", "import sys; sys.exit(0)")
        self.assertEqual(completed.returncode, 0)
        events = self.read_events()
        self.assertEqual(events[0]["event"], "preflight")
        self.assertTrue(events[0]["passed"])
        self.assertEqual(events[1]["event"], "start")
        self.assertEqual(events[-1]["event"], "exit")
        self.assertEqual(events[-1]["child_exit_code"], 0)
        self.assertIn("thresholds", events[-1])

    def test_soft_stop_terminates_child_group(self):
        self.write_meminfo(64.0)
        child_code = "import time; time.sleep(30)"
        proc = subprocess.Popen(
            [
                *self.base_args(),
                "--sample-interval-seconds",
                "0.05",
                "--",
                sys.executable,
                "-c",
                child_code,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.wait_for_event("start")
        self.write_meminfo(12.0)
        completed = self.finish_guard_process(proc)
        self.assertEqual(completed, spark_guard.EXIT_SOFT_STOP)
        events = self.read_events()
        event_names = [event["event"] for event in events]
        self.assertIn("soft_stop", event_names)
        self.assertEqual(events[-1]["stop_reason"], "soft_stop")

    def test_hard_kill_on_critical_memory(self):
        self.write_meminfo(64.0)
        proc = subprocess.Popen(
            [
                *self.base_args(),
                "--sample-interval-seconds",
                "0.05",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.wait_for_event("start")
        self.write_meminfo(4.0)
        completed = self.finish_guard_process(proc)
        self.assertEqual(completed, spark_guard.EXIT_HARD_KILL)
        events = self.read_events()
        self.assertIn("hard_kill", {event["event"] for event in events})
        self.assertEqual(events[-1]["stop_reason"], "hard_kill")


if __name__ == "__main__":
    unittest.main()
