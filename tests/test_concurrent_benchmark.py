"""Unit tests for the two-request concurrency benchmark harness."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = (
    REPO_ROOT
    / "recipes"
    / "llama-cpp"
    / "qwen38-flash-next-ud-iq4-xs"
    / "tools"
)
SCRIPT_PATH = TOOLS_DIR / "concurrent_benchmark.py"


def load_module():
    sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location("concurrent_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_module()


class PromptSpecTests(unittest.TestCase):
    def test_parses_request_id_and_path(self):
        request_id, path = benchmark.parse_prompt_spec("alpha=/tmp/a.txt")
        self.assertEqual(request_id, "alpha")
        self.assertEqual(path, Path("/tmp/a.txt"))

    def test_rejects_missing_request_id(self):
        with self.assertRaises(Exception):
            benchmark.parse_prompt_spec("/tmp/a.txt")


class ConcurrentBatchTests(unittest.TestCase):
    def test_two_workers_start_together_and_aggregate_output(self):
        starts: list[float] = []
        lock = threading.Lock()

        def fake_runner(**_kwargs):
            with lock:
                starts.append(time.perf_counter())
            time.sleep(0.02)
            return (
                {
                    "ttft_sec": 0.01,
                    "generation_tps": 20.0,
                    "output_tokens": 64,
                    "output_sha256": "abc",
                },
                None,
            )

        results, batch = benchmark.run_concurrent_batch(
            [
                ("alpha", Path("/tmp/a.txt"), "alpha prompt"),
                ("beta", Path("/tmp/b.txt"), "beta prompt"),
            ],
            base_url="http://127.0.0.1:8080",
            model="demo",
            max_tokens=64,
            timeout=1.0,
            runner=fake_runner,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["request"]["concurrency"] == 2 for result in results))
        self.assertEqual(batch["successful_requests"], 2)
        self.assertEqual(batch["total_output_tokens"], 128)
        self.assertLess(abs(starts[0] - starts[1]), 0.02)
        self.assertLess(batch["start_skew_sec"], 0.02)
        self.assertGreater(batch["aggregate_output_tps"], 0)

    def test_requires_exactly_two_prompts(self):
        with self.assertRaises(ValueError):
            benchmark.run_concurrent_batch(
                [("alpha", Path("/tmp/a"), "prompt")],
                base_url="http://127.0.0.1:8080",
                model="demo",
                max_tokens=64,
                timeout=1.0,
            )


class AggregateTests(unittest.TestCase):
    def test_summarizes_each_request_and_batch(self):
        records = [
            {
                "record_type": "concurrent_request",
                "warmup": False,
                "request_id": "alpha",
                "success": True,
                "metrics": {
                    "ttft_sec": 0.2,
                    "generation_tps": 10.0,
                    "output_sha256": "aaa",
                },
                "error": None,
            },
            {
                "record_type": "concurrent_request",
                "warmup": False,
                "request_id": "beta",
                "success": True,
                "metrics": {
                    "ttft_sec": 0.3,
                    "generation_tps": 9.0,
                    "output_sha256": "bbb",
                },
                "error": None,
            },
            {
                "record_type": "concurrent_batch",
                "warmup": False,
                "successful_requests": 2,
                "batch_wall_sec": 7.0,
                "aggregate_output_tps": 18.0,
                "start_skew_sec": 0.001,
            },
        ]

        aggregate = benchmark.aggregate_records(records)
        self.assertEqual(aggregate["successful_batches"], 1)
        self.assertEqual(
            aggregate["per_request"]["alpha"]["decode_median_tps"],
            10.0,
        )
        self.assertEqual(aggregate["aggregate_output_median_tps"], 18.0)


if __name__ == "__main__":
    unittest.main()
