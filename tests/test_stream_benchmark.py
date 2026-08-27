"""Unit tests for stream_benchmark.py (synthetic SSE and metric math only)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = (
    REPO_ROOT
    / "recipes"
    / "llama-cpp"
    / "qwen38-flash-next-ud-iq4-xs"
    / "tools"
    / "stream_benchmark.py"
)


def load_stream_benchmark_module():
    spec = importlib.util.spec_from_file_location("stream_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sb = load_stream_benchmark_module()


def chunk_event(payload: str) -> str:
    return f"data: {payload}\n\n"


def output_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FeedSseBufferTests(unittest.TestCase):
    def test_split_sse_chunks_reassemble_payloads(self):
        payloads = [
            '{"choices":[{"delta":{"content":"hel"}}]}',
            '{"choices":[{"delta":{"content":"lo"}}]}',
            "[DONE]",
        ]
        raw = "".join(chunk_event(payload) for payload in payloads)
        buffer = ""
        collected: list[str] = []
        midpoint = len(raw) // 2
        for piece in (raw[:midpoint], raw[midpoint:]):
            buffer, found = sb.feed_sse_buffer(buffer, piece)
            collected.extend(found)

        self.assertEqual(collected, payloads)

    def test_done_payload_is_preserved(self):
        _, payloads = sb.feed_sse_buffer("", chunk_event("[DONE]"))
        self.assertEqual(payloads, ["[DONE]"])
        self.assertEqual(sb.parse_sse_payload(payloads[0]), sb.DONE_SENTINEL)


class UsageExtractionTests(unittest.TestCase):
    def test_extract_usage_from_final_chunk(self):
        event = {
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 34,
                "total_tokens": 46,
            },
        }
        usage = sb.extract_usage(event)
        self.assertEqual(
            usage,
            {
                "completion_tokens": 34,
                "prompt_tokens": 12,
                "total_tokens": 46,
            },
        )

    def test_extract_usage_missing_completion_tokens_returns_none(self):
        self.assertIsNone(sb.extract_usage({"usage": {"prompt_tokens": 3}}))


class PromptVariationTests(unittest.TestCase):
    def test_replaces_placeholder_deterministically(self):
        prompt, value = sb.vary_prompt("module @ uses @", "@", 7)
        self.assertEqual(prompt, "module variant_0007 uses variant_0007")
        self.assertEqual(value, "variant_0007")

    def test_no_placeholder_configuration_keeps_prompt(self):
        prompt, value = sb.vary_prompt("unchanged", None, 1)
        self.assertEqual(prompt, "unchanged")
        self.assertIsNone(value)

    def test_missing_placeholder_is_rejected(self):
        with self.assertRaises(ValueError):
            sb.vary_prompt("unchanged", "@", 1)


class RequestMetadataTests(unittest.TestCase):
    def test_build_request_metadata_records_prompt_size_and_hash(self):
        prompt_text = "benchmark prompt\n"
        metadata = sb.build_request_metadata(
            base_url="http://127.0.0.1:8080",
            model="demo",
            max_tokens=128,
            prompt_file=Path("/tmp/prompt.txt"),
            prompt_text=prompt_text,
        )
        prompt_bytes = prompt_text.encode("utf-8")

        self.assertEqual(metadata["prompt_bytes"], len(prompt_bytes))
        self.assertEqual(
            metadata["prompt_sha256"],
            hashlib.sha256(prompt_bytes).hexdigest(),
        )


class OutputFingerprintTests(unittest.TestCase):
    def test_accumulates_content_and_reasoning_deltas_in_order(self):
        events = [
            (0.10, {"choices": [{"delta": {"reasoning_content": "Think"}}]}),
            (0.20, {"choices": [{"delta": {"content": "Answer"}}]}),
            (0.30, {"choices": [{"delta": {"content": "!"}}]}),
        ]
        _, _, output_text = sb.reduce_stream_events(events, request_start=0.0)

        self.assertEqual(output_text, "ThinkAnswer!")

    def test_identical_synthetic_output_hashes_match(self):
        events_a = [
            (0.10, {"choices": [{"delta": {"content": "hel"}}]}),
            (0.20, {"choices": [{"delta": {"content": "lo"}}]}),
            (
                1.00,
                {"choices": [], "usage": {"completion_tokens": 2}},
            ),
        ]
        events_b = [
            (0.50, {"choices": [{"delta": {"content": "hel"}}]}),
            (0.60, {"choices": [{"delta": {"content": "lo"}}]}),
            (
                2.00,
                {"choices": [], "usage": {"completion_tokens": 2}},
            ),
        ]
        marks_a, usage_a, output_a = sb.reduce_stream_events(events_a, 0.0)
        marks_b, usage_b, output_b = sb.reduce_stream_events(events_b, 0.0)
        metrics_a = sb.compute_stream_metrics(marks_a, usage_a, output_a)
        metrics_b = sb.compute_stream_metrics(marks_b, usage_b, output_b)

        self.assertEqual(output_a, "hello")
        self.assertEqual(output_b, "hello")
        self.assertEqual(metrics_a["output_sha256"], metrics_b["output_sha256"])
        self.assertEqual(metrics_a["output_chars"], 5)
        self.assertEqual(metrics_a["output_sha256"], output_sha256("hello"))

    def test_changed_output_differs(self):
        events_same = [
            (0.10, {"choices": [{"delta": {"content": "same"}}]}),
        ]
        events_changed = [
            (0.10, {"choices": [{"delta": {"content": "diff"}}]}),
        ]
        _, _, output_same = sb.reduce_stream_events(events_same, 0.0)
        _, _, output_changed = sb.reduce_stream_events(events_changed, 0.0)
        metrics_same = sb.compute_stream_metrics(
            sb.TimingMarks(0.0, 0.10, 0.20),
            None,
            output_same,
        )
        metrics_changed = sb.compute_stream_metrics(
            sb.TimingMarks(0.0, 0.10, 0.20),
            None,
            output_changed,
        )

        self.assertNotEqual(
            metrics_same["output_sha256"],
            metrics_changed["output_sha256"],
        )
        self.assertEqual(metrics_same["output_chars"], len("same"))
        self.assertEqual(metrics_changed["output_chars"], len("diff"))


class StreamReductionTests(unittest.TestCase):
    def test_empty_content_before_first_token(self):
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        events = [
            (
                0.10,
                {"choices": [{"delta": {"content": ""}}]},
            ),
            (
                0.20,
                {"choices": [{"delta": {"role": "assistant"}}]},
            ),
            (
                0.30,
                {"choices": [{"delta": {"content": "Hi"}}]},
            ),
            (
                1.00,
                {
                    "choices": [],
                    "usage": usage,
                },
            ),
            (1.00, sb.DONE_SENTINEL),
        ]
        marks, usage_snapshot, output_text = sb.reduce_stream_events(
            events, request_start=0.0
        )
        metrics = sb.compute_stream_metrics(marks, usage_snapshot, output_text)

        self.assertEqual(output_text, "Hi")
        self.assertEqual(marks.first_content, 0.30)
        self.assertEqual(usage_snapshot, usage)
        self.assertAlmostEqual(metrics["ttft_sec"], 0.30)
        self.assertAlmostEqual(metrics["total_sec"], 1.00)
        self.assertEqual(metrics["prompt_tokens"], 5)
        self.assertEqual(metrics["total_tokens"], 7)
        self.assertEqual(metrics["post_first_tokens"], 1)
        self.assertAlmostEqual(metrics["generation_tps"], 1 / 0.70)
        self.assertAlmostEqual(metrics["end_to_end_output_tps"], 2 / 1.00)
        self.assertEqual(metrics["output_chars"], 2)
        self.assertEqual(metrics["output_sha256"], output_sha256("Hi"))

    def test_reasoning_delta_counts_as_first_generated_output(self):
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        events = [
            (0.15, {"choices": [{"delta": {"reasoning_content": "Think"}}]}),
            (0.50, {"choices": [], "usage": usage}),
            (0.50, sb.DONE_SENTINEL),
        ]
        marks, usage_snapshot, output_text = sb.reduce_stream_events(
            events, request_start=0.0
        )
        metrics = sb.compute_stream_metrics(marks, usage_snapshot, output_text)

        self.assertEqual(output_text, "Think")
        self.assertEqual(marks.first_content, 0.15)
        self.assertAlmostEqual(metrics["ttft_sec"], 0.15)
        self.assertEqual(metrics["post_first_tokens"], 1)

    def test_timing_math_keeps_ttft_and_generation_rate_distinct(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 80,
            "total_tokens": 90,
        }
        marks = sb.TimingMarks(request_start=0.0, first_content=0.2, end=1.0)
        metrics = sb.compute_stream_metrics(marks, usage, "synthetic output")

        self.assertAlmostEqual(metrics["ttft_sec"], 0.2)
        self.assertAlmostEqual(metrics["total_sec"], 1.0)
        self.assertEqual(metrics["post_first_tokens"], 79)
        self.assertAlmostEqual(metrics["generation_tps"], 79 / 0.8)
        self.assertAlmostEqual(metrics["end_to_end_output_tps"], 80 / 1.0)
        self.assertEqual(metrics["prompt_tokens"], 10)
        self.assertEqual(metrics["total_tokens"], 90)
        self.assertEqual(metrics["output_chars"], len("synthetic output"))

    def test_single_completion_token_has_no_decode_tps(self):
        usage = {"completion_tokens": 1}
        marks = sb.TimingMarks(request_start=0.0, first_content=0.1, end=1.1)
        metrics = sb.compute_stream_metrics(marks, usage, "x")

        self.assertEqual(metrics["post_first_tokens"], 0)
        self.assertIsNone(metrics["generation_tps"])
        self.assertAlmostEqual(metrics["end_to_end_output_tps"], 1 / 1.1)

    def test_generation_tps_not_invented_without_usage(self):
        marks = sb.TimingMarks(request_start=0.0, first_content=0.1, end=1.1)
        metrics = sb.compute_stream_metrics(marks, usage=None, output_text="")

        self.assertIsNone(metrics["output_tokens"])
        self.assertIsNone(metrics["generation_tps"])
        self.assertIsNone(metrics["end_to_end_output_tps"])
        self.assertEqual(metrics["output_chars"], 0)

    def test_generation_tps_not_invented_when_post_first_window_zero(self):
        usage = {"completion_tokens": 5}
        marks = sb.TimingMarks(request_start=1.0, first_content=2.0, end=2.0)
        metrics = sb.compute_stream_metrics(marks, usage, "abcde")

        self.assertEqual(metrics["post_first_tokens"], 4)
        self.assertIsNone(metrics["generation_tps"])


class AggregateTests(unittest.TestCase):
    def _run_record(
        self,
        *,
        warmup: bool,
        success: bool,
        metrics: dict | None,
        run_index: int,
    ) -> dict:
        return sb.build_run_record(
            context_label="ctx-a",
            warmup=warmup,
            run_index=run_index,
            request={"model": "demo"},
            success=success,
            metrics=metrics,
            error=None if success else "failed",
        )

    def test_aggregate_excludes_warmups_and_failed_runs(self):
        records = [
            self._run_record(
                warmup=True,
                success=True,
                run_index=0,
                metrics={
                    "ttft_sec": 9.0,
                    "total_sec": 10.0,
                    "output_tokens": 1,
                    "post_first_tokens": 0,
                    "generation_tps": 1.0,
                    "end_to_end_output_tps": 0.1,
                    "output_chars": 4,
                },
            ),
            self._run_record(
                warmup=False,
                success=False,
                run_index=1,
                metrics=None,
            ),
            self._run_record(
                warmup=False,
                success=True,
                run_index=2,
                metrics={
                    "ttft_sec": 0.2,
                    "total_sec": 1.0,
                    "output_tokens": 40,
                    "post_first_tokens": 39,
                    "generation_tps": 50.0,
                    "end_to_end_output_tps": 40.0,
                    "prompt_tokens": 100,
                    "output_chars": 100,
                },
            ),
            self._run_record(
                warmup=False,
                success=True,
                run_index=3,
                metrics={
                    "ttft_sec": 0.4,
                    "total_sec": 1.2,
                    "output_tokens": 60,
                    "post_first_tokens": 59,
                    "generation_tps": 60.0,
                    "end_to_end_output_tps": 50.0,
                    "prompt_tokens": 120,
                    "output_chars": 120,
                },
            ),
        ]

        aggregate = sb.compute_aggregate(
            records,
            context_label="ctx-a",
            request={"model": "demo"},
        )

        self.assertEqual(aggregate["measured_runs"], 3)
        self.assertEqual(aggregate["successful_runs"], 2)
        self.assertAlmostEqual(aggregate["metrics"]["ttft_sec"]["median"], 0.3)
        self.assertEqual(aggregate["metrics"]["ttft_sec"]["min"], 0.2)
        self.assertEqual(aggregate["metrics"]["ttft_sec"]["max"], 0.4)
        self.assertAlmostEqual(aggregate["metrics"]["generation_tps"]["median"], 55.0)
        self.assertAlmostEqual(
            aggregate["metrics"]["prompt_tokens"]["median"],
            110.0,
        )
        self.assertAlmostEqual(aggregate["metrics"]["output_chars"]["median"], 110.0)


class JsonlAppendTests(unittest.TestCase):
    def test_append_jsonl_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            records = [
                {
                    "record_type": "run",
                    "success": True,
                    "warmup": False,
                    "metrics": {
                        "output_chars": 3,
                        "output_sha256": output_sha256("yes"),
                    },
                },
                {"record_type": "aggregate", "successful_runs": 1},
            ]
            sb.append_jsonl(path, records)
            sb.append_jsonl(path, [{"record_type": "run", "success": False}])

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["record_type"], "run")
            self.assertEqual(parsed[1]["record_type"], "aggregate")
            self.assertFalse(parsed[2]["success"])
            self.assertNotIn("output_text", parsed[0])
            self.assertNotIn("output", parsed[0])


class ParseErrorTests(unittest.TestCase):
    def test_invalid_json_raises_actionable_error(self):
        with self.assertRaises(sb.SseParseError):
            sb.parse_sse_payload("{not-json")


if __name__ == "__main__":
    unittest.main()
