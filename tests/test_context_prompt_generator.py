"""Unit tests for the deterministic context prompt generator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "recipes"
    / "llama-cpp"
    / "qwen38-flash-next-ud-iq4-xs"
    / "tools"
    / "generate_context_prompts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("generate_context_prompts", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module()


class DeterministicWordsTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible(self):
        first = generator.deterministic_words(100)
        second = generator.deterministic_words(100)
        self.assertEqual(first, second)

    def test_different_seed_changes_stream(self):
        self.assertNotEqual(
            generator.deterministic_words(100, seed=1),
            generator.deterministic_words(100, seed=2),
        )


class PrefixSearchTests(unittest.TestCase):
    def test_finds_largest_prefix_within_limit(self):
        words = ["one", "two", "three", "four", "five"]

        def count_words(text: str) -> int:
            return len(text.split())

        suffix_words = len(generator.SUFFIX.split())
        text, count = generator.largest_prefix_at_most(
            words,
            token_limit=suffix_words + 3,
            count_tokens=count_words,
        )
        self.assertEqual(count, suffix_words + 3)
        self.assertTrue(text.startswith("one two three"))
        self.assertNotIn("four", text.split()[:4])

    def test_rejects_limit_smaller_than_suffix(self):
        with self.assertRaises(ValueError):
            generator.largest_prefix_at_most(
                ["one"],
                token_limit=1,
                count_tokens=lambda _text: 10,
            )


class TargetParsingTests(unittest.TestCase):
    def test_parses_comma_separated_targets(self):
        self.assertEqual(generator.parse_targets("1024,4096"), [1024, 4096])

    def test_rejects_nonpositive_target(self):
        with self.assertRaises(Exception):
            generator.parse_targets("0")


if __name__ == "__main__":
    unittest.main()
