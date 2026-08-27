"""Unit tests for scripts/new_recipe.py."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "new_recipe.py"

TEMPLATE_TOKEN_PATTERN = re.compile(r"\{\{[A-Z0-9_]+\}\}")
EXPECTED_FILES = ("recipe.json", "README.md", "run.sh", "env.example")


def load_new_recipe_module():
    spec = importlib.util.spec_from_file_location("new_recipe", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


new_recipe = load_new_recipe_module()


def make_workspace(parent: Path) -> Path:
    workspace = parent / "workspace"
    shutil.copytree(REPO_ROOT / "config", workspace / "config")
    shutil.copytree(REPO_ROOT / "templates", workspace / "templates")
    (workspace / "recipes").mkdir()
    return workspace


class NewRecipeGeneratorTests(unittest.TestCase):
    def test_successful_generation_substitutes_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            destination = new_recipe.generate_recipe(
                root=workspace,
                runtime="vllm",
                slug="qwen-demo",
                title="Qwen Demo \"quoted\" title",
                summary="Serve with tuning: backslash \\ and newline\nsupport",
                model_repository="org/model",
                model_revision="main",
            )

            self.assertEqual(
                destination,
                workspace.resolve() / "recipes" / "vllm" / "qwen-demo",
            )
            for filename in EXPECTED_FILES:
                self.assertTrue((destination / filename).is_file(), filename)

            manifest_text = (destination / "recipe.json").read_text(encoding="utf-8")
            self.assertIsNone(TEMPLATE_TOKEN_PATTERN.search(manifest_text))
            manifest = json.loads(manifest_text)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["$schema"], "../../../schema/recipe.schema.json")
            self.assertEqual(manifest["id"], "vllm/qwen-demo")
            self.assertEqual(manifest["title"], "Qwen Demo \"quoted\" title")
            self.assertEqual(
                manifest["summary"],
                "Serve with tuning: backslash \\ and newline\nsupport",
            )
            self.assertEqual(manifest["runtime"], "vllm")
            self.assertEqual(manifest["model"]["repository"], "org/model")
            self.assertEqual(manifest["model"]["revision"], "main")
            self.assertEqual(manifest["hardware"]["target"], "NVIDIA DGX Spark")
            self.assertEqual(manifest["hardware"]["gpu"], "GB10")
            self.assertEqual(manifest["entrypoint"], "run.sh")
            self.assertEqual(manifest["status"], "draft")
            self.assertIsNone(manifest["tested_at"])

            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("Draft recipe template", readme)
            self.assertNotRegex(readme, TEMPLATE_TOKEN_PATTERN)

            env_example = (destination / "env.example").read_text(encoding="utf-8")
            self.assertIn("vllm/qwen-demo", env_example)
            self.assertNotRegex(env_example, TEMPLATE_TOKEN_PATTERN)

    def test_template_like_human_text_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            destination = new_recipe.generate_recipe(
                root=workspace,
                runtime="vllm",
                slug="literal-braces",
                title="Literal {{SUMMARY}} title",
                summary="Keep {{TITLE}} exactly as authored.",
                model_repository="org/model",
                model_revision="revision",
            )

            manifest = json.loads(
                (destination / "recipe.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["title"], "Literal {{SUMMARY}} title")
            self.assertEqual(
                manifest["summary"],
                "Keep {{TITLE}} exactly as authored.",
            )
            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("Literal {{SUMMARY}} title", readme)
            self.assertIn("Keep {{TITLE}} exactly as authored.", readme)

    def test_blank_required_values_are_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            base = {
                "root": workspace,
                "runtime": "vllm",
                "slug": "blank-input",
                "title": "Title",
                "summary": "Summary",
                "model_repository": "org/model",
                "model_revision": "revision",
            }

            for field in ("title", "summary", "model_repository", "model_revision"):
                with self.subTest(field=field):
                    values = dict(base)
                    values[field] = "   "
                    with self.assertRaises(ValueError):
                        new_recipe.generate_recipe(**values)
                    self.assertFalse(
                        (workspace / "recipes" / "vllm" / "blank-input").exists()
                    )

    def test_runtime_registry_must_match_supported_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            registry_path = workspace / "config" / "runtimes.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["runtimes"]["tgi"] = {
                "name": "TGI",
                "homepage": "https://example.invalid/tgi",
                "documentation": "https://example.invalid/tgi/docs",
            }
            registry_path.write_text(
                json.dumps(registry),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "IDs must be exactly"):
                new_recipe.generate_recipe(
                    root=workspace,
                    runtime="tgi",
                    slug="extra-runtime",
                    title="Extra runtime",
                    summary="Must be rejected",
                    model_repository="org/model",
                    model_revision="revision",
                )
            self.assertFalse((workspace / "recipes" / "tgi").exists())

    def test_invalid_rendered_manifests_are_never_published(self):
        invalid_templates = (
            "{}\n",
            '{"value": NaN}\n',
            '{"value": 1, "value": 2}\n',
        )
        for template in invalid_templates:
            with self.subTest(template=template):
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = make_workspace(Path(tmp))
                    (workspace / "templates" / "recipe" / "recipe.json").write_text(
                        template,
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        new_recipe.generate_recipe(
                            root=workspace,
                            runtime="vllm",
                            slug="invalid-template",
                            title="Invalid template",
                            summary="Must not be published",
                            model_repository="org/model",
                            model_revision="revision",
                        )
                    lane = workspace / "recipes" / "vllm"
                    self.assertFalse((lane / "invalid-template").exists())
                    self.assertFalse(
                        any(
                            path.name.startswith(".new-recipe-")
                            for path in lane.iterdir()
                        )
                    )

    def test_symlinked_recipes_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            recipes_dir = workspace / "recipes"
            recipes_dir.rmdir()
            external = Path(tmp) / "external-recipes"
            external.mkdir()
            recipes_dir.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                new_recipe.generate_recipe(
                    root=workspace,
                    runtime="vllm",
                    slug="escaped",
                    title="Escaped",
                    summary="Must remain inside the repository",
                    model_repository="org/model",
                    model_revision="revision",
                )
            self.assertEqual(list(external.iterdir()), [])

    def test_run_sh_is_executable_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            destination = new_recipe.generate_recipe(
                root=workspace,
                runtime="sglang",
                slug="draft-stub",
                title="Draft Stub",
                summary="Draft only",
                model_repository="org/model",
                model_revision="rev",
            )

            run_sh = destination / "run.sh"
            mode = run_sh.stat().st_mode
            self.assertTrue(mode & stat.S_IXUSR)
            self.assertTrue(mode & stat.S_IXGRP)
            self.assertTrue(mode & stat.S_IXOTH)

            proc = subprocess.run(
                [str(run_sh)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("draft template", proc.stderr)
            self.assertIn("sglang/draft-stub", proc.stderr)

    def test_unknown_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                new_recipe.generate_recipe(
                    root=workspace,
                    runtime="unknown-runtime",
                    slug="demo",
                    title="Demo",
                    summary="Demo",
                    model_repository="org/model",
                    model_revision="main",
                )
            self.assertIn("unknown runtime", str(ctx.exception))
            self.assertFalse((workspace / "recipes" / "unknown-runtime").exists())

    def test_invalid_slug_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                new_recipe.generate_recipe(
                    root=workspace,
                    runtime="vllm",
                    slug="Bad_Slug",
                    title="Demo",
                    summary="Demo",
                    model_repository="org/model",
                    model_revision="main",
                )
            self.assertIn("invalid slug", str(ctx.exception))
            self.assertFalse((workspace / "recipes" / "vllm" / "Bad_Slug").exists())

    def test_existing_destination_refused_without_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            args = {
                "root": workspace,
                "runtime": "llama-cpp",
                "slug": "already-there",
                "title": "First",
                "summary": "First recipe",
                "model_repository": "org/model",
                "model_revision": "main",
            }
            first = new_recipe.generate_recipe(**args)
            self.assertTrue(first.is_dir())

            second_args = {
                **args,
                "title": "Second",
                "summary": "Should not land",
            }
            with self.assertRaises(FileExistsError):
                new_recipe.generate_recipe(**second_args)

            self.assertTrue(first.is_dir())
            manifest = json.loads((first / "recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["title"], "First")

            partial_dirs = [
                p
                for p in (workspace / "recipes" / "llama-cpp").iterdir()
                if p.name.startswith(".new-recipe-")
            ]
            self.assertEqual(partial_dirs, [])

    def test_make_new_preserves_hostile_human_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            shell_marker = Path(tmp) / "shell-marker"
            make_marker = Path(tmp) / "make-marker"
            title = (
                f"Literal `touch {shell_marker}` "
                f"$(shell touch {make_marker}) with \"quotes\""
            )

            scripts_dir = workspace / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(SCRIPT_PATH, scripts_dir / "new_recipe.py")
            shutil.copy2(REPO_ROOT / "Makefile", workspace / "Makefile")
            interactive_input = "\n".join(
                (
                    "vllm",
                    "hostile-title",
                    title,
                    "Preserve caller text",
                    "org/model",
                    "revision",
                )
            ) + "\n"

            result = subprocess.run(
                ["make", "new"],
                cwd=workspace,
                input=interactive_input,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(shell_marker.exists())
            self.assertFalse(make_marker.exists())
            manifest = json.loads(
                (
                    workspace
                    / "recipes"
                    / "vllm"
                    / "hostile-title"
                    / "recipe.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["title"], title)


    def test_cli_main_returns_nonzero_on_invalid_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_workspace(Path(tmp))
            rc = new_recipe.main(
                [
                    "--runtime",
                    "vllm",
                    "--slug",
                    "INVALID",
                    "--title",
                    "Demo",
                    "--summary",
                    "Demo",
                    "--model",
                    "org/model",
                    "--revision",
                    "main",
                    "--root",
                    str(workspace),
                ]
            )
            self.assertEqual(rc, 1)
            self.assertFalse((workspace / "recipes" / "vllm" / "INVALID").exists())


if __name__ == "__main__":
    unittest.main()
