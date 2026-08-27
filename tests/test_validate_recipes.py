"""Unit tests for scripts/validate_recipes.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_recipes.py"
SCHEMA = REPO_ROOT / "schema" / "recipe.schema.json"


def write_runtimes(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (root / "schema").mkdir(parents=True, exist_ok=True)
    (root / "schema" / "recipe.schema.json").write_text(
        SCHEMA.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    runtimes = {
        "schema_version": 1,
        "runtimes": {
            "sglang": {
                "name": "SGLang",
                "homepage": "https://github.com/sgl-project/sglang",
                "documentation": "https://docs.sglang.io/",
            },
            "llama-cpp": {
                "name": "llama.cpp",
                "homepage": "https://github.com/ggml-org/llama.cpp",
                "documentation": "https://github.com/ggml-org/llama.cpp/tree/master/docs",
            },
            "vllm": {
                "name": "vLLM",
                "homepage": "https://github.com/vllm-project/vllm",
                "documentation": "https://docs.vllm.ai/",
            },
        },
    }
    (config_dir / "runtimes.json").write_text(
        json.dumps(runtimes, indent=2) + "\n",
        encoding="utf-8",
    )

    recipes_dir = root / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    (recipes_dir / "README.md").write_text("# Recipes\n", encoding="utf-8")
    for runtime_id in runtimes["runtimes"]:
        lane = recipes_dir / runtime_id
        lane.mkdir()
        (lane / "README.md").write_text(
            f"# {runtime_id}\n",
            encoding="utf-8",
        )


def base_manifest(runtime: str, slug: str, **overrides) -> dict:
    manifest = {
        "$schema": "../../../schema/recipe.schema.json",
        "schema_version": 1,
        "id": f"{runtime}/{slug}",
        "title": "Example Recipe",
        "summary": "Draft inference recipe for validation tests.",
        "runtime": runtime,
        "model": {
            "repository": "org/model",
            "revision": "main",
        },
        "hardware": {
            "target": "NVIDIA DGX Spark",
            "gpu": "GB10",
        },
        "entrypoint": "run.sh",
        "status": "draft",
        "tested_at": None,
    }
    manifest.update(overrides)
    return manifest


def write_recipe(
    root: Path,
    runtime: str,
    slug: str,
    manifest: dict,
    run_sh: str = "#!/usr/bin/env bash\nset -euo pipefail\necho draft\n",
    executable: bool = True,
    include_files: bool = True,
) -> Path:
    recipe_dir = root / "recipes" / runtime / slug
    recipe_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = recipe_dir / "recipe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if include_files:
        (recipe_dir / "README.md").write_text("# Example\n", encoding="utf-8")
        (recipe_dir / "env.example").write_text("MODEL_PATH=\n", encoding="utf-8")
        run_path = recipe_dir / "run.sh"
        run_path.write_text(run_sh, encoding="utf-8")
        if executable:
            run_path.chmod(run_path.stat().st_mode | 0o111)

    return manifest_path


def run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


class ValidateRecipesTests(unittest.TestCase):
    def test_valid_recipe_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            write_recipe(root, "sglang", "example", base_manifest("sglang", "example"))

            result = run_validator(root)
            self.assertEqual(result.returncode, 0)
            self.assertIn("validated 1 recipe(s) successfully", result.stdout)

    def test_empty_catalog_with_runtime_lanes_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            for runtime in ("sglang", "llama-cpp", "vllm"):
                (root / "recipes" / runtime).mkdir(parents=True, exist_ok=True)

            result = run_validator(root)
            self.assertEqual(result.returncode, 0)
            self.assertIn("validated 0 recipe(s) successfully", result.stdout)

    def test_malformed_schema_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            (root / "schema" / "recipe.schema.json").write_text(
                "{not-json}\n",
                encoding="utf-8",
            )

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema/recipe.schema.json: invalid JSON", result.stderr)

    def test_schema_contract_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            schema_path = root / "schema" / "recipe.schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["properties"]["runtime"]["enum"] = ["sglang"]
            schema_path.write_text(
                json.dumps(schema),
                encoding="utf-8",
            )

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("properties.runtime.enum values must be", result.stderr)

    def test_manifest_path_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest("sglang", "example")
            manifest["id"] = "sglang/wrong-slug"
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match expected \'sglang/example\'", result.stderr)

    def test_malformed_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest("sglang", "example")
            manifest["unexpected"] = True
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown field \'unexpected\'", result.stderr)

    def test_whitespace_only_human_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest("sglang", "example")
            manifest["title"] = "   "
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("title must be a non-empty string", result.stderr)

    def test_null_required_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = {
                key: None
                for key in base_manifest("sglang", "example")
            }
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("$schema must be", result.stderr)
            self.assertIn("model must be an object", result.stderr)
            self.assertIn("entrypoint must be 'run.sh'", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_recipe_directory_without_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            recipe_dir = root / "recipes" / "sglang" / "partial"
            recipe_dir.mkdir()
            (recipe_dir / "README.md").write_text("# Partial\n", encoding="utf-8")
            (recipe_dir / "env.example").write_text("", encoding="utf-8")
            run_path = recipe_dir / "run.sh"
            run_path.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            run_path.chmod(run_path.stat().st_mode | 0o111)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required file recipe.json", result.stderr)

    def test_json_decoder_failures_are_reported(self) -> None:
        invalid_payloads = (
            b'{"title":"\xff"}',
            b'{"value": NaN}',
            b'{"value": 1, "value": 2}',
            b'{"schema_version":' + (b"9" * 5000) + b"}",
            (b"[" * 2000) + b"0" + (b"]" * 2000),
        )
        for payload in invalid_payloads:
            with self.subTest(payload_prefix=payload[:20]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    write_runtimes(root)
                    manifest_path = write_recipe(
                        root,
                        "sglang",
                        "invalid-json",
                        base_manifest("sglang", "invalid-json"),
                    )
                    manifest_path.write_bytes(payload)

                    result = run_validator(root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("recipe.json: invalid", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_symlinked_recipes_container_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            recipes_dir = root / "recipes"
            external = root / "external-recipes"
            recipes_dir.rename(external)
            recipes_dir.symlink_to(external, target_is_directory=True)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("recipes directory must not be a symlink", result.stderr)

    def test_symlinked_runtime_lane_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            runtime_lane = root / "recipes" / "vllm"
            (runtime_lane / "README.md").unlink()
            runtime_lane.rmdir()
            external = root / "external-lane"
            external.mkdir()
            runtime_lane.symlink_to(external, target_is_directory=True)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("runtime lane must not be a symlink", result.stderr)

    def test_symlinked_recipe_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            external = root / "external-recipe"
            external.mkdir()
            recipe_link = root / "recipes" / "sglang" / "linked"
            recipe_link.symlink_to(external, target_is_directory=True)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("recipe directory must not be a symlink", result.stderr)

    def test_unsupported_runtime_lane_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            unsupported_dir = root / "recipes" / "unknown-runtime" / "example"
            unsupported_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = unsupported_dir / "recipe.json"
            manifest_path.write_text(
                json.dumps(base_manifest("sglang", "example"), indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported runtime lane \'unknown-runtime\'", result.stderr)

    def test_missing_entrypoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            write_recipe(
                root,
                "sglang",
                "example",
                base_manifest("sglang", "example"),
                include_files=False,
            )
            recipe_dir = root / "recipes" / "sglang" / "example"
            (recipe_dir / "README.md").write_text("# Example\n", encoding="utf-8")
            (recipe_dir / "env.example").write_text("MODEL_PATH=\n", encoding="utf-8")

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing entrypoint file \'run.sh\'", result.stderr)

    def test_non_executable_entrypoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            write_recipe(
                root,
                "sglang",
                "example",
                base_manifest("sglang", "example"),
                executable=False,
            )

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("entrypoint is not executable", result.stderr)

    def test_shell_syntax_failure_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            write_recipe(
                root,
                "sglang",
                "example",
                base_manifest("sglang", "example"),
                run_sh="#!/usr/bin/env bash\nif then\n",
            )

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shell syntax check failed", result.stderr)

    def test_malformed_scalar_types_fail_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest(
                "sglang",
                "example",
                schema_version=True,
                status=[],
                entrypoint=["run.sh"],
            )
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema_version must be integer 1", result.stderr)
            self.assertIn("invalid status value []", result.stderr)
            self.assertIn("entrypoint must be 'run.sh'", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_absolute_entrypoint_fails_without_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest(
                "sglang",
                "example",
                entrypoint="/tmp/outside.sh",
            )
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("entrypoint must not be an absolute path", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_invalid_calendar_date_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest(
                "sglang",
                "example",
                status="verified",
                tested_at="2026-99-99",
            )
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "tested_at must be null or ISO date YYYY-MM-DD",
                result.stderr,
            )

    def test_draft_with_date_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest(
                "sglang",
                "example",
                status="draft",
                tested_at="2026-08-26",
            )
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("status 'draft' requires null tested_at", result.stderr)

    def test_symlink_entrypoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            write_recipe(root, "sglang", "example", base_manifest("sglang", "example"))
            recipe_dir = root / "recipes" / "sglang" / "example"
            run_path = recipe_dir / "run.sh"
            run_path.unlink()
            external = root / "external.sh"
            external.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            external.chmod(external.stat().st_mode | 0o111)
            run_path.symlink_to(external)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("entrypoint must not be a symlink", result.stderr)

    def test_verified_without_date_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_runtimes(root)
            manifest = base_manifest("sglang", "example", status="verified", tested_at=None)
            write_recipe(root, "sglang", "example", manifest)

            result = run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("status \'verified\' requires non-null tested_at", result.stderr)


if __name__ == "__main__":
    unittest.main()
