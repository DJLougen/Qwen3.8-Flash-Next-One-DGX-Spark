#!/usr/bin/env python3
"""Generate a draft inference recipe from workspace templates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1
RUNTIME_IDS = frozenset({"sglang", "llama-cpp", "vllm"})
RUNTIME_INFO_FIELDS = frozenset({"name", "homepage", "documentation"})
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TEMPLATE_TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
MODEL_FIELD_PATTERN = re.compile(r"^\S+$")

RECIPE_TEMPLATE_FILES = (
    "recipe.json",
    "README.md",
    "run.sh",
    "env.example",
)

JSON_TEMPLATE_FIELDS = frozenset(
    {
        "RECIPE_ID",
        "TITLE",
        "SUMMARY",
        "RUNTIME",
        "MODEL_REPOSITORY",
        "MODEL_REVISION",
    }
)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")

def parse_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > 1024:
        raise ValueError("JSON integer exceeds 1024 digits")
    return int(value)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_strict_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
            parse_int=parse_json_integer,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except RecursionError as exc:
        raise ValueError(f"{label}: invalid JSON: nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError(f"{label}: invalid JSON: {exc}") from exc


def read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: invalid UTF-8 at byte {exc.start}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: cannot read file: {exc}") from exc


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_runtime_ids(root: Path) -> set[str]:
    config_dir = root / "config"
    require_directory(config_dir, "config directory")
    config_path = config_dir / "runtimes.json"
    require_regular_file(config_path, "runtime registry")
    payload = parse_strict_json(
        read_utf8(config_path, "runtime registry"),
        "runtime registry",
    )

    if not isinstance(payload, dict):
        raise ValueError("runtime registry root must be an object")
    if set(payload) != {"schema_version", "runtimes"}:
        raise ValueError(
            "runtime registry fields must be exactly schema_version and runtimes"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"runtime registry schema_version must be {SCHEMA_VERSION}")

    runtimes = payload["runtimes"]
    if not isinstance(runtimes, dict) or set(runtimes) != RUNTIME_IDS:
        expected = ", ".join(sorted(RUNTIME_IDS))
        raise ValueError(f"runtime registry IDs must be exactly: {expected}")

    for runtime_id in sorted(RUNTIME_IDS):
        info = runtimes[runtime_id]
        if not isinstance(info, dict) or set(info) != RUNTIME_INFO_FIELDS:
            raise ValueError(
                f"runtime registry entry {runtime_id!r} must contain exactly "
                "name, homepage, and documentation"
            )
        for field in sorted(RUNTIME_INFO_FIELDS):
            value = info[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"runtime registry {runtime_id!r}.{field} must be non-empty"
                )
        for field in ("homepage", "documentation"):
            if not info[field].startswith(("https://", "http://")):
                raise ValueError(
                    f"runtime registry {runtime_id!r}.{field} must be an HTTP URL"
                )

    return set(runtimes)


def validate_slug(slug: str) -> None:
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            "invalid slug: must match [a-z0-9]+(?:-[a-z0-9]+)* "
            f"(got {slug!r})"
        )


def validate_required_text(
    name: str,
    value: str,
    *,
    no_whitespace: bool = False,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if no_whitespace and not MODEL_FIELD_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must not contain whitespace")


def json_string_fragment(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    return encoded[1:-1]


def substitute_template(
    content: str,
    mapping: Mapping[str, str],
    *,
    json_fields: frozenset[str] = frozenset(),
) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            raise ValueError(f"template contains unknown token: {key}")
        value = mapping[key]
        return json_string_fragment(value) if key in json_fields else value

    return TEMPLATE_TOKEN_PATTERN.sub(replace, content)


def expected_manifest(
    runtime: str,
    slug: str,
    title: str,
    summary: str,
    model_repository: str,
    model_revision: str,
) -> dict[str, Any]:
    return {
        "$schema": "../../../schema/recipe.schema.json",
        "schema_version": SCHEMA_VERSION,
        "id": f"{runtime}/{slug}",
        "title": title,
        "summary": summary,
        "runtime": runtime,
        "model": {
            "repository": model_repository,
            "revision": model_revision,
        },
        "hardware": {
            "target": "NVIDIA DGX Spark",
            "gpu": "GB10",
        },
        "entrypoint": "run.sh",
        "status": "draft",
        "tested_at": None,
    }


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@contextmanager
def repository_lock(root: Path) -> Iterator[None]:
    cache_dir = root / ".cache"
    if cache_dir.is_symlink():
        raise ValueError(f"generator cache must not be a symlink: {cache_dir}")
    cache_dir.mkdir(exist_ok=True)
    if not cache_dir.is_dir():
        raise ValueError(f"generator cache is not a directory: {cache_dir}")

    lock_path = cache_dir / "recipe-generator.lock"
    if lock_path.is_symlink():
        raise ValueError(f"generator lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def generate_recipe(
    root: Path,
    runtime: str,
    slug: str,
    title: str,
    summary: str,
    model_repository: str,
    model_revision: str,
) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root not found: {root}")

    runtime_ids = load_runtime_ids(root)
    if runtime not in runtime_ids:
        known = ", ".join(sorted(runtime_ids))
        raise ValueError(f"unknown runtime {runtime!r}; supported: {known}")

    validate_slug(slug)
    validate_required_text("title", title)
    validate_required_text("summary", summary)
    validate_required_text(
        "model repository",
        model_repository,
        no_whitespace=True,
    )
    validate_required_text(
        "model revision",
        model_revision,
        no_whitespace=True,
    )

    templates_root = root / "templates"
    require_directory(templates_root, "templates directory")
    template_dir = templates_root / "recipe"
    require_directory(template_dir, "recipe templates directory")
    for filename in RECIPE_TEMPLATE_FILES:
        require_regular_file(template_dir / filename, f"recipe template {filename}")

    recipes_root = root / "recipes"
    if recipes_root.is_symlink():
        raise ValueError(f"recipes directory must not be a symlink: {recipes_root}")
    recipes_root.mkdir(exist_ok=True)
    if not recipes_root.is_dir():
        raise ValueError(f"recipes path is not a directory: {recipes_root}")

    runtime_lane = recipes_root / runtime
    if runtime_lane.is_symlink():
        raise ValueError(f"runtime lane must not be a symlink: {runtime_lane}")
    runtime_lane.mkdir(exist_ok=True)
    if not runtime_lane.is_dir():
        raise ValueError(f"runtime lane is not a directory: {runtime_lane}")

    destination = runtime_lane / slug
    mapping: dict[str, str] = {
        "RECIPE_ID": f"{runtime}/{slug}",
        "TITLE": title,
        "SUMMARY": summary,
        "RUNTIME": runtime,
        "MODEL_REPOSITORY": model_repository,
        "MODEL_REVISION": model_revision,
    }
    expected = expected_manifest(
        runtime,
        slug,
        title,
        summary,
        model_repository,
        model_revision,
    )

    with repository_lock(root):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"recipe destination already exists: {destination}"
            )

        staging_parent = Path(
            tempfile.mkdtemp(prefix=".new-recipe-", dir=runtime_lane)
        )
        staging_dir = staging_parent / slug
        staging_dir.mkdir()
        try:
            for filename in RECIPE_TEMPLATE_FILES:
                template_path = template_dir / filename
                require_regular_file(
                    template_path,
                    f"recipe template {filename}",
                )
                raw = read_utf8(template_path, f"recipe template {filename}")
                json_fields = (
                    JSON_TEMPLATE_FIELDS
                    if filename == "recipe.json"
                    else frozenset()
                )
                rendered = substitute_template(
                    raw,
                    mapping,
                    json_fields=json_fields,
                )
                output_path = staging_dir / filename
                output_path.write_text(rendered, encoding="utf-8")
                if filename == "run.sh":
                    make_executable(output_path)

            manifest_path = staging_dir / "recipe.json"
            rendered_manifest = parse_strict_json(
                read_utf8(manifest_path, "rendered recipe manifest"),
                "rendered recipe manifest",
            )
            if rendered_manifest != expected:
                raise ValueError(
                    "rendered recipe manifest does not match the schema v1 draft contract"
                )

            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"recipe destination already exists: {destination}"
                )
            staging_dir.rename(destination)
            return destination
        finally:
            if staging_parent.exists():
                shutil.rmtree(staging_parent, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a draft inference recipe from workspace templates.",
    )
    parser.add_argument("--runtime", help="Supported runtime ID")
    parser.add_argument("--slug", help="Recipe slug")
    parser.add_argument("--title", help="Human-readable recipe title")
    parser.add_argument("--summary", help="Short recipe summary")
    parser.add_argument("--model", help="Hugging Face model repository")
    parser.add_argument(
        "--revision",
        help="Model revision (immutable for verified recipes)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to parent of scripts/)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for any missing recipe fields",
    )
    return parser


def collect_recipe_values(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    prompts = {
        "runtime": "Runtime (sglang, llama-cpp, or vllm): ",
        "slug": "Recipe slug: ",
        "title": "Title: ",
        "summary": "Summary: ",
        "model": "Model repository: ",
        "revision": "Model revision: ",
    }
    values: dict[str, str] = {}
    missing: list[str] = []
    for field, prompt in prompts.items():
        value = getattr(args, field)
        if value is None and args.interactive:
            try:
                value = input(prompt)
            except EOFError as exc:
                raise ValueError(
                    f"interactive input ended before {field} was provided"
                ) from exc
        if value is None:
            missing.append(f"--{field}")
        else:
            values[field] = value

    if missing:
        parser.error(
            "the following arguments are required unless --interactive is used: "
            + ", ".join(missing)
        )
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        values = collect_recipe_values(args, parser)
        root = args.root if args.root is not None else default_repo_root()
        root = root.resolve()
        destination = generate_recipe(
            root=root,
            runtime=values["runtime"],
            slug=values["slug"],
            title=values["title"],
            summary=values["summary"],
            model_repository=values["model"],
            model_revision=values["revision"],
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
