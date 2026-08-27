#!/usr/bin/env python3
"""Validate inference recipe manifests and on-disk recipe layouts."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCHEMA_REL_PATH = "schema/recipe.schema.json"
RUNTIMES_REL_PATH = "config/runtimes.json"
RECIPES_DIR = "recipes"

SCHEMA_REF = "../../../schema/recipe.schema.json"
SCHEMA_VERSION = 1
RUNTIME_IDS = frozenset({"sglang", "llama-cpp", "vllm"})
STATUS_VALUES = frozenset({"draft", "verified", "deprecated"})
TOP_LEVEL_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "id",
        "title",
        "summary",
        "runtime",
        "model",
        "hardware",
        "entrypoint",
        "status",
        "tested_at",
    }
)
HARDWARE_TARGET = "NVIDIA DGX Spark"
HARDWARE_GPU = "GB10"
ENTRYPOINT = "run.sh"
ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
ID_RE = re.compile(r"^(sglang|llama-cpp|vllm)/[a-z0-9]+(-[a-z0-9]+)*$")
MODEL_FIELD_RE = re.compile(r"^[^\s]+$")


class ValidationError:
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")

def parse_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > 1024:
        raise ValueError("JSON integer exceeds 1024 digits")
    return int(value)


def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return (
                json.load(
                    handle,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_json_constant,
                    parse_int=parse_json_integer,
                ),
                None,
            )
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}"
    except UnicodeDecodeError as exc:
        return None, f"invalid UTF-8 at byte {exc.start}"
    except RecursionError:
        return None, "invalid JSON: nesting is too deep"
    except ValueError as exc:
        return None, f"invalid JSON value: {exc}"
    except OSError as exc:
        return None, f"cannot read file: {exc}"

def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_valid_iso_date(value: object) -> bool:
    if not isinstance(value, str) or not ISO_DATE_RE.fullmatch(value):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False

def schema_contract_errors(schema: Dict[str, Any]) -> List[str]:
    messages: List[str] = []
    missing = object()

    def nested(path: Tuple[str, ...]) -> Any:
        value: Any = schema
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return missing
            value = value[key]
        return value

    def expect(path: Tuple[str, ...], expected: Any) -> None:
        actual = nested(path)
        if actual != expected:
            actual_display = "<missing>" if actual is missing else repr(actual)
            messages.append(
                f"{'.'.join(path)} must be {expected!r}, got {actual_display}"
            )

    def expect_key_set(path: Tuple[str, ...], expected: Set[str]) -> None:
        actual = nested(path)
        if not isinstance(actual, dict) or set(actual) != expected:
            messages.append(
                f"{'.'.join(path)} keys must be {sorted(expected)!r}"
            )

    def expect_value_set(path: Tuple[str, ...], expected: Set[str]) -> None:
        actual = nested(path)
        try:
            actual_values = set(actual) if isinstance(actual, list) else None
        except TypeError:
            actual_values = None
        if actual_values != expected:
            messages.append(
                f"{'.'.join(path)} values must be {sorted(expected)!r}"
            )

    top_fields = set(TOP_LEVEL_FIELDS)
    expect(("type",), "object")
    expect(("additionalProperties",), False)
    expect_value_set(("required",), top_fields)
    expect_key_set(("properties",), top_fields)

    scalar_checks = (
        (("properties", "$schema", "type"), "string"),
        (("properties", "$schema", "const"), SCHEMA_REF),
        (("properties", "schema_version", "type"), "integer"),
        (("properties", "schema_version", "const"), SCHEMA_VERSION),
        (("properties", "id", "type"), "string"),
        (("properties", "id", "pattern"), ID_RE.pattern),
        (("properties", "title", "type"), "string"),
        (("properties", "title", "minLength"), 1),
        (("properties", "title", "pattern"), r"\S"),
        (("properties", "summary", "type"), "string"),
        (("properties", "summary", "minLength"), 1),
        (("properties", "summary", "pattern"), r"\S"),
        (("properties", "runtime", "type"), "string"),
        (("properties", "model", "type"), "object"),
        (("properties", "model", "additionalProperties"), False),
        (("properties", "hardware", "type"), "object"),
        (("properties", "hardware", "additionalProperties"), False),
        (("properties", "hardware", "properties", "target", "type"), "string"),
        (("properties", "hardware", "properties", "gpu", "type"), "string"),
        (("properties", "hardware", "properties", "target", "const"), HARDWARE_TARGET),
        (("properties", "hardware", "properties", "gpu", "const"), HARDWARE_GPU),
        (("properties", "entrypoint", "type"), "string"),
        (("properties", "entrypoint", "const"), ENTRYPOINT),
        (("properties", "status", "type"), "string"),
    )
    for path, expected in scalar_checks:
        expect(path, expected)

    expect_value_set(
        ("properties", "runtime", "enum"),
        set(RUNTIME_IDS),
    )
    expect_value_set(
        ("properties", "status", "enum"),
        set(STATUS_VALUES),
    )
    expect_value_set(
        ("properties", "model", "required"),
        {"repository", "revision"},
    )
    expect_key_set(
        ("properties", "model", "properties"),
        {"repository", "revision"},
    )
    for field in ("repository", "revision"):
        expect(("properties", "model", "properties", field, "type"), "string")
        expect(("properties", "model", "properties", field, "minLength"), 1)
        expect(
            ("properties", "model", "properties", field, "pattern"),
            MODEL_FIELD_RE.pattern,
        )
    expect_value_set(
        ("properties", "hardware", "required"),
        {"target", "gpu"},
    )
    expect_key_set(
        ("properties", "hardware", "properties"),
        {"target", "gpu"},
    )

    tested_options = nested(("properties", "tested_at", "anyOf"))
    if not isinstance(tested_options, list):
        messages.append("properties.tested_at.anyOf must be an array")
    else:
        option_type_values = [
            option.get("type")
            for option in tested_options
            if isinstance(option, dict)
        ]
        try:
            option_types = set(option_type_values)
        except TypeError:
            option_types = set()
        if option_types != {"null", "string"}:
            messages.append(
                "properties.tested_at.anyOf must allow exactly null and string"
            )
        string_options = [
            option
            for option in tested_options
            if isinstance(option, dict) and option.get("type") == "string"
        ]
        if len(string_options) != 1:
            messages.append(
                "properties.tested_at.anyOf must contain one string rule"
            )
        else:
            if string_options[0].get("format") != "date":
                messages.append("tested_at string rule must use date format")
            if string_options[0].get("pattern") != ISO_DATE_RE.pattern:
                messages.append("tested_at string rule must use YYYY-MM-DD pattern")

    def nested_rule(value: Any, path: Tuple[str, ...]) -> Any:
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    all_of = schema.get("allOf")
    status_rules: Dict[str, Dict[str, Any]] = {}
    if isinstance(all_of, list):
        for rule in all_of:
            status = nested_rule(
                rule,
                ("if", "properties", "status", "const"),
            )
            tested_rule = nested_rule(
                rule,
                ("then", "properties", "tested_at"),
            )
            if isinstance(status, str) and isinstance(tested_rule, dict):
                status_rules[status] = tested_rule
    if set(status_rules) != {"draft", "verified"}:
        messages.append("allOf must define tested_at rules for draft and verified")
    else:
        if status_rules["draft"] != {"type": "null"}:
            messages.append("draft tested_at rule must require null")
        verified_rule = status_rules["verified"]
        if (
            verified_rule.get("type") != "string"
            or verified_rule.get("format") != "date"
            or verified_rule.get("pattern") != ISO_DATE_RE.pattern
        ):
            messages.append(
                "verified tested_at rule must require an ISO YYYY-MM-DD date"
            )

    return messages


def validate_runtimes_config(root: Path) -> Tuple[Set[str], List[ValidationError]]:
    config_dir = root / "config"
    config_rel = rel(root, config_dir)
    if config_dir.is_symlink():
        return set(), [
            ValidationError(config_rel, "config directory must not be a symlink")
        ]
    if not config_dir.is_dir():
        return set(), [ValidationError(config_rel, "missing config directory")]

    path = root / RUNTIMES_REL_PATH
    rel_path = rel(root, path)
    errors: List[ValidationError] = []
    if path.is_symlink():
        return set(), [
            ValidationError(rel_path, "runtime registry must not be a symlink")
        ]
    if not path.is_file():
        return set(), [ValidationError(rel_path, "missing runtimes registry")]

    data, parse_error = load_json(path)
    if parse_error is not None:
        return set(), [ValidationError(rel_path, parse_error)]

    if not isinstance(data, dict):
        return set(), [ValidationError(rel_path, "root must be an object")]

    unknown_top = sorted(set(data.keys()) - {"schema_version", "runtimes"})
    for key in unknown_top:
        errors.append(ValidationError(rel_path, f"unknown top-level field '{key}'"))

    if "schema_version" not in data:
        errors.append(ValidationError(rel_path, "missing required field 'schema_version'"))
    elif type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        errors.append(
            ValidationError(
                rel_path,
                f"schema_version must be integer {SCHEMA_VERSION}, got {data['schema_version']!r}",
            )
        )

    runtimes = data.get("runtimes")
    if "runtimes" not in data:
        errors.append(ValidationError(rel_path, "missing required field 'runtimes'"))
        return set(), errors
    if not isinstance(runtimes, dict):
        errors.append(ValidationError(rel_path, "runtimes must be an object"))
        return set(), errors

    supported: Set[str] = set()
    for runtime_id, runtime_info in sorted(runtimes.items()):
        if not isinstance(runtime_id, str) or not SLUG_RE.fullmatch(runtime_id):
            errors.append(
                ValidationError(rel_path, f"invalid runtime id '{runtime_id}'")
            )
            continue
        if runtime_id not in RUNTIME_IDS:
            errors.append(
                ValidationError(rel_path, f"unsupported runtime id '{runtime_id}'")
            )
        supported.add(runtime_id)

        if not isinstance(runtime_info, dict):
            errors.append(
                ValidationError(rel_path, f"runtimes['{runtime_id}'] must be an object")
            )
            continue

        unknown_runtime_fields = sorted(
            set(runtime_info.keys()) - {"name", "homepage", "documentation"}
        )
        for field in unknown_runtime_fields:
            errors.append(
                ValidationError(
                    rel_path,
                    f"runtimes['{runtime_id}'] has unknown field '{field}'",
                )
            )

        for field in ("name", "homepage", "documentation"):
            if field not in runtime_info:
                errors.append(
                    ValidationError(
                        rel_path,
                        f"runtimes['{runtime_id}'] missing required field '{field}'",
                    )
                )
            elif not isinstance(runtime_info[field], str) or not runtime_info[field]:
                errors.append(
                    ValidationError(
                        rel_path,
                        f"runtimes['{runtime_id}'].{field} must be a non-empty string",
                    )
                )

    if supported != RUNTIME_IDS:
        missing = sorted(RUNTIME_IDS - supported)
        for runtime_id in missing:
            errors.append(
                ValidationError(rel_path, f"missing supported runtime '{runtime_id}'")
            )

    return supported, errors


def manifest_field_errors(
    manifest_path: str,
    manifest: Dict[str, Any],
) -> List[ValidationError]:
    errors: List[ValidationError] = []

    allowed_top = TOP_LEVEL_FIELDS
    for key in sorted(set(manifest.keys()) - allowed_top):
        errors.append(ValidationError(manifest_path, f"unknown field '{key}'"))

    for field in allowed_top:
        if field not in manifest:
            errors.append(ValidationError(manifest_path, f"missing required field '{field}'"))

    schema_value = manifest.get("$schema")
    if "$schema" in manifest and (
        not isinstance(schema_value, str) or schema_value != SCHEMA_REF
    ):
        errors.append(
            ValidationError(
                manifest_path,
                f"$schema must be '{SCHEMA_REF}', got {schema_value!r}",
            )
        )

    schema_version = manifest.get("schema_version")
    if "schema_version" in manifest and (
        type(schema_version) is not int or schema_version != SCHEMA_VERSION
    ):
        errors.append(
            ValidationError(
                manifest_path,
                f"schema_version must be integer {SCHEMA_VERSION}, got {schema_version!r}",
            )
        )

    manifest_id = manifest.get("id")
    if "id" in manifest and (
        not isinstance(manifest_id, str) or not ID_RE.fullmatch(manifest_id)
    ):
        errors.append(
            ValidationError(manifest_path, f"invalid id value {manifest_id!r}")
        )

    for field in ("title", "summary"):
        if field in manifest and not is_nonempty_string(manifest[field]):
            errors.append(
                ValidationError(manifest_path, f"{field} must be a non-empty string")
            )

    runtime = manifest.get("runtime")
    if "runtime" in manifest and (
        not isinstance(runtime, str) or runtime not in RUNTIME_IDS
    ):
        errors.append(
            ValidationError(manifest_path, f"invalid runtime value {runtime!r}")
        )

    model = manifest.get("model")
    if "model" in manifest:
        if not isinstance(model, dict):
            errors.append(ValidationError(manifest_path, "model must be an object"))
        else:
            for key in sorted(set(model.keys()) - {"repository", "revision"}):
                errors.append(
                    ValidationError(manifest_path, f"model has unknown field '{key}'")
                )
            for field in ("repository", "revision"):
                if field not in model:
                    errors.append(
                        ValidationError(
                            manifest_path,
                            f"model missing required field '{field}'",
                        )
                    )
                    continue
                value = model[field]
                if not is_nonempty_string(value):
                    errors.append(
                        ValidationError(
                            manifest_path,
                            f"model.{field} must be a non-empty string",
                        )
                    )
                elif not MODEL_FIELD_RE.fullmatch(value):
                    errors.append(
                        ValidationError(
                            manifest_path,
                            f"model.{field} must not contain whitespace",
                        )
                    )

    hardware = manifest.get("hardware")
    if "hardware" in manifest:
        if not isinstance(hardware, dict):
            errors.append(ValidationError(manifest_path, "hardware must be an object"))
        else:
            for key in sorted(set(hardware.keys()) - {"target", "gpu"}):
                errors.append(
                    ValidationError(manifest_path, f"hardware has unknown field '{key}'")
                )
            target = hardware.get("target")
            if "target" not in hardware:
                errors.append(
                    ValidationError(manifest_path, "hardware missing required field 'target'")
                )
            elif target != HARDWARE_TARGET:
                errors.append(
                    ValidationError(
                        manifest_path,
                        f"hardware.target must be '{HARDWARE_TARGET}', got {target!r}",
                    )
                )
            gpu = hardware.get("gpu")
            if "gpu" not in hardware:
                errors.append(
                    ValidationError(manifest_path, "hardware missing required field 'gpu'")
                )
            elif gpu != HARDWARE_GPU:
                errors.append(
                    ValidationError(
                        manifest_path,
                        f"hardware.gpu must be '{HARDWARE_GPU}', got {gpu!r}",
                    )
                )

    entrypoint = manifest.get("entrypoint")
    if "entrypoint" in manifest and (
        not isinstance(entrypoint, str) or entrypoint != ENTRYPOINT
    ):
        errors.append(
            ValidationError(
                manifest_path,
                f"entrypoint must be '{ENTRYPOINT}', got {entrypoint!r}",
            )
        )

    status = manifest.get("status")
    if "status" in manifest and (
        not isinstance(status, str) or status not in STATUS_VALUES
    ):
        errors.append(
            ValidationError(manifest_path, f"invalid status value {status!r}")
        )

    tested_at = manifest.get("tested_at")
    if tested_at is not None and not is_valid_iso_date(tested_at):
        errors.append(
            ValidationError(
                manifest_path,
                f"tested_at must be null or ISO date YYYY-MM-DD, got {tested_at!r}",
            )
        )

    if status == "draft" and tested_at is not None:
        errors.append(
            ValidationError(
                manifest_path,
                "status 'draft' requires null tested_at",
            )
        )

    if status == "verified" and tested_at is None:
        errors.append(
            ValidationError(
                manifest_path,
                "status 'verified' requires non-null tested_at",
            )
        )

    return errors


def entrypoint_path_errors(entrypoint: str) -> List[str]:
    messages: List[str] = []
    if not isinstance(entrypoint, str) or not entrypoint:
        return ["entrypoint must be a non-empty string"]

    if Path(entrypoint).is_absolute() or entrypoint.startswith("\\"):
        messages.append("entrypoint must not be an absolute path")

    if ".." in Path(entrypoint).parts:
        messages.append("entrypoint must not contain path traversal ('..')")

    if "/" in entrypoint.replace("\\", "/") and entrypoint != ENTRYPOINT:
        messages.append("entrypoint must be a simple filename without directories")

    if entrypoint != ENTRYPOINT:
        messages.append(f"entrypoint must be '{ENTRYPOINT}'")

    return messages


def discover_recipe_manifests(
    root: Path,
    supported_runtimes: Set[str],
) -> Tuple[List[Path], List[ValidationError]]:
    manifests: List[Path] = []
    errors: List[ValidationError] = []
    recipes_root = root / RECIPES_DIR
    recipes_root_rel = rel(root, recipes_root)

    if recipes_root.is_symlink():
        return manifests, [
            ValidationError(recipes_root_rel, "recipes directory must not be a symlink")
        ]
    if not recipes_root.is_dir():
        return manifests, [
            ValidationError(recipes_root_rel, "missing recipes directory")
        ]

    catalog_readme = recipes_root / "README.md"
    if catalog_readme.is_symlink():
        errors.append(
            ValidationError(rel(root, catalog_readme), "catalog README must not be a symlink")
        )
    elif not catalog_readme.is_file():
        errors.append(
            ValidationError(recipes_root_rel, "missing catalog README.md")
        )

    try:
        root_entries = sorted(recipes_root.iterdir())
    except OSError as exc:
        return manifests, [
            ValidationError(recipes_root_rel, f"cannot read recipes directory: {exc}")
        ]

    for runtime_dir in root_entries:
        if runtime_dir.name == "README.md" or runtime_dir.name in supported_runtimes:
            continue
        if runtime_dir.is_symlink() or runtime_dir.is_dir():
            errors.append(
                ValidationError(
                    rel(root, runtime_dir),
                    f"unsupported runtime lane '{runtime_dir.name}'",
                )
            )

    for runtime_id in sorted(supported_runtimes):
        runtime_dir = recipes_root / runtime_id
        runtime_rel = rel(root, runtime_dir)
        if runtime_dir.is_symlink():
            errors.append(
                ValidationError(runtime_rel, "runtime lane must not be a symlink")
            )
            continue
        if not runtime_dir.is_dir():
            errors.append(ValidationError(runtime_rel, "missing runtime lane directory"))
            continue

        lane_readme = runtime_dir / "README.md"
        if lane_readme.is_symlink():
            errors.append(
                ValidationError(rel(root, lane_readme), "lane README must not be a symlink")
            )
        elif not lane_readme.is_file():
            errors.append(ValidationError(runtime_rel, "missing lane README.md"))

        try:
            lane_entries = sorted(runtime_dir.iterdir())
        except OSError as exc:
            errors.append(
                ValidationError(runtime_rel, f"cannot read runtime lane: {exc}")
            )
            continue

        for slug_dir in lane_entries:
            if slug_dir.name == "README.md" or not (
                slug_dir.is_symlink() or slug_dir.is_dir()
            ):
                continue

            slug_rel = rel(root, slug_dir)
            if slug_dir.is_symlink():
                errors.append(
                    ValidationError(slug_rel, "recipe directory must not be a symlink")
                )
                continue
            if not SLUG_RE.fullmatch(slug_dir.name):
                errors.append(
                    ValidationError(slug_rel, f"invalid slug directory '{slug_dir.name}'")
                )

            manifest_path = slug_dir / "recipe.json"
            manifest_rel = rel(root, manifest_path)
            if manifest_path.is_symlink():
                errors.append(
                    ValidationError(manifest_rel, "manifest must not be a symlink")
                )
            elif not manifest_path.is_file():
                errors.append(
                    ValidationError(slug_rel, "missing required file recipe.json")
                )
            else:
                manifests.append(manifest_path)

    return manifests, errors


def validate_recipe_layout(
    root: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
) -> List[ValidationError]:
    errors: List[ValidationError] = []
    recipe_dir = manifest_path.parent
    recipe_dir_rel = rel(root, recipe_dir)
    manifest_rel = rel(root, manifest_path)

    runtime_dir = recipe_dir.parent.name
    slug_dir = recipe_dir.name


    manifest_runtime = manifest.get("runtime")
    if isinstance(manifest_runtime, str) and manifest_runtime != runtime_dir:
        errors.append(
            ValidationError(
                manifest_rel,
                f"runtime '{manifest_runtime}' does not match directory '{runtime_dir}'",
            )
        )

    manifest_id = manifest.get("id")
    expected_id = f"{runtime_dir}/{slug_dir}"
    if isinstance(manifest_id, str) and manifest_id != expected_id:
        errors.append(
            ValidationError(
                manifest_rel,
                f"id '{manifest_id}' does not match expected '{expected_id}'",
            )
        )

    for filename in ("README.md", "env.example"):
        required_path = recipe_dir / filename
        if required_path.is_symlink():
            errors.append(
                ValidationError(recipe_dir_rel, f"required file {filename} must not be a symlink")
            )
        elif not required_path.is_file():
            errors.append(
                ValidationError(recipe_dir_rel, f"missing required file {filename}")
            )

    entrypoint = manifest.get("entrypoint")
    if not isinstance(entrypoint, str):
        return errors

    path_messages = entrypoint_path_errors(entrypoint)
    for message in path_messages:
        errors.append(ValidationError(manifest_rel, message))
    if path_messages:
        return errors

    entrypoint_path = recipe_dir / entrypoint
    entrypoint_rel = rel(root, entrypoint_path)
    if entrypoint_path.is_symlink():
        errors.append(ValidationError(entrypoint_rel, "entrypoint must not be a symlink"))
        return errors
    if not entrypoint_path.is_file():
        errors.append(ValidationError(recipe_dir_rel, f"missing entrypoint file '{entrypoint}'"))
        return errors

    if not os.access(entrypoint_path, os.X_OK):
        errors.append(ValidationError(entrypoint_rel, "entrypoint is not executable"))

    if entrypoint.endswith(".sh"):
        try:
            result = subprocess.run(
                ["bash", "-n", entrypoint_rel],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            errors.append(
                ValidationError(entrypoint_rel, f"cannot run bash syntax check: {exc}")
            )
        else:
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                if detail:
                    errors.append(
                        ValidationError(
                            entrypoint_rel,
                            f"shell syntax check failed: {detail}",
                        )
                    )
                else:
                    errors.append(
                        ValidationError(entrypoint_rel, "shell syntax check failed")
                    )

    return errors


def validate_repository(root: Path) -> Tuple[List[ValidationError], int]:
    errors: List[ValidationError] = []

    schema_dir = root / "schema"
    schema_dir_rel = rel(root, schema_dir)
    schema_path = root / SCHEMA_REL_PATH
    schema_rel = rel(root, schema_path)
    if schema_dir.is_symlink():
        errors.append(
            ValidationError(schema_dir_rel, "schema directory must not be a symlink")
        )
    elif not schema_dir.is_dir():
        errors.append(ValidationError(schema_dir_rel, "missing schema directory"))
    elif schema_path.is_symlink():
        errors.append(
            ValidationError(schema_rel, "recipe JSON Schema must not be a symlink")
        )
    elif not schema_path.is_file():
        errors.append(ValidationError(schema_rel, "missing recipe JSON Schema"))
    else:
        schema, parse_error = load_json(schema_path)
        if parse_error is not None:
            errors.append(ValidationError(schema_rel, parse_error))
        elif not isinstance(schema, dict):
            errors.append(
                ValidationError(schema_rel, "recipe JSON Schema must be an object")
            )
        else:
            for message in schema_contract_errors(schema):
                errors.append(ValidationError(schema_rel, message))

    supported_runtimes, runtime_errors = validate_runtimes_config(root)
    errors.extend(runtime_errors)

    if runtime_errors:
        return errors, 0

    manifests, discovery_errors = discover_recipe_manifests(root, supported_runtimes)
    errors.extend(discovery_errors)
    for manifest_path in manifests:
        manifest_rel = rel(root, manifest_path)
        if manifest_path.is_symlink():
            errors.append(
                ValidationError(manifest_rel, "manifest must not be a symlink")
            )
            continue
        manifest, parse_error = load_json(manifest_path)
        if parse_error is not None:
            errors.append(ValidationError(manifest_rel, parse_error))
            continue
        if not isinstance(manifest, dict):
            errors.append(ValidationError(manifest_rel, "manifest must be a JSON object"))
            continue

        errors.extend(manifest_field_errors(manifest_rel, manifest))
        errors.extend(validate_recipe_layout(root, manifest_path, manifest))

    return errors, len(manifests)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate inference recipe manifests.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current working directory)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: root directory does not exist: {root}", file=sys.stderr)
        return 1

    errors, recipe_count = validate_repository(root)
    if errors:
        for error in sorted(errors, key=lambda item: (item.path, item.message)):
            print(str(error), file=sys.stderr)
        print(
            f"validation failed with {len(errors)} error(s)",
            file=sys.stderr,
        )
        return 1

    print(f"validated {recipe_count} recipe(s) successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
