#!/usr/bin/env python3
"""Inspect or evict only the PLE tensor's Linux page-cache range."""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import mmap
import os
from pathlib import Path
from typing import Any

from gguf import GGUFReader

PLE_TENSOR = "per_layer_token_embd.weight"


def shard_paths(path: Path) -> list[Path]:
    name = path.name
    if "-of-" not in name or not name.endswith(".gguf"):
        return [path]
    head, _, tail = name.rpartition("-of-")
    stem = head.rsplit("-", 1)[0]
    total = tail.removesuffix(".gguf")
    matches = sorted(glob.glob(str(path.parent / f"{stem}-*-of-{total}.gguf")))
    return [Path(match) for match in matches] or [path]


def find_tensor(path: Path, tensor_name: str = PLE_TENSOR) -> dict[str, Any]:
    for shard in shard_paths(path):
        reader = GGUFReader(shard, "r")
        for tensor in reader.tensors:
            if tensor.name == tensor_name:
                return {
                    "shard": shard,
                    "offset": int(tensor.data_offset),
                    "length": int(tensor.n_bytes),
                    "tensor_type": tensor.tensor_type.name,
                    "shape": [int(value) for value in tensor.shape],
                }
    raise ValueError(f"tensor {tensor_name!r} not found")


def resident_pages(path: Path, offset: int, length: int) -> tuple[int, int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    base = (offset // page_size) * page_size
    span = (offset - base) + length
    page_count = (span + page_size - 1) // page_size

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    descriptor = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(
            None,
            span,
            mmap.PROT_READ,
            mmap.MAP_SHARED,
            descriptor,
            base,
        )
        invalid = ctypes.c_void_p(-1).value
        if address in (None, 0, invalid):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vector = (ctypes.c_ubyte * page_count)()
            if libc.mincore(ctypes.c_void_p(address), span, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident = sum(1 for value in vector if value & 1)
        finally:
            libc.munmap(ctypes.c_void_p(address), span)
    finally:
        os.close(descriptor)
    return resident, page_count, page_size


def evict_range(path: Path, offset: int, length: int) -> None:
    if not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def build_record(info: dict[str, Any]) -> dict[str, Any]:
    resident, pages, page_size = resident_pages(
        info["shard"], info["offset"], info["length"]
    )
    return {
        "tensor": PLE_TENSOR,
        "shard": str(info["shard"]),
        "offset": info["offset"],
        "length": info["length"],
        "length_gib": info["length"] / 2**30,
        "tensor_type": info["tensor_type"],
        "shape": info["shape"],
        "page_size": page_size,
        "resident_pages": resident,
        "total_pages": pages,
        "resident_gib": resident * page_size / 2**30,
        "resident_percent": 100.0 * resident / pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    info = find_tensor(args.model)
    before = build_record(info)
    result: dict[str, Any] = {"before": before, "drop_requested": args.drop}
    if args.drop:
        evict_range(info["shard"], info["offset"], info["length"])
        result["after"] = build_record(info)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
