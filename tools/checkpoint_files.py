# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Streaming byte fingerprints for the files used by local HF checkpoints."""

import hashlib
import json
from pathlib import Path, PurePosixPath

_ASSETS = ("*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken")


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(model):
    model = Path(model)
    files = {p.relative_to(model).as_posix(): p for pattern in _ASSETS
             for p in model.glob(pattern)}
    weights = {p.relative_to(model).as_posix(): p for p in model.glob("*.safetensors")}
    index = model / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("checkpoint index requires a nonempty weight_map")
        for name in mapping.values():
            if (not isinstance(name, str) or "\\" in name
                    or PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts
                    or not name.endswith(".safetensors")):
                raise ValueError("invalid checkpoint shard path")
            path = model / name
            if not path.is_file():
                raise ValueError(f"missing checkpoint shard: {name}")
            weights[name] = path
    if not (model / "config.json").is_file() or not weights:
        raise ValueError("reference requires local config and real safetensors")
    files.update(weights)
    if any(not path.is_file() for path in files.values()):
        raise ValueError("checkpoint asset is not a regular file")
    return files


def fingerprint(model):
    """Hash actual bytes, rejecting in-flight changes and incomplete indices."""
    files = _files(model)
    result, stamps = {}, {}
    for name, path in sorted(files.items()):
        before = path.stat()
        digest = _digest(path)
        after = path.stat()
        if _identity(before) != _identity(after):
            raise ValueError(f"checkpoint asset changed while hashing: {name}")
        stamps[name] = _identity(after)
        result[name] = digest
    if set(_files(model)) != set(files):
        raise ValueError("checkpoint asset inventory changed while hashing")
    if any(_identity(path.stat()) != stamps[name] for name, path in files.items()):
        raise ValueError("checkpoint asset changed while hashing other files")
    # Metadata may have coarse timestamps or be cached on a shared filesystem.
    # Confirm bytes too: earlier assets can change while later shards are read,
    # without producing any observable stat difference.
    if any(_digest(path) != result[name] for name, path in sorted(files.items())):
        raise ValueError("checkpoint asset bytes changed while hashing other files")
    if set(_files(model)) != set(files):
        raise ValueError("checkpoint asset inventory changed during byte confirmation")
    return result


def audit(model, expected):
    """Keep a failed recheck report instead of discarding preceding evidence."""
    try:
        actual = fingerprint(model)
    except (OSError, ValueError) as exc:
        return {"passed": False, "error": str(exc)}
    return {"passed": actual == expected, "files_sha256": actual}
