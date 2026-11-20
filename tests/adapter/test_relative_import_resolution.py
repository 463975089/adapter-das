# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Static resolution audit for every relative import under hcu_adapter/.

Function-local ``from .. import X`` lines escape import-smoke (module import
never executes them) and pyflakes (valid syntax, resolved at runtime only).
Two such lines survived the layout migration and shipped as serve-path
ImportErrors caught by bridge isolation: infra/model_env.py:185 and
infra/hcu_env.py:824 (L1.6 caught the first, 2026-11-19). This test parses
the whole tree with ast and RESOLVES every relative import -- module anchor
and imported names -- against the filesystem, so the class cannot recur.
"""

import ast
import os
import re

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG = os.path.join(REPO, "hcu_adapter")


def _init_exports(init_path):
    """Names bound at module level of an __init__.py (assign/import/def)."""
    if not os.path.exists(init_path):
        return set()
    tree = ast.parse(open(init_path, encoding="utf-8").read())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _iter_bad():
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, REPO).replace("\\", "/")
            parts = rel.split("/")  # hcu_adapter/infra/x.py
            tree = ast.parse(open(full, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                anchor = parts[:-1]
                up = node.level - 1
                if up > len(anchor) - 1:
                    yield f"{rel}:{node.level} beyond package root"
                    continue
                base = anchor[: len(anchor) - up]
                segs = base + (node.module.split(".") if node.module else [])
                cand = os.path.join(REPO, *segs)
                if not (os.path.exists(cand + ".py") or os.path.isdir(cand)):
                    yield f"{rel}:{node.lineno} -> {'.'.join(segs)} (module missing)"
                    continue
                if os.path.isdir(cand):
                    exports = _init_exports(os.path.join(cand, "__init__.py"))
                    for alias in node.names:
                        n = alias.name
                        if (
                            os.path.exists(os.path.join(cand, n + ".py"))
                            or os.path.isdir(os.path.join(cand, n))
                            or n in exports
                        ):
                            continue
                        yield (
                            f"{rel}:{node.lineno} name {n!r} missing from "
                            f"{'.'.join(segs)} (moved sibling? use 'from . import')"
                        )


def test_every_relative_import_resolves():
    bad = list(_iter_bad())
    assert not bad, "\n".join(bad)


def test_the_two_serve_path_regressions_stay_fixed():
    """Pin the exact lines that shipped broken (guards against reverts)."""
    src = open(
        os.path.join(PKG, "infra", "model_env.py"), encoding="utf-8"
    ).read()
    assert "from . import registry" in src
    assert "from .. import registry" not in src
    src = open(os.path.join(PKG, "infra", "hcu_env.py"), encoding="utf-8").read()
    assert "from . import registry as _reg" in src
