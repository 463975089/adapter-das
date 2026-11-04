# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Render BACKEND_CATALOG into docs/tier-matrix.md (the tier panorama).

Answers "which op has which tier, which tier is missing, at what status"
from the single source of truth (BACKEND_CATALOG in hcu_adapter/hcu_config.py).
File naming (governance §3b) is only navigation; this table is the registry.

L0 `impl-naming` regenerates this in memory and fails on drift, so the
table cannot go stale unnoticed.

Run: python tools/gen_tier_matrix.py [--check]
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "tier-matrix.md"

# Tier column order; catalog branch keys are matched case-insensitively.
TIERS = ["triton", "cpp", "emu", "fork", "aiter", "lightop", "dtk", "vendor", "torch"]
GENERIC_KEYS = {"status", "note", "env", "default", "impl", "envs", "candidates"}


def load_catalog():
    sys.path.insert(0, str(ROOT))
    from hcu_adapter.hcu_config import BACKEND_CATALOG

    return BACKEND_CATALOG


def cell_for(branch):
    """Compact one catalog branch entry into a table cell."""
    if isinstance(branch, dict):
        if "status" in branch:
            st = str(branch["status"])
        else:
            envs = [k for k in branch if k.startswith("HCU_")]
            st = "routed" if envs else "entry"
        env = next((k for k in branch if k.startswith("HCU_")), "")
        note = str(branch.get("note", ""))[:36].replace("|", "/").replace("\n", " ")
        parts = [st]
        if env:
            parts.append(f"`{env}`")
        if note:
            parts.append(note)
        return " ".join(parts)
    return str(branch)[:60].replace("|", "/")


def render() -> str:
    catalog = load_catalog()
    lines = [
        "# 算子档位矩阵（自动生成，勿手改）",
        "",
        f"> 生成：`python tools/gen_tier_matrix.py`（最近 {date.today().isoformat()}）"
        "；事实源 = `BACKEND_CATALOG`（hcu_adapter/hcu_config.py）。",
        "> L0 `impl-naming` 做新鲜度校验改表须随 catalog 同 commit（governance §3b）。",
        "> 文件命名导航规则见 governance §3b；缺档 = 该行空白——那是实现排期的直接输入。",
        "",
    ]
    for op, body in catalog.items():
        if not isinstance(body, dict):
            lines.append(f"## {op}\n\n- {body}\n")
            continue
        backends = body.get("backends")
        if not isinstance(backends, dict):
            # catalog entry without per-tier backends: render verbatim summary
            summary = str(body.get("summary", "")).replace("\n", " ") or str(body)
            lines += [f"## {op}", "", summary, ""]
            continue
        rows = sorted(
            backends.items(),
            key=lambda kv: (TIERS.index(str(kv[0]).lower())
                            if str(kv[0]).lower() in TIERS else 99,
                            str(kv[0])),
        )
        lines.append(f"## {op}")
        if body.get("summary"):
            lines += ["", str(body["summary"]).replace("\n", " ")]
        var_line = ""
        if body.get("var"):
            var_line = f"`{body['var']}`"
        elif body.get("vars"):
            var_line = "`" + "`/`".join(
                next(iter(v.values())) for v in body["vars"].values() if isinstance(v, dict)
            ) + "`"
        lines += ["", "| 档位 | 状态 / 备注 |", "|---|---|"]
        for tier, val in rows:
            lines.append(f"| {tier} | {cell_for(val)} |")
        lines += ["", f"路由变量：{var_line or '（auto）'}", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the doc is stale")
    args = ap.parse_args()
    out = render()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        # ignore the date line when comparing (it changes daily)
        strip = lambda s: "\n".join(
            ln for ln in s.splitlines() if not ln.startswith("> 生成：")
        )
        if strip(out) != strip(current):
            print(
                "docs/tier-matrix.md is stale; regenerate: python tools/gen_tier_matrix.py",
                file=sys.stderr,
            )
            return 1
        return 0
    DOC.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(ROOT)} ({out.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
