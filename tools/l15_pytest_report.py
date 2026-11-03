"""Pytest plugin used by L1.5 runs to emit machine-readable per-node outcomes.

Loaded explicitly with ``-p tools.l15_pytest_report``.  The output path is
required through ``L15_PYTEST_REPORT``; no repository or upstream files are
modified by the plugin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_RESULTS: dict[str, dict[str, Any]] = {}
_COLLECTION_ERRORS: list[str] = []


def pytest_sessionstart(session: Any) -> None:
    _RESULTS.clear()
    _COLLECTION_ERRORS.clear()


def pytest_collectreport(report: Any) -> None:
    if report.failed:
        _COLLECTION_ERRORS.append(_bounded(report.longrepr))


def _bounded(value: object, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


def _normalize_nodeid(nodeid: str) -> str:
    if nodeid.startswith("registered/"):
        return "test/" + nodeid
    return nodeid


def pytest_runtest_logreport(report: Any) -> None:
    if report.when not in {"setup", "call", "teardown"}:
        return
    nodeid = _normalize_nodeid(report.nodeid)
    record = _RESULTS.setdefault(
        nodeid,
        {"nodeid": nodeid, "outcome": "not_collected", "phases": {}},
    )
    phase = {"outcome": report.outcome}
    if hasattr(report, "wasxfail"):
        phase["wasxfail"] = _bounded(report.wasxfail)
    if report.failed:
        phase["fingerprint"] = _bounded(report.longrepr)
    if report.skipped:
        phase["reason"] = _bounded(report.longrepr)
    # pytest-subtests emits multiple call reports for one node. Preserve each
    # one: a final passing parent must not erase a skipped/failed subcase.
    if hasattr(report, "context"):
        record.setdefault("subtests", []).append(phase)
        return
    record["phases"][report.when] = phase

    if report.when == "setup" and report.skipped:
        record["outcome"] = "skipped"
    elif report.when == "setup" and report.failed:
        record["outcome"] = "error"
    elif report.when == "call":
        if report.skipped and hasattr(report, "wasxfail"):
            record["outcome"] = "xfail"
        elif report.passed and hasattr(report, "wasxfail"):
            record["outcome"] = "xpass"
        else:
            record["outcome"] = report.outcome
    elif report.when == "teardown" and report.failed:
        record["outcome"] = "error"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    output = os.environ.get("L15_PYTEST_REPORT")
    if not output:
        raise RuntimeError("L15_PYTEST_REPORT is required by l15_pytest_report")
    document = {
        "schema_version": 1,
        "artifact": "l15-pytest-outcomes",
        "exitstatus": int(exitstatus),
        "tests": sorted(_RESULTS.values(), key=lambda item: item["nodeid"]),
        "collection_errors": list(_COLLECTION_ERRORS),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
