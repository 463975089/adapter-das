# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Cross-framework bootstrap guard ordering and isolation.

Run: python3 -m pytest -q tests/adapter/test_common_bridge.py
"""

from types import SimpleNamespace

from hcu_adapter.bridges import common
import hcu_adapter


def test_common_guards_run_in_pre_import_order(monkeypatch):
    events = []
    monkeypatch.setattr(
        hcu_adapter,
        "hcu_env",
        SimpleNamespace(log_summary=lambda: events.append("environment")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "nccl_topo_guard",
        SimpleNamespace(install=lambda: events.append("nccl")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "tvmffi_warmup",
        SimpleNamespace(install=lambda: events.append("tvmffi")),
        raising=False,
    )

    common.install()

    assert events == ["environment", "nccl", "tvmffi"]


def test_one_common_guard_failure_does_not_block_later_guards(monkeypatch):
    events = []

    def fail_environment():
        events.append("environment")
        raise RuntimeError("summary failed")

    monkeypatch.setattr(
        hcu_adapter,
        "hcu_env",
        SimpleNamespace(log_summary=fail_environment),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "nccl_topo_guard",
        SimpleNamespace(install=lambda: events.append("nccl")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "tvmffi_warmup",
        SimpleNamespace(install=lambda: events.append("tvmffi")),
        raising=False,
    )

    common.install()

    assert events == ["environment", "nccl", "tvmffi"]
