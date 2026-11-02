"""Portable contracts for the vLLM optimization extension host.

Run: HCU_ADAPTER_HOST=none python3 -m pytest -q \
  tests/adapter/test_vllm_extensions.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import pytest

try:
    from hcu_adapter import extensions
except ImportError:
    extensions = None


def _api():
    assert extensions is not None, "hcu_adapter.extensions is not implemented"
    return extensions


@dataclass
class FakeEntryPoint:
    name: str
    factory: object
    loads: int = 0

    def load(self):
        self.loads += 1
        return self.factory


def _context(*, required=(), capabilities=()):
    api = _api()
    return api.ExtensionContext(
        adapter_version="0.1.0",
        vllm_version="0.32.0+rocm723",
        requested_extensions=required,
        requested_capabilities=capabilities,
    )


def _factory(
    name,
    *,
    api_version=1,
    priority=0,
    capabilities=(),
    activated=None,
):
    def make():
        def activate(_context):
            if activated is not None:
                activated.append(name)

        return _api().ExtensionSpec(
            name=name,
            api_version=api_version,
            priority=priority,
            capabilities=capabilities,
            activate=activate,
        )

    return make


def test_context_and_reports_freeze_caller_owned_capability_sets():
    api = _api()
    requested = {"rms-norm"}
    context = api.ExtensionContext(
        "0.1.0", "0.32.0", requested_capabilities=requested
    )
    capabilities = {"rms-norm"}
    report = api.ExtensionReport(
        "das_hcu",
        "das-hcu",
        api.ExtensionStatus.ACTIVE,
        capabilities,
        "",
    )

    requested.add("silu-and-mul")
    capabilities.add("silu-and-mul")

    assert api.EXTENSION_API_VERSION == 1
    assert context.requested_capabilities == frozenset({"rms-norm"})
    assert report.capabilities == frozenset({"rms-norm"})


def test_extensions_load_once_and_activate_by_priority_then_name():
    activated = []
    points = [
        FakeEntryPoint("low", _factory("low", priority=1, activated=activated)),
        FakeEntryPoint("zeta", _factory("zeta", priority=10, activated=activated)),
        FakeEntryPoint("alpha", _factory("alpha", priority=10, activated=activated)),
    ]

    reports = _api().activate_extensions(_context(), points)

    assert activated == ["alpha", "zeta", "low"]
    assert [point.loads for point in points] == [1, 1, 1]
    assert [report.entry_point_name for report in reports] == [
        "alpha",
        "low",
        "zeta",
    ]


def test_duplicate_entry_point_names_fail_without_loading():
    points = [
        FakeEntryPoint("same", _factory("first")),
        FakeEntryPoint("same", _factory("second")),
    ]

    reports = _api().activate_extensions(_context(), points)

    assert [point.loads for point in points] == [0, 0]
    assert len(reports) == 2
    assert all(report.status is _api().ExtensionStatus.FAILED for report in reports)
    assert all("duplicate entry-point name" in report.reason for report in reports)


def test_duplicate_declared_names_fail_without_activation():
    activated = []
    reports = _api().activate_extensions(
        _context(),
        [
            FakeEntryPoint("first", _factory("same", activated=activated)),
            FakeEntryPoint("second", _factory("same", activated=activated)),
        ],
    )

    assert activated == []
    assert len(reports) == 2
    assert all("duplicate declared extension name" in report.reason for report in reports)


@pytest.mark.parametrize("required", (False, True))
def test_duplicate_capability_owners_never_activate_even_with_different_priorities(required):
    api = _api()
    activated = []
    points = [
        FakeEntryPoint("a", _factory("alpha", priority=100,
            capabilities={"attention", "norm"}, activated=activated)),
        FakeEntryPoint("b", _factory("beta", priority=1,
            capabilities={"attention"}, activated=activated)),
        FakeEntryPoint("c", _factory("independent", capabilities={"moe"},
            activated=activated)),
    ]
    context = _context(capabilities={"attention"} if required else ())
    if required:
        with pytest.raises(api.RequiredExtensionError, match="attention") as caught:
            api.activate_extensions(context, points)
        reports = caught.value.reports
    else:
        reports = api.activate_extensions(context, points)
    assert activated == ["independent"]
    assert [point.loads for point in points] == [1, 1, 1]
    assert [report.status for report in reports] == [
        api.ExtensionStatus.FAILED, api.ExtensionStatus.FAILED, api.ExtensionStatus.ACTIVE,
    ]
    assert all("'attention': alpha, beta" in r.reason for r in reports[:2])


def test_incompatible_api_does_not_reserve_capability_from_valid_owner():
    api = _api()
    activated = []
    reports = api.activate_extensions(_context(capabilities={"attention"}), [
        FakeEntryPoint("bad", _factory("old", api_version=2, capabilities={"attention"})),
        FakeEntryPoint("good", _factory("current", capabilities={"attention"}, activated=activated)),
    ])
    assert activated == ["current"]
    assert [r.status for r in reports] == [api.ExtensionStatus.FAILED, api.ExtensionStatus.ACTIVE]


@pytest.mark.parametrize("capabilities", ({""}, {" "}, {None}, {1}))
def test_invalid_capability_declaration_fails_before_activation(capabilities):
    activated = []
    report, = _api().activate_extensions(_context(), [FakeEntryPoint(
        "invalid", _factory("invalid", capabilities=capabilities, activated=activated),
    )])
    assert activated == []
    assert report.status is _api().ExtensionStatus.FAILED
    assert "capability names must be non-empty strings" in report.reason


def test_incompatible_optional_api_is_a_failed_report():
    report = _api().activate_extensions(
        _context(), [FakeEntryPoint("old", _factory("old", api_version=2))]
    )[0]

    assert report.status is _api().ExtensionStatus.FAILED
    assert report.extension_name == "old"
    assert report.reason == "incompatible extension API version: 2 (expected 1)"


@pytest.mark.parametrize("stage", ("load", "activation"))
def test_optional_failures_are_reports_not_startup_errors(stage):
    api = _api()
    if stage == "load":
        def broken():
            raise RuntimeError("provider unavailable")
    else:
        def broken():
            return api.ExtensionSpec(
                "broken",
                api.EXTENSION_API_VERSION,
                0,
                frozenset({"rms-norm"}),
                lambda _context: (_ for _ in ()).throw(
                    RuntimeError("registration unavailable")
                ),
            )

    report = api.activate_extensions(
        _context(), [FakeEntryPoint("optional", broken)]
    )[0]

    assert report.status is api.ExtensionStatus.FAILED
    assert f"{stage} failed" in report.reason


def test_required_extension_failure_raises_with_complete_reports():
    api = _api()

    def broken():
        raise RuntimeError("required provider unavailable")

    with pytest.raises(api.RequiredExtensionError, match="required") as raised:
        api.activate_extensions(
            _context(required={"required"}),
            [FakeEntryPoint("required", broken)],
        )

    assert len(raised.value.reports) == 1
    assert raised.value.reports[0].entry_point_name == "required"


def test_required_capability_counts_only_active_extensions():
    api = _api()

    def broken():
        return api.ExtensionSpec(
            "broken",
            api.EXTENSION_API_VERSION,
            0,
            frozenset({"rms-norm"}),
            lambda _context: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )

    with pytest.raises(api.RequiredExtensionError, match="rms-norm") as raised:
        api.activate_extensions(
            _context(capabilities={"rms-norm"}),
            [FakeEntryPoint("broken", broken)],
        )

    assert raised.value.reports[0].status is api.ExtensionStatus.FAILED


def test_injected_empty_entry_points_do_not_discover_metadata(monkeypatch):
    api = _api()
    monkeypatch.setattr(
        api.metadata,
        "entry_points",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata discovery must not run")
        ),
    )

    assert api.activate_extensions(_context(), entry_points=[]) == ()


def test_default_discovery_uses_only_adapter_extension_group(monkeypatch):
    api = _api()
    calls = []

    def discover(*, group):
        calls.append(group)
        return []

    monkeypatch.setattr(api.metadata, "entry_points", discover)

    assert api.activate_extensions(_context()) == ()
    assert calls == ["hcu_adapter.extensions"]


def test_importing_protocol_does_not_import_runtime_or_provider_packages():
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HCU_ADAPTER_HOST"] = "none"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; before=set(sys.modules); "
                "import hcu_adapter.extensions; "
                "loaded=set(sys.modules)-before; "
                "blocked=('torch','vllm','lightop','hcu_vllm_das'); "
                "assert not {n for n in loaded if n.split('.',1)[0] in blocked}, "
                "sorted(loaded)"
            ),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
