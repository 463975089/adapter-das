"""Device-count evidence must cover every requested HCU ordinal."""

import json
import contextlib
import io
import sys
from types import SimpleNamespace

import pytest

from tools import l15_runtime as runtime


@pytest.mark.parametrize("fault", ["clean", "count", "ordinal", "allocation"])
def test_preflight_rejects_missing_or_unusable_tp_device(tmp_path, monkeypatch, fault):
    up = tmp_path / "upstream"
    adapter = tmp_path / "adapter"
    manifest = tmp_path / "bindings.json"
    manifest.write_text(json.dumps({"upstream": {"binding_files": {}}}))
    monkeypatch.setattr(runtime, "audit_source", lambda *args: {"passed": True})
    monkeypatch.setattr(runtime, "git_stamp", lambda *args: {"commit": "fixed"})

    def probe(argv, **kwargs):
        assert argv[-1] == "2"
        if fault == "allocation":
            return SimpleNamespace(returncode=1, stdout="", stderr="second device unusable")
        evidence = {
            "device_count": 1 if fault == "count" else 2,
            "devices": [{"index": 0}, {"index": 0 if fault == "ordinal" else 1}],
            "origins": {"sglang": str(up / "python/sglang/__init__.py"),
                        "sgl_kernel": str(adapter / "sgl_kernel/__init__.py"),
                        "hcu_adapter": str(adapter / "hcu_adapter/__init__.py")},
        }
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="L15_RUNTIME " + json.dumps(evidence))

    monkeypatch.setattr(runtime.subprocess, "run", probe)
    if fault == "clean":
        assert runtime.preflight(up, adapter, manifest, device_count=2)["device_count"] == 2
    else:
        with pytest.raises(ValueError, match="preflight"):
            runtime.preflight(up, adapter, manifest, device_count=2)


@pytest.mark.parametrize("count", [True, False, 0, -1, "2"])
def test_preflight_invalid_count_fails_before_touching_devices(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        runtime.preflight(tmp_path, tmp_path, tmp_path / "missing.json", device_count=count)


@pytest.mark.parametrize("host", ["adapter", "adapter+plugin"])
@pytest.mark.parametrize("registered", [False, True])
def test_tp_probe_keeps_plugin_admission_and_checks_each_device(
    tmp_path, monkeypatch, host, registered
):
    """Execute the generated child probe, including plugin and device guards."""
    up = tmp_path / "upstream"
    adapter = tmp_path / "adapter"
    manifest = tmp_path / "bindings.json"
    manifest.write_text(json.dumps({"upstream": {"binding_files": {}}}))
    monkeypatch.setattr(runtime, "audit_source", lambda *args: {"passed": True})
    monkeypatch.setattr(runtime, "git_stamp", lambda *args: {"commit": "fixed"})
    allocations, loaded = [], []

    def probe(argv, **kwargs):
        assert kwargs["env"]["SGLANG_PLUGINS"] == (
            "hcu" if host == "adapter+plugin" else "none")
        current = []

        @contextlib.contextmanager
        def device(index):
            current.append(index)
            try:
                yield
            finally:
                current.pop()

        def ones(*args, **kwargs):
            allocations.append(current[-1])
            return SimpleNamespace(add=lambda value: SimpleNamespace(item=lambda: 2))

        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 2, device=device,
            synchronize=lambda: None,
            get_device_properties=lambda index: SimpleNamespace(
                name="HCU", gcnArchName="test-arch"),
        )
        sources = [("hook", None, SimpleNamespace(plugin_name="hcu"))] if registered else []
        plugins = SimpleNamespace(load_plugins=lambda: loaded.append(True))
        with monkeypatch.context() as child:
            child.setitem(sys.modules, "torch", SimpleNamespace(
                version=SimpleNamespace(hip="test"), __version__="test",
                cuda=cuda, ones=ones))
            child.setitem(sys.modules, "sglang", SimpleNamespace(
                srt=SimpleNamespace(plugins=plugins)))
            child.setitem(sys.modules, "sglang.srt", SimpleNamespace(plugins=plugins))
            child.setitem(sys.modules, "sglang.srt.plugins", plugins)
            child.setitem(sys.modules, "sglang.srt.plugins.hook_registry",
                          SimpleNamespace(HookRegistry=SimpleNamespace(
                              _hooks={"target": sources})))
            child.setattr("importlib.util.find_spec", lambda name: SimpleNamespace(
                origin=str((up / "python" if name == "sglang" else adapter)
                           / name / "__init__.py")))
            child.setenv("SGLANG_PLUGINS", kwargs["env"]["SGLANG_PLUGINS"])
            child.setattr(sys, "argv", ["-c", argv[-1]])
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(compile(argv[2], "<preflight-child>", "exec"), {})
            except AssertionError as exc:
                return SimpleNamespace(returncode=1, stdout=output.getvalue(), stderr=str(exc))
        return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", probe)
    if host == "adapter+plugin" and not registered:
        with pytest.raises(ValueError, match="did not register any hook"):
            runtime.preflight(up, adapter, manifest, host, device_count=2)
    else:
        evidence = runtime.preflight(up, adapter, manifest, host, device_count=2)
        assert [d["index"] for d in evidence["devices"]] == [0, 1]
        assert evidence["host"] == host
        assert evidence["plugin"]["loaded"] is (host == "adapter+plugin")
    assert allocations == [0, 1]
    assert loaded == ([True] if host == "adapter+plugin" else [])
