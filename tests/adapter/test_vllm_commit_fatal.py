"""Public activation/runtime contracts; CPU doubles and controlled actual fork."""

import builtins
from dataclasses import dataclass
import os
import select
import signal
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest

from hcu_adapter import extensions as ext
from hcu_adapter.bridges import vllm as bridge


class BaselineCommitSignal(RuntimeError):
    """Only lets RED exercise old swallowing, not just missing imports."""


def fatal(message="partially published"):
    return getattr(ext, "ExtensionCommitError", BaselineCommitSignal)(message)


@dataclass
class Point:
    name: str
    priority: int
    callback: object
    capabilities: tuple = ()

    def load(self):
        return lambda: ext.ExtensionSpec(self.name, 1, self.priority, self.capabilities, self.callback)


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(True, "gfx936", "CPU double"))
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible", lambda: "0.32.0+cpu")
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "cpu")
    points = []
    monkeypatch.setattr(ext, "_entry_points", lambda: points)
    return points


@pytest.fixture
def platform(monkeypatch):
    # Lazy __getattr__ caches the generated class. Ensure monkeypatch records
    # a real prior value, so the CPU-double class cannot leak to later suites.
    bridge.HcuPlatform
    class RocmPlatform:
        @classmethod
        def get_device_capability(cls):
            return object()

        @classmethod
        def get_attn_backend_cls(cls, selected, config, num_heads=None):
            return selected

    rocm = ModuleType("vllm.platforms.rocm")
    rocm.RocmPlatform = RocmPlatform
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = NS(FLASH_ATTN="FLASH", TRITON_ATTN="TRITON")
    monkeypatch.setitem(sys.modules, rocm.__name__, rocm)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.delitem(bridge.__dict__, "HcuPlatform", raising=False)
    imports = []
    real_import = bridge.import_module

    def load(name):
        if name == "hcu_vllm_das.attention.flash_attn":
            imports.append(name)
            return NS(HcuFlashAttentionBackend=NS(validate_configuration=lambda **kwargs: []))
        return real_import(name)

    monkeypatch.setattr(bridge, "import_module", load)
    return bridge.HcuPlatform, NS(_asdict=lambda: {}, use_mla=False, use_sparse=False), imports


def test_optional_clean_failure_stays_optional_and_later_owner_runs(startup):
    calls = []
    def fail(_):
        raise ImportError("optional provider absent")
    startup.extend([Point("clean", 10, fail), Point("later", 0, lambda _: calls.append(1))])
    bridge.hcu_runtime_plugin()
    bridge.hcu_runtime_plugin()
    assert calls == [1]
    assert [r.status.value for r in bridge.extension_reports()] == ["failed", "active"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_public_host_fatal_stops_later_owners_and_preserves_reports(wrapped):
    calls = []
    def fail(_):
        calls.append("published")
        try:
            raise fatal()
        except Exception as error:
            if wrapped:
                raise ValueError("existing wrapper") from error
            raise
    points = [Point("early", 20, lambda _: None, ("hcu-flash-attention",)),
              Point("dirty", 10, fail), Point("later", 0, lambda _: calls.append("BAD"))]
    with pytest.raises(getattr(ext, "ExtensionCommitError", BaselineCommitSignal)) as caught:
        ext.activate_extensions(ext.ExtensionContext("cpu", "0.32.0"), points)
    assert calls == ["published"]
    assert {r.entry_point_name:r.status.value for r in caught.value.reports} == {"early":"active", "dirty":"failed"}


def test_fatal_factory_does_not_continue_activation():
    calls = []
    def load():
        raise fatal("factory published state")
    with pytest.raises(getattr(ext, "ExtensionCommitError", BaselineCommitSignal)) as caught:
        ext.activate_extensions(ext.ExtensionContext("cpu", "0.32.0"),
                                [NS(name="bad", load=load), Point("later", 0, lambda _: calls.append(1))])
    assert not calls and caught.value.reports[0].entry_point_name == "bad"


@pytest.mark.parametrize("render_error", [ValueError, KeyboardInterrupt])
def test_runtime_refusal_survives_unrenderable_action_error(startup, render_error):
    calls = []

    class BrokenString(RuntimeError):
        def __str__(self):
            raise render_error("diagnostic formatting failed")

    def action():
        calls.append("attempt")
        raise BrokenString()

    actions = bridge._RUNTIME_ACTIONS
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="details unavailable") as caught:
        actions.run_once("vllm-runtime", action)
    assert isinstance(caught.value.__cause__, BrokenString)
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.require_runtime_healthy()
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.run_once("vllm-runtime", lambda: calls.append("BAD retry"))
    if hasattr(os, "fork"):
        def child():
            with pytest.raises(bridge.LatchedVllmRuntimeError):
                bridge.require_runtime_healthy()
            with pytest.raises(bridge.LatchedVllmRuntimeError):
                actions.run_once("vllm-runtime", lambda: None)
        child_result(child)
    assert calls == ["attempt"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual fork")
def test_runtime_unrenderable_error_fork_sees_refusal_before_formatting(startup):
    observations = []

    class ForkString(RuntimeError):
        def __str__(self):
            def child():
                with pytest.raises(bridge.LatchedVllmRuntimeError):
                    bridge.require_runtime_healthy()
            child_result(child)
            # Read the real stored refusal from this thread through the public
            # query; reentrant formatting must not find a healthy runtime.
            with pytest.raises(bridge.LatchedVllmRuntimeError):
                bridge.require_runtime_healthy()
            observations.append("refused")
            raise KeyboardInterrupt("formatting interrupted after fork")

    def action():
        raise ForkString()

    with pytest.raises(bridge.LatchedVllmRuntimeError, match="details unavailable"):
        bridge._RUNTIME_ACTIONS.run_once("vllm-runtime", action)
    assert observations == ["refused"]


@pytest.mark.parametrize("stage", ["discovery", "activation"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("render_error", [ValueError, KeyboardInterrupt])
def test_unrenderable_fatal_preserves_failed_report_and_stops_later(stage, wrapped, render_error):
    calls = []
    base = RuntimeError if wrapped else ext.ExtensionCommitError

    class BrokenString(base):
        def __str__(self):
            raise render_error("diagnostic formatting failed")

    def fail(*args):
        if wrapped:
            try:
                raise ext.ExtensionCommitError("explicit publication failure")
            except ext.ExtensionCommitError as cause:
                raise BrokenString("original failure") from cause
        raise BrokenString("original failure")

    bad = (NS(name="dirty", load=fail) if stage == "discovery"
           else Point("dirty", 10, fail, ("dirty-capability",)))
    points = [Point("early", 20, lambda _: calls.append("early")), bad,
              Point("later", 0, lambda _: calls.append("BAD later"))]
    with pytest.raises(ext.ExtensionCommitError) as caught:
        ext.activate_extensions(ext.ExtensionContext("cpu", "0.32.0"), points)
    reports = caught.value.reports
    assert [r.entry_point_name for r in reports].count("dirty") == 1
    dirty = next(r for r in reports if r.entry_point_name == "dirty")
    assert dirty.status is ext.ExtensionStatus.FAILED
    assert "details unavailable" in dirty.reason
    assert isinstance(caught.value.__cause__, BrokenString)
    if stage == "activation":
        assert dirty.extension_name == "dirty" and dirty.capabilities == {"dirty-capability"}
        assert {r.entry_point_name: r.status.value for r in reports} == {"early": "active", "dirty": "failed"}
        assert calls == ["early"]
    else:
        assert calls == []


@pytest.mark.parametrize("stage", ["discovery", "activation"])
@pytest.mark.parametrize("required", [False, True])
def test_unrenderable_clean_error_keeps_optional_and_required_policy(stage, required):
    class BrokenString(ImportError):
        def __str__(self):
            raise KeyboardInterrupt("diagnostic formatting failed")

    def fail(*args):
        raise BrokenString()

    later = []
    point = NS(name="clean", load=fail) if stage == "discovery" else Point("clean", 10, fail)
    context = ext.ExtensionContext("cpu", "0.32.0", requested_extensions={"clean"} if required else ())
    points = [point, Point("later", 0, lambda _: later.append(1))]
    if required:
        with pytest.raises(ext.RequiredExtensionError) as caught:
            ext.activate_extensions(context, points)
        reports = caught.value.reports
    else:
        reports = ext.activate_extensions(context, points)
    assert {r.entry_point_name: r.status.value for r in reports} == {"clean": "failed", "later": "active"}
    assert "details unavailable" in reports[0].reason
    assert later == [1]


def test_unrenderable_wrapper_cannot_hide_fatal_cause_during_formatting():
    class MutatingString(RuntimeError):
        def __str__(self):
            self.__cause__ = None
            self.__context__ = None
            raise ValueError("diagnostic formatting changed its exception chain")

    def fail(_):
        try:
            raise ext.ExtensionCommitError("explicit publication failure")
        except ext.ExtensionCommitError as cause:
            raise MutatingString() from cause

    with pytest.raises(ext.ExtensionCommitError) as caught:
        ext.activate_extensions(ext.ExtensionContext("cpu", "0.32.0"), [Point("dirty", 0, fail)])
    assert caught.value.reports[0].status is ext.ExtensionStatus.FAILED
    assert isinstance(caught.value.__cause__, MutatingString)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("selected,mla,sparse", [("FLASH",False,False), (None,False,False), (None,True,False), (None,True,True)])
def test_caught_boot_failure_blocks_all_routes_without_import_or_reactivation(startup, platform, monkeypatch, required, selected, mla, sparse):
    cls, cfg, imports = platform
    calls = []
    def fail(_):
        calls.append("commit" if not required else "clean")
        raise ValueError("clean required failure") if required else fatal()
    startup.extend([Point("early", 20, lambda _: None, ("hcu-flash-attention",)), Point("dirty", 10, fail)])
    if required:
        # Existing PR17 public context API, supplied by this CPU caller. This
        # tests required/fatal compatibility without importing PR20 env policy.
        activate = bridge.activate_extensions
        monkeypatch.setattr(bridge, "activate_extensions", lambda context, **kwargs: activate(
            ext.ExtensionContext(context.adapter_version, context.vllm_version,
                                 requested_extensions={"dirty"}), **kwargs))
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.hcu_runtime_plugin()
    # Reports remain diagnostic; changing them cannot erase runtime failure.
    bridge._RUNTIME_ACTIONS.record_extension_reports((ext.ExtensionReport("flash", "flash", ext.ExtensionStatus.ACTIVE, {"hcu-flash-attention"}, ""),))
    cfg.use_mla, cfg.use_sparse = mla, sparse
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        cls.get_attn_backend_cls(selected, cfg)
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.hcu_runtime_plugin()
    assert len(calls) == 1 and imports == []


def test_health_query_is_readonly_reentrant_during_activation(startup):
    calls = []
    def callback(_):
        bridge.require_runtime_healthy()
        bridge.require_runtime_healthy()
        calls.append(1)
    startup.append(Point("provider", 0, callback))
    bridge.hcu_runtime_plugin()
    assert calls == [1]


def test_selector_waiting_for_concurrent_failure_rejects(startup, platform):
    cls, cfg, imports = platform
    entered, release, selecting = threading.Event(), threading.Event(), threading.Event()
    results = []
    def fail(_):
        entered.set()
        assert release.wait(3)
        raise fatal()
    startup.extend([Point("early", 20, lambda _: None, ("hcu-flash-attention",)), Point("dirty", 10, fail)])
    def boot():
        try: bridge.hcu_runtime_plugin()
        except bridge.LatchedVllmRuntimeError: results.append("boot-refused")
    def choose():
        selecting.set()
        try: cls.get_attn_backend_cls("FLASH", cfg)
        except bridge.LatchedVllmRuntimeError: results.append("selector-refused")
    a=threading.Thread(target=boot,daemon=True); b=threading.Thread(target=choose,daemon=True)
    a.start()
    try:
        assert entered.wait(3)
        b.start(); assert selecting.wait(3)
    finally: release.set()
    a.join(4); b.join(4)
    assert not a.is_alive() and not b.is_alive()
    assert sorted(results)==["boot-refused","selector-refused"] and not imports


def child_result(callback):
    """Actual fork with child alarm + bounded parent wait/kill, no deadlocks."""
    reader,writer=os.pipe()
    pid=os.fork()
    if pid==0:
        os.close(reader);signal.alarm(3)
        try:
            callback()
            os.write(writer,b"ok");os._exit(0)
        except BaseException:
            os._exit(7)
    os.close(writer)
    try:
        ready,_,_=select.select([reader],[],[],5)
        if not ready:
            os.kill(pid,signal.SIGKILL)
        data=os.read(reader,32) if ready else b"timeout"
        _,status=os.waitpid(pid,0)
        assert data==b"ok" and os.waitstatus_to_exitcode(status)==0
    finally: os.close(reader)


@pytest.mark.skipif(not hasattr(os,"fork"),reason="requires actual fork")
@pytest.mark.parametrize("active", [False,True])
def test_failed_or_inprogress_runtime_stays_refused_after_actual_fork(startup, active):
    entered,release=threading.Event(),threading.Event();results=[]
    def fail(_):
        entered.set()
        if active: assert release.wait(6)
        raise fatal()
    startup.append(Point("dirty",0,fail))
    def boot():
        try: bridge.hcu_runtime_plugin()
        except bridge.LatchedVllmRuntimeError: results.append("refused")
    thread=threading.Thread(target=boot,daemon=True)
    if active:
        thread.start();assert entered.wait(3)
    else: boot()
    def child():
        assert bridge.extension_reports()==()
        for call in (bridge.require_runtime_healthy,bridge.hcu_runtime_plugin):
            try: call()
            except bridge.LatchedVllmRuntimeError: pass
            else: raise AssertionError("inherited failed/incomplete runtime was erased")
    try: child_result(child)
    finally:
        release.set()
        if active: thread.join(4)
    assert results==["refused"]


@pytest.mark.skipif(not hasattr(os,"fork"),reason="requires actual fork")
def test_healthy_fork_and_unrelated_failed_action_keep_original_reset(startup):
    bridge.hcu_runtime_plugin()
    def fail(): raise ValueError("unrelated action")
    with pytest.raises(bridge.LatchedVllmRuntimeError): bridge._RUNTIME_ACTIONS.run_once("other",fail)
    def child():
        assert not bridge.extension_reports()
        bridge.require_runtime_healthy()
        bridge._RUNTIME_ACTIONS.run_once("other",lambda: None)
        bridge.hcu_runtime_plugin()
    child_result(child)
    with pytest.raises(bridge.LatchedVllmRuntimeError): bridge._RUNTIME_ACTIONS.run_once("other",lambda: None)


def test_fresh_interpreter_does_not_inherit_failed_runtime(startup):
    startup.append(Point("dirty", 0, lambda _: (_ for _ in ()).throw(fatal())))
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.hcu_runtime_plugin()
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "from hcu_adapter.bridges import vllm as b; "
         "b.require_runtime_healthy(); "
         "assert b.extension_reports() == (); "
         "b._RUNTIME_ACTIONS.run_once('vllm-runtime', lambda: None); "
         "b.require_runtime_healthy()"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.require_runtime_healthy()


@pytest.mark.parametrize("shape", ["plain","cause","context","suppressed","cycle","group"])
def test_classifier_preserves_public_signal_through_wrappers(shape):
    error=fatal()
    if shape=="cause":
        wrapper=ValueError("wrapper");wrapper.__cause__=error;error=wrapper
    elif shape in ("context","suppressed"):
        try:
            try: raise error
            except Exception: raise ValueError("wrapper") from None
        except ValueError as wrapper: error=wrapper
    elif shape=="cycle":
        wrapper=ValueError("wrapper");wrapper.__context__=error;error.__cause__=wrapper;error=wrapper
    elif shape=="group":
        group=getattr(builtins,"ExceptionGroup",None)
        if group is None: pytest.skip("no native ExceptionGroup")
        error=group("nested",[ValueError("ordinary"),group("inner",[error])])
    assert ext.is_fatal_extension_error(error)


def test_classifier_does_not_guess_from_messages_or_cycle_without_signal():
    error=RuntimeError("OOM fatal commit partially published")
    error.__cause__=error
    assert not ext.is_fatal_extension_error(error)
