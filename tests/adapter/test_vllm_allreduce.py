# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Official target27c Python/dispatcher contracts; CPU provider/group doubles.

Core wrapper contracts migrated from the frozen experimental Plugin candidate.
No Plugin import/activation, native math, HCU admission or model pass is claimed.
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
from hcu_adapter import registry as adapter_registry, allreduce_aiter
from hcu_adapter.allreduce import MIB
from hcu_adapter.extensions import ExtensionCommitError
from hcu_adapter.bridges import vllm as bridge, vllm_allreduce as compat
from vllm import envs, _aiter_ops as ops_module
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.aiter_custom_all_reduce import AiterCustomAllreduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
REAL_PROVIDER_IDENTITY = compat._provider_identity


@pytest.mark.parametrize("kill,optin,child", [("aiter_ar", "1", "1"), ("", "0", "1"), ("", "1", "0")])
@pytest.mark.parametrize("required", [False, True])
def test_dispatch_off_is_not_fatal_even_when_availability_required(monkeypatch, provider, kill, optin, child, required):
    calls, _, _ = provider
    monkeypatch.setenv("HCU_DISABLE_OPS", kill)
    monkeypatch.setenv("HCU_AITER_AR", optin)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", child)
    monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", "not-parsed-while-off")
    original = ops_module.rocm_aiter_ops.register_custom_allreduce_ops
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops",
                        staticmethod(lambda: pytest.fail("OFF registered fused ops")))
    status = compat.install(require_available=required)
    assert status.installed and not status.selected and not status.fused_ops_registered
    assert status.available is (True if required else None)
    assert not ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled()
    assert json.loads(compat.effective_compile_factor())["implementation"] == {}
    assert compat.state() == status and calls == []
    with pytest.raises(compat.AllReduceCompatibilityError, match="dispatch is OFF"):
        AiterCustomAllreduce(None, "cuda:0")
    assert calls == [] and compat._commit_failure is None
    # Restore only the test double, never a runtime registration rollback.
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops", staticmethod(original))


def test_default_runtime_is_neutral_and_does_not_import_optional_provider(monkeypatch):
    before = dict(envs.environment_variables)
    previous = compat.import_module
    def load(name):
        assert not name.startswith(("aiter", "vllm")), "unselected startup imported runtime/provider"
        return previous(name)
    monkeypatch.setattr(compat, "import_module", load)
    status = compat.prepare_runtime()
    assert not status.installed and status.available is None
    assert dict(envs.environment_variables) == before


@pytest.mark.parametrize("problem", ["size", "transport", "provider-abi", "group", "cache", "factor-owner", "fabric", "binding-owner", "registry-owner"])
def test_clean_prepare_failure_precedes_all_owned_mutations(monkeypatch, provider, problem):
    _, _, vendor = provider
    enable(monkeypatch)
    if problem == "size": monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", "0")
    elif problem == "transport": monkeypatch.setenv("AITER_AR_TRANSPORT", "invalid")
    elif problem == "provider-abi": monkeypatch.setattr(vendor, "CustomAllreduce", SimpleNamespace(__init__=lambda self: None))
    elif problem == "group": monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    elif problem == "cache": envs.enable_envs_cache()
    elif problem == "factor-owner": monkeypatch.setitem(envs.environment_variables, compat.FACTOR, lambda: "foreign")
    elif problem == "fabric": monkeypatch.setenv("AITER_AR_TRANSPORT", "fabric")
    elif problem == "binding-owner": monkeypatch.setattr(AiterCustomAllreduce, "__init__", lambda *a, **kw: None)
    elif problem == "registry-owner": adapter_registry.register("vllm_allreduce", "foreign", "foreign.target", "rebind")
    init, gate = AiterCustomAllreduce.__init__, ops_module.rocm_aiter_ops.__dict__["is_custom_all_reduce_enabled"]
    before, records = dict(envs.environment_variables), adapter_registry.records()
    with pytest.raises((compat.AllReduceCompatibilityError, TypeError)):
        compat.install()
    assert compat._installed is None and compat._commit_failure is None
    assert AiterCustomAllreduce.__init__ is init
    assert ops_module.rocm_aiter_ops.__dict__["is_custom_all_reduce_enabled"] is gate
    assert dict(envs.environment_variables) == before and adapter_registry.records() == records


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("missing", ["method", "error-type", "async"])
def test_public_target_abi_preflight_before_provider_and_mutation(monkeypatch, provider, required, missing):
    if not required: enable(monkeypatch)
    if missing == "method": monkeypatch.delattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops")
    elif missing == "error-type": monkeypatch.delattr(ops_module, "CustomAllreduceRegistrationError")
    else:
        async def bad(): pass
        monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops", staticmethod(bad))
    previous = compat.import_module
    def load(name):
        assert name != compat.PROVIDER, "missing ABI imported provider"
        return previous(name)
    monkeypatch.setattr(compat, "import_module", load)
    with pytest.raises(compat.AllReduceCompatibilityError, match="public Custom AR"):
        compat.install(require_available=required)
    assert compat._installed is None and compat.FACTOR not in envs.environment_variables


@pytest.mark.parametrize("failure", ["clean", "partial", "opaque", "invalid-result"])
def test_public_registration_clean_vs_commit_fatal(monkeypatch, provider, failure):
    enable(monkeypatch)
    init = AiterCustomAllreduce.__init__
    def register():
        if failure in ("clean", "partial"):
            raise ops_module.CustomAllreduceRegistrationError("CPU seam failure double", registration_started=failure == "partial")
        if failure == "opaque": raise RuntimeError("opaque failure double")
        return None
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops", staticmethod(register))
    error = ops_module.CustomAllreduceRegistrationError if failure == "clean" else ExtensionCommitError
    with pytest.raises(error): compat.install()
    assert (compat._commit_failure is None) == (failure == "clean")
    assert AiterCustomAllreduce.__init__ is init and compat._installed is None
    assert compat.FACTOR not in envs.environment_variables
    if failure != "clean":
        for query in (compat.state, compat.install):
            with pytest.raises(ExtensionCommitError): query()


@pytest.mark.parametrize("phase", ["factor", "runtime-record"])
def test_late_publication_failure_is_sticky_and_not_rolled_back(monkeypatch, provider, phase):
    enable(monkeypatch)
    original = AiterCustomAllreduce.__init__
    if phase == "factor":
        class Reject(dict):
            def __setitem__(self, key, value):
                if key == compat.FACTOR: raise RuntimeError("CPU factor-publication failure")
                return super().__setitem__(key, value)
        monkeypatch.setattr(envs, "environment_variables", Reject(envs.environment_variables))
    else:
        monkeypatch.setattr(adapter_registry, "register", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("CPU runtime-record failure")))
    with pytest.raises(ExtensionCommitError): compat.install()
    assert compat._installed is not None and compat._commit_failure is not None
    assert (AiterCustomAllreduce.__init__ is original) == (phase == "factor")
    assert "invalid" in json.loads(compat.effective_compile_factor())["implementation"]
    for query in (compat.state, compat.install):
        with pytest.raises(ExtensionCommitError): query()


def test_actual_runtime_order_is_admission_then_compat_then_optional_extensions(monkeypatch, provider):
    enable(monkeypatch)
    events = []
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: events.append("platform"))
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: events.append("path"))
    monkeypatch.setattr(bridge, "_require_vllm_hip_runtime_abi", lambda: events.append("runtime-abi"))
    from hcu_adapter.bridges import vllm_fp8
    monkeypatch.setattr(vllm_fp8, "install", lambda: events.append("fp8"))
    # Source checkout has no installed Adapter metadata yet; this contract is
    # runtime ordering with explicit hardware/metadata doubles, not a CLI pass.
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "0.1.0+cpu-metadata-double")
    def extensions(context, *, health_check_sink):
        assert compat.state().selected and compat.state().fused_ops_registered
        assert adapter_registry.get("vllm_allreduce") is not None
        events.append("plugins-after-AR")
        return ()
    monkeypatch.setattr(bridge, "activate_extensions", extensions)
    bridge.hcu_runtime_plugin()
    bridge.hcu_runtime_plugin()
    assert events == ["platform", "path", "runtime-abi", "fp8", "plugins-after-AR"]


@pytest.mark.parametrize("started", [False, True])
def test_pr23_runtime_latches_ar_failure_before_extensions(monkeypatch, provider, started):
    """Real host latch; explicit CPU ABI/FP8 boundary fixtures, no HIP calls."""
    enable(monkeypatch)
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "0.1.0+cpu-metadata-double")
    def fail():
        raise ops_module.CustomAllreduceRegistrationError(
            "CPU registration failure", registration_started=started)
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops", staticmethod(fail))
    monkeypatch.setattr(bridge, "activate_extensions", lambda context, **kwargs: pytest.fail("activation after AR failure"))
    with pytest.raises(bridge.LatchedVllmRuntimeError) as caught:
        bridge.hcu_runtime_plugin()
    assert isinstance(caught.value.__cause__, ExtensionCommitError) is started
    assert (compat._commit_failure is not None) is started
    assert compat._installed is None and compat.FACTOR not in envs.environment_variables
    for query in (bridge.require_runtime_healthy, bridge.hcu_runtime_plugin, compat.state):
        with pytest.raises(bridge.LatchedVllmRuntimeError):
            query()


@pytest.mark.parametrize("started", [False, True])
def test_bad_registration_error_rendering_cannot_heal_runtime(monkeypatch, provider, started):
    """AR consumer contract for the corrected PR23 latch; real action/fork."""
    enable(monkeypatch)
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "0.1.0+cpu-metadata-double")
    calls = []
    class BadError(ops_module.CustomAllreduceRegistrationError):
        def __str__(self):
            raise RuntimeError("CPU broken error rendering")
    def fail():
        calls.append("register")
        raise BadError("CPU registration failure", registration_started=started)
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops", staticmethod(fail))
    monkeypatch.setattr(bridge, "activate_extensions", lambda context, **kwargs: pytest.fail("activation after AR failure"))
    observed = []
    for query in (bridge.hcu_runtime_plugin, bridge.require_runtime_healthy, bridge.hcu_runtime_plugin):
        try:
            query()
        except BaseException as error:
            observed.append(type(error).__name__)  # Never render the injected error.
        else:
            observed.append("healthy")
    receive, send = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(receive)
        signal.alarm(3)
        try:
            bridge.require_runtime_healthy()
        except bridge.LatchedVllmRuntimeError:
            os.write(send, b"refused")
        else:
            os.write(send, b"healthy")
        os._exit(0)
    os.close(send)
    try:
        ready, _, _ = select.select([receive], [], [], 5)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("bounded runtime fork health timed out")
        child = os.read(receive, 64).decode()
    finally:
        os.close(receive)
        os.waitpid(pid, 0)
    assert observed == ["LatchedVllmRuntimeError"] * 3 and calls == ["register"] and child == "refused", (observed, calls, child)


def test_drift_refused_by_both_actual_hash_collectors_after_env_cache(monkeypatch, provider):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.compilation.caching import aot_compile_hash_factors
    enable(monkeypatch)
    compat.install()
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    envs.enable_envs_cache()
    before = envs.compile_factors()[compat.FACTOR]
    old_aot = aot_compile_hash_factors(config)[0]
    monkeypatch.setenv("HCU_DISABLE_OPS", "aiter_ar")
    current = envs.compile_factors()[compat.FACTOR]
    assert current != before and "invalid" in json.loads(current)["implementation"]
    assert aot_compile_hash_factors(config)[0] != old_aot
    with pytest.raises(compat.AllReduceCompatibilityError, match="policy changed"):
        compat.state()


def test_registration_never_changes_official_global_gate_or_pass_manager(monkeypatch, provider):
    enable(monkeypatch)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    before = ops_module.rocm_aiter_ops.__dict__["is_enabled"]
    status = compat.install()
    assert status.fused_ops_registered and status.fusion_pass == "official-global-gate-unchanged"
    assert ops_module.rocm_aiter_ops.__dict__["is_enabled"] is before
    assert os.environ["VLLM_ROCM_USE_AITER"] == "0"
    # Exact target source provenance is checked independently of this textual
    # existing upstream condition; no pass body is rewritten by the Adapter.
    import vllm
    source = Path(vllm.__file__).parent / "compilation/passes/pass_manager.py"
    assert "if rocm_aiter_ops.is_enabled():" in source.read_text()


def test_exact_target_and_declared_phase_origins():
    declared = os.environ["AR_ADAPTER_ROOT"]
    assert Path(compat.__file__).resolve().is_relative_to(Path(declared).resolve())
    target = Path(os.environ["VLLM_TARGET_ROOT"])
    import importlib
    for name in (compat.WRAPPER, "vllm._aiter_ops", "vllm.distributed.device_communicators.cuda_communicator"):
        module = importlib.import_module(name)
        assert Path(module.__file__).read_bytes() == (target / (name.replace(".", "/") + ".py")).read_bytes()


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    assert not torch.cuda.is_initialized()
    envs.disable_envs_cache()
    registry = dict(envs.environment_variables)
    monkeypatch.setattr(adapter_registry, "_RECORDS", {})
    monkeypatch.setattr(adapter_registry, "_BOUND", {})
    init = AiterCustomAllreduce.__init__
    gate = ops_module.rocm_aiter_ops.__dict__["is_custom_all_reduce_enabled"]
    for name in tuple(os.environ):
        if name.startswith("AITER_AR_") or name in (
            "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_CUSTOM_AR", "HCU_AITER_AR", "HCU_DISABLE_OPS"
        ):
            monkeypatch.delenv(name, raising=False)
    for name, value in (("_installed", None), ("_bindings", None), ("_attempted", False),
                        ("_commit_failure", None), ("_fork_error", False), ("_installing", False)):
        monkeypatch.setattr(compat, name, value)
    monkeypatch.setattr(bridge, "detect_hcu", lambda: SimpleNamespace(detected=True))
    # These tests exercise official AR Python/dispatcher behavior, not the
    # independently admitted HIP loader or FP8 registration. The production
    # admission order is covered with the real loader seam in runtime_abi.
    from hcu_adapter.bridges import vllm_fp8
    monkeypatch.setattr(bridge, "_require_vllm_hip_runtime_abi", lambda: None)
    monkeypatch.setattr(vllm_fp8, "install", lambda: False)
    for name in ("_lazy_init", "init", "current_device", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: pytest.fail("CPU contract queried GPU"))
    yield
    envs.disable_envs_cache()
    # Undo test-local replacements before restoring the original class bindings;
    # otherwise monkeypatch can restore a candidate binding after this teardown.
    monkeypatch.undo()
    envs.environment_variables.clear()
    envs.environment_variables.update(registry)
    AiterCustomAllreduce.__init__ = init
    ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled = gate
    assert not torch.cuda.is_initialized()


@pytest.fixture
def provider(monkeypatch):
    calls = []
    state = {"fabric": False, "decline": False}

    class Provider:
        def __init__(self, group, device, max_size, enable_register_for_capturing=True):
            calls.append((group, device, max_size, enable_register_for_capturing))
            self.max_size = max_size
            self.world_size = 2
            self.disabled = False
            self.enable_register_for_capturing = enable_register_for_capturing

        def should_custom_ar(self, x):
            return not state["decline"]

        def custom_all_reduce(self, x):
            calls.append(("ordinary", x))
            return state["ordinary"]

        def custom_fused_ar_rms(self, x, residual, weight, epsilon, **kwargs):
            calls.append(("fused", x, residual, weight, epsilon, kwargs))
            return state["fused"]

    module = ModuleType(compat.PROVIDER)
    module.custom_ar = True
    module.CustomAllreduce = Provider
    monkeypatch.setitem(sys.modules, compat.PROVIDER, module)
    original = compat.import_module

    def load(name):
        if name == "aiter":
            return SimpleNamespace(fabric_ar_available=lambda: state["fabric"])
        return original(name)

    monkeypatch.setattr(compat, "import_module", load)
    monkeypatch.setattr(compat, "_provider_identity", lambda: {"cpu-provider-double": "explicit"})
    return calls, state, module


def enable(monkeypatch):
    monkeypatch.setenv("HCU_AITER_AR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1")


def _cpu_hcu_platform(monkeypatch, architecture="gfx936"):
    """Explicit import/identity metadata double, never a hardware execution claim."""
    import vllm.platforms.rocm as rocm
    monkeypatch.setattr(rocm, "_GCN_ARCH", architecture)
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(
        architecture in bridge.SUPPORTED_HCU_ARCHES, architecture, "CPU identity fixture"))
    return bridge.HcuPlatform()


@pytest.mark.parametrize("case", ["on", "config", "off", "kill", "child-veto", "batch-invariant", "user-disable"])
def test_real_platform_config_worker_and_communicator_chain(monkeypatch, provider, tmp_path, case):
    """Regression: CLI config must reach AITER, not just a direct TP constructor.

    Real ParallelConfig, actual GPU worker init function, real global-enable
    setter and full official CudaCommunicator constructor/dispatch. Only process
    group establishment, PyNccl native allocation and AITER native provider are
    CPU doubles. Actual QuickAllReduce rejects supplied gfx936 metadata.
    """
    from vllm.config import DeviceConfig, ParallelConfig, VllmConfig, set_current_vllm_config
    import vllm.config.parallel as parallel_module
    from vllm.v1.worker import gpu_worker
    from vllm.distributed.device_communicators import cuda_communicator, quick_all_reduce, pynccl, custom_all_reduce
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    enable(monkeypatch)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    for name in ("VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER",
                 "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "VLLM_USE_NCCL_SYMM_MEM"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if case == "batch-invariant" else "0")
    if case == "off": monkeypatch.setenv("HCU_AITER_AR", "0")
    if case == "kill": monkeypatch.setenv("HCU_DISABLE_OPS", "aiter_ar")
    if case == "child-veto": monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "0")
    if case == "config":
        from hcu_adapter import hcu_config, hcu_env
        config_file = tmp_path / "hcu.json"
        config_file.write_text('{"version":1,"ops":{"allreduce":"aiter"}}')
        monkeypatch.setenv("HCU_CONFIG", str(config_file))
        monkeypatch.delenv("HCU_AITER_AR")
        monkeypatch.setattr(hcu_config, "_STATE", dict(loaded=False, path=None, ops={}, overrides={}, hosts={}))
        assert hcu_env.resolve("HCU_AITER_AR", host="vllm") == ("1", "config")
    if case == "batch-invariant":
        with pytest.raises(compat.AllReduceCompatibilityError, match="VLLM_BATCH_INVARIANT"):
            compat.install()
        assert not compat.state().installed
    else:
        compat.install()
    platform = _cpu_hcu_platform(monkeypatch)
    monkeypatch.setattr(parallel_module, "current_platform", platform)
    monkeypatch.setattr(cuda_communicator, "current_platform", platform)
    monkeypatch.setattr(quick_all_reduce, "current_platform", platform)
    parallel = ParallelConfig(tensor_parallel_size=2, distributed_executor_backend="external_launcher",
                              disable_custom_all_reduce=case == "user-disable")
    expected = case in ("on", "config")
    assert parallel.disable_custom_all_reduce is not expected
    config.parallel_config = parallel
    # Restore the actual module-global enable flag after the real worker mutates it.
    monkeypatch.setattr(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", parallel_state._ENABLE_CUSTOM_ALL_REDUCE)
    calls, native_state, _ = provider
    source, selected_output, fallback_output = (torch.tensor([value]) for value in (2., 7., 11.))
    native_state["ordinary"] = selected_output
    boundaries = []

    class Group:
        def rank(self): return 0
        def size(self): return 2

    class CpuPyNccl:
        disabled = False
        world_size = 2

        def __init__(self, group, device):
            boundaries.append("pynccl-native-boundary")

        def all_reduce(self, tensor):
            assert tensor is source
            boundaries.append("pynccl-fallback")
            return fallback_output

    def forbid_native(*args, **kwargs):
        pytest.fail("platform admission reached native C++ CustomAllreduce")

    monkeypatch.setattr(pynccl, "PyNcclCommunicator", CpuPyNccl)
    monkeypatch.setattr(custom_all_reduce, "CustomAllreduce", forbid_native)
    # The normal batch-invariant initializer allocates backend state; isolate
    # that native boundary while retaining its real env gate in communicator.
    from vllm.model_executor.determinism import batch_invariant
    monkeypatch.setattr(batch_invariant, "init_batch_invariance", lambda: boundaries.append("batch-native-boundary"))
    monkeypatch.setattr(gpu_worker, "init_distributed_environment", lambda *a, **kw: boundaries.append("groups-native-boundary"))
    monkeypatch.setattr(gpu_worker, "ensure_ec_transfer_initialized", lambda c: None)
    actual = []
    metadata_calls = []

    def build_group(*args):
        assert parallel_state._ENABLE_CUSTOM_ALL_REDUCE is expected
        # A metadata-only property fixture lets the real QR constructor take
        # its real architecture rejection; no device API/runtime is executed.
        with monkeypatch.context() as metadata:
            def properties(device):
                metadata_calls.append(device)
                return SimpleNamespace(gcnArchName="gfx936")
            metadata.setattr(torch.cuda, "get_device_properties", properties)
            actual.append(CudaCommunicator(Group(), torch.device("cpu"), unique_name="tp:cpu-contract",
                                          global_ranks=[0, 1], global_world_size=2))

    monkeypatch.setattr(gpu_worker, "ensure_model_parallel_initialized", build_group)
    with set_current_vllm_config(config):
        gpu_worker.init_worker_distributed_environment(config, rank=0, backend="gloo")
    comm, = actual
    assert comm.use_aiter_allreduce is expected
    assert comm.ca_comm is None  # Never widen to the rejected native C++ provider.
    assert metadata_calls == ([0] if expected else [])
    assert (comm.qr_comm is not None) is expected
    if comm.qr_comm is not None:
        assert comm.qr_comm.disabled  # Actual QR gfx936 rejection, no private override.
    if expected:
        assert type(comm.aiter_ar_comm) is AiterCustomAllreduce
        assert comm.all_reduce(source) is selected_output
        assert calls[-1] == ("ordinary", source)
    else:
        assert comm.aiter_ar_comm is None and calls == []
        # Batch invariant's separate numerical implementation is out of scope;
        # the constructor above already proves neither custom provider allocates.
        if case != "batch-invariant":
            assert comm.all_reduce(source) is fallback_output
    assert not envs.VLLM_ROCM_USE_AITER and not torch.cuda.is_initialized()


@pytest.mark.parametrize("case", ["default", "unsupported-arch", "missing-abi", "late-policy", "late-batch-invariant", "owner-fatal", "host-fatal"])
def test_platform_extension_requires_healthy_explicit_installed_abi(monkeypatch, provider, case):
    platform = _cpu_hcu_platform(monkeypatch)
    if case == "default":
        assert not platform.use_custom_allreduce()
        assert not compat.state().installed
        return
    enable(monkeypatch)
    if case == "missing-abi":
        monkeypatch.delattr(ops_module.rocm_aiter_ops, "register_custom_allreduce_ops")
        with pytest.raises(compat.AllReduceCompatibilityError, match="public Custom AR"):
            compat.install()
        assert not platform.use_custom_allreduce()
        return
    if case == "unsupported-arch":
        compat.install()
        platform = _cpu_hcu_platform(monkeypatch, "gfx999")
        assert not platform.use_custom_allreduce()
        return
    compat.install()
    if case == "late-policy":
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "0")
        error = compat.AllReduceCompatibilityError
    elif case == "late-batch-invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        with pytest.raises(compat.AllReduceCompatibilityError, match="VLLM_BATCH_INVARIANT"):
            ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled()
        assert "invalid" in json.loads(compat.effective_compile_factor())["implementation"]
        error = compat.AllReduceCompatibilityError
    elif case == "owner-fatal":
        monkeypatch.setattr(compat, "_commit_failure", "CPU sticky fatal fixture")
        error = ExtensionCommitError
    else:
        actions = bridge.ProcessLocalActions()
        monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", actions)
        with pytest.raises(RuntimeError):
            actions.run_once("vllm-runtime", lambda: (_ for _ in ()).throw(RuntimeError("CPU host fatal fixture")))
        error = bridge.LatchedVllmRuntimeError
    with pytest.raises(error):
        platform.use_custom_allreduce()


def test_platform_keeps_official_supported_arch_behavior(monkeypatch):
    platform = _cpu_hcu_platform(monkeypatch, "gfx942")
    assert platform.use_custom_allreduce()  # Actual official super predicate.
    assert not compat.state().installed  # Does not activate Adapter AR.


@pytest.fixture
def identity_records(monkeypatch, tmp_path, provider):
    # Public package/native module records and bytes are CPU doubles. No native
    # file in this fixture is ever dlopened and no provider Python is executed.
    _, state, module = provider
    root = tmp_path / "aiter"
    for name in allreduce_aiter.PYTHON_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU public AR dependency file double\n")
    resource = root.parent / allreduce_aiter.DTYPE_RESOURCES[0]
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("// CPU dtype resource double\n")
    package = ModuleType("aiter")
    package.__file__ = str(root / "__init__.py")
    package.fabric_ar_available = lambda: state["fabric"]
    module.__file__ = str(root / "dist/device_communicators/custom_all_reduce.py")
    module.fp8 = torch.float8_e4m3fn
    module.FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
    native = tmp_path / "module_custom_all_reduce.so"
    native.write_bytes(b"CPU native-file identity double, never dlopened")
    record = ModuleType("aiter.jit.module_custom_all_reduce")
    record.__file__ = str(native)
    monkeypatch.setitem(sys.modules, record.__name__, record)
    previous = compat.import_module
    monkeypatch.setattr(compat, "import_module", lambda name: package if name == "aiter" else previous(name))
    monkeypatch.setattr(compat, "_provider_identity", REAL_PROVIDER_IDENTITY)
    return root, native, module


def test_disabled_vendor_absence_and_neutral_policy(monkeypatch):
    original = compat.import_module

    def load(name):
        assert not name.startswith("aiter"), "disabled owner imported vendor"
        return original(name)

    monkeypatch.setattr(compat, "import_module", load)
    compat.install()
    first = compat.effective_compile_factor()
    monkeypatch.setenv("AITER_AR_TRANSPORT", "invalid-but-disabled")
    monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", "invalid-but-disabled")
    assert compat.effective_compile_factor() == first
    assert not ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled()


@pytest.mark.parametrize("size,explicit,expected", [
    (None, None, 128), ("32", None, 128), ("128", None, 256),
    ("128", 48, 48),
])
def test_actual_official_constructor_copy_in_sizing_and_class_identity(monkeypatch, provider, size, explicit, expected):
    calls, _, _ = provider
    enable(monkeypatch)
    if size:
        monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", size)
    original_methods = {name: getattr(AiterCustomAllreduce, name) for name in (
        "custom_all_reduce", "capture", "close", "use_1stage_fused_ar_rms"
    )}
    compat.install()
    ar = AiterCustomAllreduce("cpu-group-double", "cuda:0", None if explicit is None else explicit * MIB)
    assert type(ar) is AiterCustomAllreduce
    assert calls == [("cpu-group-double", "cuda:0", expected * MIB, False)]
    assert not ar.aiter_ca.enable_register_for_capturing
    assert AiterCustomAllreduce.MAX_SIZE == 128 * MIB
    assert ar.effective_max_size() == 64 * MIB  # Fused cap is independent.
    assert all(getattr(AiterCustomAllreduce, name) is value for name, value in original_methods.items())


@pytest.mark.parametrize("opt_in,caller,expected", [("0", True, False), ("1", True, True), ("1", False, False)])
def test_registration_override_explicit_and_official_false_wins(monkeypatch, provider, opt_in, caller, expected):
    enable(monkeypatch)
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", opt_in)
    compat.install()
    ar = AiterCustomAllreduce(None, "cuda:0", register_graph_buffers=caller)
    assert ar.aiter_ca.enable_register_for_capturing is expected


def test_real_communicator_all_reduce_delegates_no_tensor_copy_or_global_enable(monkeypatch, provider):
    calls, state, _ = provider
    enable(monkeypatch)
    old_other_gate = ops_module.rocm_aiter_ops.__dict__["is_enabled"]
    compat.install()
    assert ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled() is True
    assert ops_module.rocm_aiter_ops.__dict__["is_enabled"] is old_other_gate
    assert "VLLM_ROCM_USE_AITER" not in os.environ
    ar = AiterCustomAllreduce(None, "cuda:0")
    comm = CudaCommunicator.__new__(CudaCommunicator)  # No group/device initialization.
    for field in ("fi_ar_comm", "pynccl_comm", "qr_comm", "fi_pcie_ipc_ar_comm", "ca_comm", "symm_mem_comm"):
        setattr(comm, field, None)
    comm.use_aiter_allreduce = True
    comm.aiter_ar_comm = ar
    state["ordinary"] = output = torch.tensor([37.0])
    source = torch.tensor([2.0])
    assert comm.all_reduce(source) is output
    assert calls[-1][0] == "ordinary" and calls[-1][1] is source
    assert source.item() == 2.0


def test_real_official_fused_path_preserves_output_and_residual(monkeypatch, provider):
    calls, state, _ = provider
    enable(monkeypatch)
    compat.install()
    ar = AiterCustomAllreduce(None, "cuda:0")
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(device_communicator=SimpleNamespace(aiter_ar_comm=ar)))
    source = torch.ones((2, 16), dtype=torch.bfloat16)
    residual = torch.full_like(source, 2)
    weight = torch.ones(16, dtype=source.dtype)
    state["fused"] = outputs = (torch.full_like(source, 3), torch.full_like(source, 4))
    actual = ops_module._rocm_aiter_fused_allreduce_rmsnorm_impl(source, residual, weight, 1e-6)
    assert actual[0] is outputs[0] and actual[1] is outputs[1]
    args = calls[-1]
    assert args[0] == "fused" and args[1] is source and args[2] is residual and args[3] is weight
    assert args[5] == {"use_1stage": True, "gemma_norm": False}
    assert torch.all(source == 1) and torch.all(residual == 2)


def test_real_fused_decline_preserves_official_collective_and_norm_fallback(monkeypatch, provider):
    _, state, _ = provider
    enable(monkeypatch)
    compat.install()
    ar = AiterCustomAllreduce(None, "cuda:0")
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(device_communicator=SimpleNamespace(aiter_ar_comm=ar)))
    import vllm.distributed as distributed
    calls = []
    reduced = torch.full((2, 16), 5, dtype=torch.bfloat16)
    residual = torch.ones_like(reduced)
    outputs = (object(), object())
    def collective(x):
        calls.append(("fallback-collective", x))
        return reduced
    def norm(x, res, weight, epsilon, gemma):
        calls.append(("fallback-norm", x, res))
        return outputs
    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce", collective)
    monkeypatch.setattr(torch.ops.vllm_aiter, "fused_add_rms_norm", norm, raising=False)
    state["fused"] = None
    source = torch.ones_like(reduced)
    actual = ops_module._rocm_aiter_fused_allreduce_rmsnorm_impl(source, residual, source[0], 1e-6)
    assert actual is outputs
    assert calls[0][0] == "fallback-collective" and calls[0][1] is source
    assert calls[1][0] == "fallback-norm" and calls[1][1] is reduced and calls[1][2] is residual


def test_actual_normal_and_aot_factors_include_prepared_policy(monkeypatch, provider):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.utils import hash_factors
    from vllm.compilation.caching import aot_compile_hash_factors
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    ordinary_before = hash_factors(envs.compile_factors())
    aot_before = aot_compile_hash_factors(config)[0]
    assert ordinary_before == aot_before
    enable(monkeypatch)
    compat.install()
    assert envs.compile_factors()[compat.FACTOR] == compat.effective_compile_factor()
    ordinary = hash_factors(envs.compile_factors())
    assert ordinary != ordinary_before
    assert aot_compile_hash_factors(config)[0] == ordinary
    envs.enable_envs_cache()
    assert hash_factors(envs.compile_factors()) == ordinary
    assert aot_compile_hash_factors(config)[0] == ordinary


def test_selected_provider_identity_requires_actual_module_record_and_hashes_bytes(monkeypatch, identity_records):
    root, native, _ = identity_records
    before = compat._provider_identity()
    native.write_bytes(b"different CPU native-file identity double")
    after_native = compat._provider_identity()
    assert before["native"] != after_native["native"]
    python = root / "dist/device_communicators/custom_all_reduce.py"
    python.write_text("# changed CPU provider file double\n")
    assert after_native["python"] != compat._provider_identity()["python"]
    monkeypatch.delitem(sys.modules, "aiter.jit.module_custom_all_reduce")
    with pytest.raises(compat.AllReduceCompatibilityError, match="native custom-AR module"):
        compat._provider_identity()


@pytest.mark.parametrize("name", ["dist/device_communicators/topology.py", "dist/parallel_state.py",
                                 "jit/core.py", "utility/dtypes.py", "csrc/include/aiter_enum.h"])
def test_ar_abi_dependency_or_resource_change_affects_identity(identity_records, name):
    root, _, _ = identity_records
    before = compat._provider_identity()
    path = (root.parent if name in allreduce_aiter.DTYPE_RESOURCES else root) / name
    path.write_bytes(path.read_bytes() + b"# changed CPU dependency/resource double\n")
    assert compat._provider_identity() != before


@pytest.mark.parametrize("name", ["aiter.moe", "aiter.ops.attention"])
def test_unrelated_import_order_and_bytes_neutral_in_identity_and_both_hash_layers(monkeypatch, identity_records, tmp_path, name):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.utils import hash_factors
    from vllm.compilation.caching import aot_compile_hash_factors
    identity = compat._provider_identity()
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("# unrelated CPU module double\n")
    record = ModuleType(name)
    record.__file__ = str(unrelated)
    monkeypatch.setitem(sys.modules, name, record)  # Imported BEFORE activation.
    assert compat._provider_identity() == identity
    unrelated.write_text("# unrelated changed bytes, still irrelevant to AR\n")
    assert compat._provider_identity() == identity
    enable(monkeypatch)
    compat.install()
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    before = hash_factors(envs.compile_factors())
    assert aot_compile_hash_factors(config)[0] == before
    monkeypatch.delitem(sys.modules, name)  # Unrelated import absent instead.
    assert compat._provider_identity() == identity
    assert hash_factors(envs.compile_factors()) == before
    assert aot_compile_hash_factors(config)[0] == before


def test_native_alias_same_bytes_neutral_different_bytes_rejected(monkeypatch, identity_records, tmp_path):
    _, native, _ = identity_records
    before = compat._provider_identity()
    alias = ModuleType("module_custom_all_reduce")
    alias.__file__ = str(native)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    assert compat._provider_identity() == before
    other = tmp_path / "other.so"
    other.write_bytes(b"different CPU native module double")
    alias.__file__ = str(other)
    with pytest.raises(compat.AllReduceCompatibilityError, match="ambiguous"):
        compat._provider_identity()


def test_constructor_failure_blocks_retry_and_preserves_cause(monkeypatch, provider):
    _, _, vendor = provider
    def fail(self, group, device, max_size, enable_register_for_capturing=True):
        raise RuntimeError("injected native allocation failure; CPU double")
    monkeypatch.setattr(vendor.CustomAllreduce, "__init__", fail)
    enable(monkeypatch)
    compat.install()
    with pytest.raises(ExtensionCommitError) as caught:
        AiterCustomAllreduce(None, "cuda:0")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert compat._attempted
    with pytest.raises(ExtensionCommitError, match="construction failed"):
        ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled()


def test_foreign_binding_after_activation_refuses_hash_and_constructor(monkeypatch, provider):
    enable(monkeypatch)
    compat.install()
    monkeypatch.setattr(ops_module.rocm_aiter_ops, "is_custom_all_reduce_enabled", lambda: True)
    with pytest.raises(compat.AllReduceCompatibilityError, match="bindings changed"):
        compat.state()
    assert "invalid" in json.loads(compat.effective_compile_factor())["implementation"]
    with pytest.raises(compat.AllReduceCompatibilityError, match="bindings changed"):
        AiterCustomAllreduce(None, "cuda:0")


@pytest.mark.parametrize("variable,value", [("AITER_AR_MAX_SIZE_MB", "128"), ("AITER_AR_TRANSPORT", "auto"),
                                          ("VLLM_ROCM_USE_AITER_CUSTOM_AR", "0")])
def test_policy_change_after_cache_refused_by_factor_selector_and_constructor(monkeypatch, provider, variable, value):
    enable(monkeypatch)
    compat.install()
    before = compat.effective_compile_factor()
    envs.enable_envs_cache()
    assert getattr(envs, compat.FACTOR) == before
    monkeypatch.setenv(variable, value)
    for call in (compat.state, ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled,
                 lambda: AiterCustomAllreduce(None, "cuda:0"), lambda: compat.install()):
        with pytest.raises(compat.AllReduceCompatibilityError, match="policy changed"):
            call()


def test_official_public_dispatcher_for_gemma_flag(monkeypatch, provider):
    _, state, _ = provider
    enable(monkeypatch)
    compat.install()
    ar = AiterCustomAllreduce(None, "cuda:0")
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(device_communicator=SimpleNamespace(aiter_ar_comm=ar)))
    source = torch.ones((2, 16), dtype=torch.bfloat16)
    residual = torch.ones_like(source)
    state["fused"] = expected = (torch.full_like(source, 7), torch.full_like(source, 9))
    # The real target dispatcher has CUDA and Meta, not a CPU numerical kernel.
    # FakeTensor exercises its real public handle/shape contract without
    # installing a CPU override or treating a provider double as native math.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        x, r = (mode.from_tensor(t) for t in (source, residual))
        w = x[0]
        result = ops_module.rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(x, r, w, 1e-6, True)
        assert result[0].shape == source.shape and result[1].shape == residual.shape
        assert result[0].dtype == source.dtype and result[1].dtype == residual.dtype
    assert torch.all(source == 1) and torch.all(residual == 1)


@pytest.mark.parametrize("constructed,expected", [(False, "ok"), (True, "refused")])
def test_actual_fork_declarations_safe_but_native_construction_inheritance_refused(monkeypatch, provider, constructed, expected):
    enable(monkeypatch)
    compat.install()
    if constructed:
        AiterCustomAllreduce(None, "cuda:0")  # CPU provider double, conservatively owned.
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            compat.state()
            result = "ok"
        except compat.AllReduceCompatibilityError:
            result = "refused"
        os.write(writer, result.encode())
        os._exit(0)
    os.close(writer)
    try:
        ready, _, _ = select.select([reader], [], [], 5)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("fork child deadlocked")
        assert os.read(reader, 100).decode() == expected
    finally:
        os.close(reader)
        os.waitpid(pid, 0)
