"""Official vLLM v0.32 platform/runtime seams and shared custom-op glue."""

from __future__ import annotations

import os
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from threading import RLock

from packaging.version import InvalidVersion, Version


TARGET_VLLM_RELEASE = (0, 32, 0)
SUPPORTED_HCU_ARCHES = frozenset({"gfx928", "gfx936", "gfx938"})
HYGON_PCI_VENDOR_ID = "0x1d94"
_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_PLATFORM_CLASS = "hcu_adapter.bridges.vllm_platform.HcuPlatform"
_TOOLCHAIN_ROOTS = ("DTKROOT", "ROCM_PATH", "HIP_PATH")
_VLLM_RUNTIME_LIBRARY = "VLLM_CUDART_SO_PATH"


class VllmCompatibilityError(RuntimeError):
    """Raised before an unsupported vLLM release can select HCU."""


class LatchedVllmRuntimeError(RuntimeError):
    """Raised when a required process-local action already failed."""


@dataclass(frozen=True)
class HcuDetection:
    detected: bool
    architecture: str | None
    reason: str


def is_supported_vllm_version(raw_version: str | None) -> bool:
    """Accept build suffixes on the exact vLLM 0.32.0 release line."""
    if raw_version is None:
        return False
    try:
        parsed = Version(raw_version)
    except (InvalidVersion, TypeError):
        return False
    return parsed.epoch == 0 and parsed.release == TARGET_VLLM_RELEASE


def _installed_vllm_version() -> str | None:
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None


def _ensure_vllm_compatible() -> str:
    version = _installed_vllm_version()
    if not is_supported_vllm_version(version):
        expected = ".".join(map(str, TARGET_VLLM_RELEASE))
        raise VllmCompatibilityError(
            f"unsupported vLLM release {version!r}; expected release {expected}"
        )
    return version


def _has_hygon_pci_device(
    pci_devices_root: Path = _PCI_DEVICES_ROOT,
) -> bool:
    """Check vendor identity without initializing an accelerator runtime."""
    try:
        vendor_files = pci_devices_root.glob("*/vendor")
        for vendor_file in vendor_files:
            try:
                vendor_id = vendor_file.read_text(encoding="ascii").strip().casefold()
            except OSError:
                continue
            if vendor_id == HYGON_PCI_VENDOR_ID:
                return True
    except OSError:
        return False
    return False


def detect_hcu(torch_module: object | None = None) -> HcuDetection:
    """Detect a validated HCU lazily and fail safely on other HIP hosts."""
    if torch_module is None:
        if not _has_hygon_pci_device():
            return HcuDetection(
                False,
                None,
                "no Hygon PCI device evidence was found in sysfs",
            )
        try:
            import torch as torch_module
        except Exception as exc:
            return HcuDetection(
                False,
                None,
                f"unable to import Torch: {type(exc).__name__}: {exc}",
            )

    try:
        hip_version = torch_module.version.hip  # type: ignore[union-attr]
    except Exception as exc:
        return HcuDetection(
            False,
            None,
            f"unable to inspect Torch HIP support: {type(exc).__name__}: {exc}",
        )
    if hip_version is None:
        return HcuDetection(False, None, "Torch was not built with HIP support")

    try:
        accelerator = torch_module.cuda  # type: ignore[union-attr]
        if not accelerator.is_available():
            return HcuDetection(False, None, "no HIP accelerator is available")
        properties = accelerator.get_device_properties(0)
        architecture = str(properties.gcnArchName).strip().split(":", 1)[0]
    except Exception as exc:
        return HcuDetection(
            False,
            None,
            f"unable to read HCU device property: {type(exc).__name__}: {exc}",
        )

    if architecture not in SUPPORTED_HCU_ARCHES:
        return HcuDetection(
            False,
            architecture,
            f"unsupported HCU architecture: {architecture}",
        )
    return HcuDetection(True, architecture, "")


def configure_vllm_hip_runtime_path(
    environment: MutableMapping[str, str] | None = None,
) -> str | None:
    """Point vLLM's runtime wrapper at an existing DTK HIP library."""
    target = os.environ if environment is None else environment
    explicit = target.get(_VLLM_RUNTIME_LIBRARY)
    if explicit:
        return explicit

    visited = set()
    for variable in _TOOLCHAIN_ROOTS:
        raw_root = target.get(variable)
        if not raw_root:
            continue
        root = Path(raw_root)
        if not root.is_absolute():
            continue
        try:
            canonical_root = root.resolve(strict=True)
        except OSError:
            continue
        if canonical_root in visited:
            continue
        visited.add(canonical_root)
        try:
            resolved = (canonical_root / "lib" / "libamdhip64.so").resolve(
                strict=True
            )
            resolved.relative_to(canonical_root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file():
            continue
        value = str(resolved)
        target[_VLLM_RUNTIME_LIBRARY] = value
        return value
    return None


class ProcessLocalActions:
    """Run required callbacks once per PID and latch partial failures."""

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._lock = RLock()
        self._records: dict[str, str | None] = {}
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _reset(self, owner_pid: int) -> None:
        self._owner_pid = owner_pid
        self._lock = RLock()
        self._records = {}

    def _after_fork_child(self) -> None:
        self._reset(os.getpid())

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if pid != self._owner_pid:
            self._reset(pid)

    def run_once(self, action_id: str, callback: Callable[[], object]) -> None:
        self._ensure_process_local()
        with self._lock:
            if action_id in self._records:
                failure = self._records[action_id]
                if failure is not None:
                    raise LatchedVllmRuntimeError(failure)
                return
            try:
                callback()
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                self._records[action_id] = failure
                raise LatchedVllmRuntimeError(failure) from exc
            self._records[action_id] = None


_RUNTIME_ACTIONS = ProcessLocalActions()


def install() -> None:
    """Defer vLLM activation to its official plugin discovery lifecycle."""


def hcu_platform_plugin() -> str | None:
    """Return the lazy HCU platform class only on a validated HCU host."""
    if not detect_hcu().detected:
        return None
    _RUNTIME_ACTIONS.run_once("vllm-platform", _ensure_vllm_compatible)
    return _PLATFORM_CLASS


def _ensure_hcu_platform_selected() -> None:
    platforms = import_module("vllm.platforms")
    selected = getattr(platforms, "current_platform")
    platform_module = import_module("hcu_adapter.bridges.vllm_platform")
    expected_class = platform_module.HcuPlatform
    selected_class = selected if isinstance(selected, type) else type(selected)
    if selected_class is not expected_class:
        selected_name = f"{selected_class.__module__}.{selected_class.__qualname__}"
        raise RuntimeError(
            f"vLLM selected platform {selected_name!r}; expected {_PLATFORM_CLASS!r}"
        )


def hcu_runtime_plugin() -> None:
    """Apply the required vLLM runtime guards once in each HCU process."""
    if not detect_hcu().detected:
        return None

    def initialize() -> None:
        _ensure_vllm_compatible()
        _ensure_hcu_platform_selected()
        configure_vllm_hip_runtime_path()

    _RUNTIME_ACTIONS.run_once("vllm-runtime", initialize)


def install_rms_norm_bridge() -> bool:
    """Bind vLLM's HCU RMSNorm ABI to the adapter's equivalent kernels."""
    try:
        import vllm._custom_ops as custom_ops

        from sgl_kernel import rmsnorm
    except Exception:
        return False

    def rms_norm(out, input, weight, epsilon):
        rmsnorm(input, weight, epsilon, out=out)

    def fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon):
        residual_destination = (
            residual if residual_out is residual else residual_out
        )
        residual_destination.copy_(residual)
        residual_destination.add_(input)
        rmsnorm(residual_destination, weight, epsilon, out=out)

    custom_ops.rms_norm = rms_norm
    custom_ops.fused_add_rms_norm = fused_add_rms_norm
    return True
