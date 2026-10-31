"""Official vLLM v0.32 platform/runtime seams and shared custom-op glue."""

from __future__ import annotations

import os
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from threading import RLock

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionReport,
    RequiredExtensionError,
    activate_extensions,
)


SUPPORTED_VLLM_RELEASE_PREFIX = (0, 32)
SUPPORTED_HCU_ARCHES = frozenset({"gfx928", "gfx936", "gfx938"})
HYGON_PCI_VENDOR_ID = "0x1d94"
_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_PLATFORM_CLASS = "hcu_adapter.bridges.vllm.HcuPlatform"
_FLASH_ATTN_BACKEND_CLASS = (
    "hcu_vllm_das.attention.flash_attn.HcuFlashAttentionBackend"
)
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
    """Accept prerelease, patch, post, and local builds on vLLM 0.32.x."""
    if raw_version is None:
        return False
    try:
        parsed = Version(raw_version)
    except (InvalidVersion, TypeError):
        return False
    return (
        parsed.epoch == 0
        and parsed.release[:2] == SUPPORTED_VLLM_RELEASE_PREFIX
    )


def _installed_vllm_version() -> str | None:
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None


def _installed_adapter_version() -> str:
    try:
        return metadata.version("sgl-kernel")
    except metadata.PackageNotFoundError as exc:
        raise VllmCompatibilityError(
            "adapter distribution metadata is unavailable: 'sgl-kernel'"
        ) from exc


def _ensure_vllm_compatible() -> str:
    version = _installed_vllm_version()
    if not is_supported_vllm_version(version):
        expected = ".".join(map(str, SUPPORTED_VLLM_RELEASE_PREFIX)) + ".x"
        raise VllmCompatibilityError(
            f"unsupported vLLM release {version!r}; expected release line "
            f"{expected} (>=0.32.0,<0.33.0)"
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
        self._extension_reports: tuple[ExtensionReport, ...] = ()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _reset(self, owner_pid: int) -> None:
        self._owner_pid = owner_pid
        self._lock = RLock()
        self._records = {}
        self._extension_reports = ()

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

    def record_extension_reports(
        self,
        reports: tuple[ExtensionReport, ...],
    ) -> None:
        self._ensure_process_local()
        with self._lock:
            self._extension_reports = tuple(reports)

    def extension_reports(self) -> tuple[ExtensionReport, ...]:
        self._ensure_process_local()
        with self._lock:
            return self._extension_reports


_RUNTIME_ACTIONS = ProcessLocalActions()


def extension_reports() -> tuple[ExtensionReport, ...]:
    """Return the latest process-local optimization activation reports."""
    return _RUNTIME_ACTIONS.extension_reports()


def install() -> None:
    """Defer vLLM activation to its official plugin discovery lifecycle."""


def _hcu_platform_class() -> type:
    """Build the official ROCm-derived platform only when vLLM requests it."""
    existing = globals().get("HcuPlatform")
    if existing is not None:
        return existing

    from vllm.platforms.rocm import RocmPlatform

    class _HcuPlatform(RocmPlatform):
        """Reuse official ROCm behavior under an HCU platform identity."""

        device_name = "hcu"

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            # Upstream owns config validation/defaults, including worker choice.
            super().check_and_update_config(vllm_config)
            from hcu_adapter.model_load import (
                OFFICIAL_MODEL_LOAD_WORKER,
                POST_MODEL_LOAD_WORKER,
            )

            parallel = vllm_config.parallel_config
            if (
                parallel.worker_cls == OFFICIAL_MODEL_LOAD_WORKER
                and parallel.distributed_executor_backend in ("uni", "mp")
            ):
                parallel.worker_cls = POST_MODEL_LOAD_WORKER

        @classmethod
        def get_attn_backend_cls(
            cls,
            selected_backend,
            attn_selector_config,
            num_heads=None,
        ) -> str:
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
            )

            # Keep the validated ordinary-attention default, but let the
            # official ROCm priority table choose specialized MLA backends.
            # Forcing TRITON_ATTN here makes both MLA and sparse MLA fail
            # capability validation before model weights are loaded.
            uses_mla = bool(getattr(attn_selector_config, "use_mla", False))
            uses_sparse = bool(
                getattr(attn_selector_config, "use_sparse", False)
            )
            if selected_backend is None and not (uses_mla or uses_sparse):
                selected_backend = AttentionBackendEnum.TRITON_ATTN
            if selected_backend == AttentionBackendEnum.FLASH_ATTN:
                module_name, class_name = _FLASH_ATTN_BACKEND_CLASS.rsplit(".", 1)
                backend_class = getattr(import_module(module_name), class_name)
                device_capability = cls.get_device_capability()
                if device_capability is None:
                    raise VllmCompatibilityError(
                        "HCU FLASH_ATTN requires a detected device capability"
                    )
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
                if invalid_reasons:
                    raise ValueError(
                        "Selected HCU FLASH_ATTN backend is not valid for this "
                        f"configuration. Reason: {invalid_reasons}"
                    )
                return _FLASH_ATTN_BACKEND_CLASS
            return super().get_attn_backend_cls(
                selected_backend,
                attn_selector_config,
                num_heads,
            )

    _HcuPlatform.__name__ = "HcuPlatform"
    _HcuPlatform.__qualname__ = "HcuPlatform"
    globals()["HcuPlatform"] = _HcuPlatform
    return _HcuPlatform


def __getattr__(name: str):
    """Expose HcuPlatform lazily without importing vLLM at adapter startup."""
    if name == "HcuPlatform":
        return _hcu_platform_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def hcu_platform_plugin() -> str | None:
    """Return the lazy HCU platform class only on a validated HCU host."""
    if not detect_hcu().detected:
        return None
    _RUNTIME_ACTIONS.run_once("vllm-platform", _ensure_vllm_compatible)
    return _PLATFORM_CLASS


def _ensure_hcu_platform_selected() -> None:
    platforms = import_module("vllm.platforms")
    selected = getattr(platforms, "current_platform")
    expected_class = _hcu_platform_class()
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
        vllm_version = _ensure_vllm_compatible()
        _ensure_hcu_platform_selected()
        configure_vllm_hip_runtime_path()
        context = ExtensionContext(
            adapter_version=_installed_adapter_version(),
            vllm_version=vllm_version,
        )
        try:
            reports = activate_extensions(context)
        except RequiredExtensionError as exc:
            _RUNTIME_ACTIONS.record_extension_reports(exc.reports)
            raise
        _RUNTIME_ACTIONS.record_extension_reports(reports)

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
