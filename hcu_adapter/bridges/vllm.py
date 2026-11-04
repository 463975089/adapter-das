"""Official vLLM platform/runtime seams and shared custom-op glue."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionReport,
    ExtensionCommitError,
    RequiredExtensionError,
    activate_extensions,
)
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle


SUPPORTED_VLLM_RELEASE_PREFIXES = frozenset({(0, 32), (0, 33)})
SUPPORTED_HCU_ARCHES = frozenset({"gfx928", "gfx92a", "gfx936", "gfx938"})
HYGON_PCI_VENDOR_ID = "0x1d94"
_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_PLATFORM_CLASS = "hcu_adapter.bridges.vllm.HcuPlatform"
_FLASH_ATTN_CAPABILITY = "hcu-flash-attention"
_TOOLCHAIN_ROOTS = ("DTKROOT", "ROCM_PATH", "HIP_PATH")
_VLLM_RUNTIME_LIBRARY = "VLLM_CUDART_SO_PATH"
_REQUIRED_EXTENSIONS = "HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS"
_REQUIRED_CAPABILITIES = "HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES"


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
    """Accept supported release lines; runtime ABI admission remains separate."""
    if raw_version is None:
        return False
    try:
        parsed = Version(raw_version)
    except (InvalidVersion, TypeError):
        return False
    return (
        parsed.epoch == 0
        and parsed.release[:2] in SUPPORTED_VLLM_RELEASE_PREFIXES
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
        expected = ", ".join(
            ".".join(map(str, prefix)) + ".x"
            for prefix in sorted(SUPPORTED_VLLM_RELEASE_PREFIXES)
        )
        raise VllmCompatibilityError(
            f"unsupported vLLM release {version!r}; expected release line "
            f"{expected}; required runtime APIs must also be available"
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
    """Detect a supported HCU identity lazily; provider admission is separate."""
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


def _require_vllm_hip_runtime_abi() -> None:
    """Admit the runtime through vLLM's own loader, without executing HIP ops.

    Its constructor resolves the actual loaded library before the configured
    fallback and binds the required symbols/types. Preserve that precedence
    and the operator's path spelling; do not duplicate its native ABI table.
    """
    try:
        wrapper = import_module(
            "vllm.distributed.device_communicators.cuda_wrapper"
        )
        wrapper.CudaRTLibrary()
    except Exception as exc:
        raise VllmCompatibilityError(
            "vLLM HIP runtime ABI admission failed before operator registration: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class ProcessLocalActions(ProcessLocalLifecycle):
    """Supply only vLLM action identity and its existing public error type."""

    def __init__(self) -> None:
        super().__init__(runtime_action="vllm-runtime", error_type=LatchedVllmRuntimeError)

_RUNTIME_ACTIONS = ProcessLocalActions()


def extension_reports() -> tuple[ExtensionReport, ...]:
    """Return the latest process-local optimization activation reports."""
    return _RUNTIME_ACTIONS.extension_reports()


def _require_flash_capability() -> None:
    # Reports are process-local and cleared after fork. Do not activate or
    # repair extensions from a backend selector, or trust a parent report.
    require_extension_healthy(_FLASH_ATTN_CAPABILITY)


def require_runtime_healthy() -> None:
    """Refuse failed boot without activation, repair or backend selection.

    Unknown/healthy state is allowed; this is not evidence activation completed.
    Reports remain available after failure for diagnosis.
    """
    _RUNTIME_ACTIONS.require_runtime_healthy()
    from hcu_adapter.bridges.vllm_mhc import require_healthy

    require_healthy()


def require_extension_healthy(capability: str) -> None:
    """Check a unique current-process ACTIVE owner without activation/repair."""
    _RUNTIME_ACTIONS.require_extension_healthy(capability)


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
            require_runtime_healthy()
            # Upstream owns config validation/defaults, including worker choice.
            super().check_and_update_config(vllm_config)
            from hcu_adapter.bridges.vllm_model_load import (
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
            require_runtime_healthy()
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
                _require_flash_capability()
                # The owner publishes via vLLM's public backend registry.
                # Official ROCm selection owns class lookup and validation.
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
    # A healthy fork must first activate/adopt its inherited mHC registration;
    # the full read-only owner check runs after this process's runtime action.
    _RUNTIME_ACTIONS.require_runtime_healthy()
    if not detect_hcu().detected:
        return None

    def initialize() -> None:
        # Parse before any operator registration. These are startup requirements,
        # inherited by spawned/forked workers, not runtime enable/disable knobs.
        required_extensions = _required_names(_REQUIRED_EXTENSIONS)
        required_capabilities = _required_names(_REQUIRED_CAPABILITIES)
        vllm_version = _ensure_vllm_compatible()
        _ensure_hcu_platform_selected()
        configure_vllm_hip_runtime_path()
        _require_vllm_hip_runtime_abi()
        from hcu_adapter.bridges.vllm_fp8 import install as install_fp8_fallback

        install_fp8_fallback()
        from hcu_adapter.bridges.vllm_mhc import install as install_mhc_fallback

        install_mhc_fallback()
        context = ExtensionContext(
            adapter_version=_installed_adapter_version(),
            vllm_version=vllm_version,
            requested_extensions=required_extensions,
            requested_capabilities=required_capabilities,
        )
        try:
            reports = activate_extensions(
                context, health_check_sink=_RUNTIME_ACTIONS.record_extension_health,
            )
        except (RequiredExtensionError, ExtensionCommitError) as exc:
            _RUNTIME_ACTIONS.record_extension_reports(exc.reports)
            raise
        _RUNTIME_ACTIONS.record_extension_reports(reports)

    _RUNTIME_ACTIONS.run_once("vllm-runtime", initialize)
    require_runtime_healthy()


def _required_names(variable: str) -> frozenset[str]:
    """Read a comma-separated list; reject malformed nonempty requirements."""
    from hcu_adapter import hcu_env

    raw, _source = hcu_env.resolve(variable)
    if not raw.strip():
        return frozenset()
    names = tuple(name.strip() for name in raw.split(","))
    if any(not name or any(char.isspace() for char in name) for name in names):
        raise ValueError(f"{variable} requires comma-separated non-empty names")
    return frozenset(names)


def install_rms_norm_bridge() -> bool:
    """Install the compatibility bindings consumed by the SGLang bridge.

    The SGLang startup chain calls this helper; the vLLM runtime plugin does not.
    Its six-argument fused shim has separate output destinations, unlike vLLM
    0.32's four-argument, in-place fused ABI. Do not install it in the vLLM
    runtime to replace the official RMS operators.
    """
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
