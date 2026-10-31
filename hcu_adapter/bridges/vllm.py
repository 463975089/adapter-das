"""Official vLLM v0.32 platform/runtime seams and shared custom-op glue."""

from __future__ import annotations

import os
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from threading import RLock, local

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionReport,
    ExtensionStatus,
    ExtensionCommitError,
    RequiredExtensionError,
    activate_extensions,
)


SUPPORTED_VLLM_RELEASE_PREFIX = (0, 32)
SUPPORTED_HCU_ARCHES = frozenset({"gfx928", "gfx936", "gfx938"})
HYGON_PCI_VENDOR_ID = "0x1d94"
_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_PLATFORM_CLASS = "hcu_adapter.bridges.vllm.HcuPlatform"
_FLASH_ATTN_CAPABILITY = "hcu-flash-attention"
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
        self._running: set[str] = set()
        self._extension_reports: tuple[ExtensionReport, ...] = ()
        self._health_bindings: dict[str, tuple[ExtensionReport, Callable[[], None], int]] = {}
        self._report_generation = 0
        self._health_inflight = 0
        self._health_context = local()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _reset(self, owner_pid: int) -> None:
        # Only the existing vLLM runtime action inherits failure. Its registry
        # mutations also survive fork. Other actions keep their old reset.
        failure = self._records.get("vllm-runtime")
        if failure is None and "vllm-runtime" in self._running:
            failure = "inherited incomplete vllm-runtime activation; spawn a fresh process"
        if failure is None and self._health_inflight:
            failure = "inherited incomplete extension health check; spawn a fresh process"
        self._owner_pid = owner_pid
        self._lock = RLock()
        self._records = {} if failure is None else {"vllm-runtime": failure}
        self._running = set()
        self._extension_reports = ()
        self._health_bindings = {}
        self._report_generation = 0
        self._health_inflight = 0
        self._health_context = local()

    def _after_fork_child(self) -> None:
        self._reset(os.getpid())

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if pid != self._owner_pid:
            self._reset(pid)

    def run_once(self, action_id: str, callback: Callable[[], object]) -> None:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            if action_id in self._records:
                failure = self._records[action_id]
                if failure is not None:
                    raise LatchedVllmRuntimeError(failure)
                return
            if action_id == "vllm-runtime" and action_id in self._running:
                raise LatchedVllmRuntimeError("recursive vllm-runtime activation")
            self._running.add(action_id)
            try:
                callback()
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                self._records[action_id] = failure
                raise LatchedVllmRuntimeError(failure) from exc
            finally:
                self._running.discard(action_id)
            if self._records.get(action_id) is not None:
                raise LatchedVllmRuntimeError(self._records[action_id])
            self._records[action_id] = None

    def require_runtime_healthy(self) -> None:
        """Read failure only; safe/reentrant inside the runtime callback."""
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            failure = self._records.get("vllm-runtime")
            if failure is not None:
                raise LatchedVllmRuntimeError(failure)

    def record_extension_reports(
        self,
        reports: tuple[ExtensionReport, ...],
    ) -> None:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            self._extension_reports = tuple(reports)
            self._report_generation += 1
            # A value-equal replacement report must not borrow a prior checker.
            # Discard stale bindings permanently; no later report can resurrect it.
            self._health_bindings = {
                name: binding for name, binding in self._health_bindings.items()
                if binding[2] == self._owner_pid
                and sum(report.entry_point_name == name for report in reports) == 1
                and any(report is binding[0] and report.status is ExtensionStatus.ACTIVE
                        for report in reports)
            }

    def _reject_health_reentry(self) -> None:
        # Never acquire a host lock from an owner checker. The outer call will
        # latch after the owner lock has unwound, even if the owner caught this.
        if getattr(self._health_context, "checking", False):
            self._health_context.violation = True
            raise ExtensionCommitError("health checker must not re-enter host lifecycle")

    def record_extension_health(self, report: ExtensionReport, checker: Callable[[], None]) -> None:
        """Activation sink: bind exact report/checker/PID, not backend names."""
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            if report.status is not ExtensionStatus.ACTIVE or not callable(checker):
                raise ExtensionCommitError("invalid ACTIVE extension health binding")
            if report.entry_point_name in self._health_bindings:
                raise ExtensionCommitError("duplicate extension health binding")
            self._health_bindings[report.entry_point_name] = (report, checker, self._owner_pid)

    def require_extension_healthy(self, capability: str) -> None:
        """Snapshot under host lock, call owner unlocked, verify snapshot again.

        Lock order is host-only, owner-only, host-only. The checker must not call
        host lifecycle APIs; reentry fails without taking the host lock. This
        does not make arbitrary concurrent third-party registry writes atomic.
        """
        self.require_runtime_healthy()
        inflight = False
        owner_pid = os.getpid()
        try:
            with self._lock:
                # Cover validation, owner callback, snapshot verification AND
                # failure publication. Fork must see inflight or a latched
                # failure until the complete transaction has finished.
                self._health_inflight += 1
                inflight = True
                if self._running:
                    raise ExtensionCommitError("extension health requires completed host activation")
                owners = [report for report in self._extension_reports
                          if capability in report.capabilities]
                if len(owners) != 1 or owners[0].status is not ExtensionStatus.ACTIVE:
                    raise ExtensionCommitError(
                        f"capability {capability!r} requires exactly one ACTIVE extension owner")
                report = owners[0]
                binding = self._health_bindings.get(report.entry_point_name)
                if binding is None or binding[0] is not report or binding[2] != os.getpid():
                    raise ExtensionCommitError(f"capability {capability!r} has no exact current-process health binding")
                generation = self._report_generation
            self._health_context.checking = True
            self._health_context.violation = False
            try:
                result = binding[1]()
                if result is not None:
                    # Close an accidentally returned native coroutine without
                    # running it, avoiding an unawaited-coroutine resource leak.
                    import inspect
                    if inspect.iscoroutine(result):
                        result.close()
                    raise ExtensionCommitError("health checker must return None synchronously")
                if self._health_context.violation:
                    raise ExtensionCommitError("health checker caught forbidden host lifecycle reentry")
            finally:
                self._health_context.checking = False
            with self._lock:
                if (generation != self._report_generation
                        or self._health_bindings.get(report.entry_point_name) is not binding):
                    raise ExtensionCommitError("extension reports/health binding changed during health check")
                self.require_runtime_healthy()
        except BaseException as exc:
            # Owner locks have unwound before the host latch is acquired.
            published = False
            with self._lock:
                failure = self._records.get("vllm-runtime")
                if failure is None:
                    # Publish refusal BEFORE arbitrary exception __str__ code.
                    # Formatting can raise, re-enter, block or fork. A child
                    # must inherit a failure even at that exact boundary.
                    failure = "extension health failed; exception details unavailable"
                    self._records["vllm-runtime"] = failure
                    published = True
            if published:
                try:
                    detail = f"extension health failed: {type(exc).__name__}: {exc}"
                except BaseException:
                    detail = failure
                with self._lock:
                    self._records["vllm-runtime"] = detail
                    failure = detail
            raise LatchedVllmRuntimeError(failure) from exc
        finally:
            # On failure, the except block has already published the host
            # latch. Decrementing earlier creates a fork-to-healthy window.
            if inflight and os.getpid() == owner_pid:
                with self._lock:
                    self._health_inflight -= 1

    def extension_reports(self) -> tuple[ExtensionReport, ...]:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            return self._extension_reports


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
    require_runtime_healthy()
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
            reports = activate_extensions(
                context, health_check_sink=_RUNTIME_ACTIONS.record_extension_health,
            )
        except (RequiredExtensionError, ExtensionCommitError) as exc:
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
