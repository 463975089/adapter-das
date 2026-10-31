"""Versioned protocol for optional vLLM optimization extensions.

This module intentionally imports no vLLM, Torch, or optimization package.
Installed entry points are the only discovery boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
import builtins
import inspect
from typing import Protocol, runtime_checkable


EXTENSION_API_VERSION = 1
EXTENSION_ENTRY_POINT_GROUP = "hcu_adapter.extensions"


@dataclass(frozen=True)
class ExtensionContext:
    adapter_version: str
    vllm_version: str | None
    requested_extensions: frozenset[str] = frozenset()
    requested_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_extensions",
            frozenset(self.requested_extensions),
        )
        object.__setattr__(
            self,
            "requested_capabilities",
            frozenset(self.requested_capabilities),
        )


@dataclass(frozen=True)
class ExtensionSpec:
    name: str
    api_version: int
    priority: int
    capabilities: frozenset[str]
    activate: Callable[[ExtensionContext], None]
    check_health: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


class ExtensionStatus(str, Enum):
    ACTIVE = "active"
    FAILED = "failed"


@dataclass(frozen=True)
class ExtensionReport:
    entry_point_name: str
    extension_name: str | None
    status: ExtensionStatus
    capabilities: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


class RequiredExtensionError(RuntimeError):
    """Raised when an explicitly requested extension cannot activate."""

    def __init__(
        self,
        message: str,
        reports: Iterable[ExtensionReport] = (),
    ) -> None:
        super().__init__(message)
        self.reports = tuple(reports)


class ExtensionCommitError(RuntimeError):
    """Activation published state and cannot safely continue in this process.

    This signals failure, not rollback. The host supplies reports when it
    propagates the error; an owner need only provide the reason.
    """

    def __init__(self, message: str, reports: Iterable[ExtensionReport] = ()) -> None:
        super().__init__(message)
        self.reports = tuple(reports)


def is_fatal_extension_error(error: BaseException) -> bool:
    """Recognize the public signal through wrappers, suppression and cycles.

    No message/type heuristics infer dirty state. Owners explicitly signal
    their commit boundary. Feature detection keeps the Python floor.
    """
    group_type = getattr(builtins, "BaseExceptionGroup", ())
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, ExtensionCommitError):
            return True
        pending.extend((current.__cause__, current.__context__))
        if group_type and isinstance(current, group_type):
            pending.extend(current.exceptions)
    return False


@runtime_checkable
class EntryPointLike(Protocol):
    name: str

    def load(self) -> object: ...


def _entry_points() -> tuple[EntryPointLike, ...]:
    return tuple(metadata.entry_points(group=EXTENSION_ENTRY_POINT_GROUP))


def _exception_description(error: BaseException) -> str:
    """Diagnostics cannot replace the original failure or its classification."""
    try:
        return f"{type(error).__name__}: {error}"
    except BaseException:
        return "exception details unavailable"


def _failure(
    entry_point_name: str,
    reason: str,
    extension_name: str | None = None,
    capabilities: frozenset[str] = frozenset(),
) -> ExtensionReport:
    return ExtensionReport(
        entry_point_name=entry_point_name,
        extension_name=extension_name,
        status=ExtensionStatus.FAILED,
        capabilities=capabilities,
        reason=reason,
    )


def _ordered_reports(
    reports: Iterable[ExtensionReport],
) -> tuple[ExtensionReport, ...]:
    return tuple(
        sorted(
            reports,
            key=lambda report: (report.entry_point_name, report.reason),
        )
    )


def activate_extensions(
    context: ExtensionContext,
    entry_points: Iterable[EntryPointLike] | None = None,
    *,
    health_check_sink: Callable[[ExtensionReport, Callable[[], None]], None] | None = None,
) -> tuple[ExtensionReport, ...]:
    """Activate deterministically; bind optional read-only synchronous checkers.

    A sink receives the exact ACTIVE report returned to the caller. It runs only
    after activation succeeded, so its failure is commit-fatal, never optional.
    Report serialization and API1's existing positional fields stay unchanged.
    """
    if health_check_sink is not None and not callable(health_check_sink):
        raise TypeError("health_check_sink must be callable or None")
    points = sorted(
        tuple(_entry_points() if entry_points is None else entry_points),
        key=lambda point: point.name,
    )
    duplicate_entry_points = {
        name for name, count in Counter(point.name for point in points).items()
        if count > 1
    }
    reports: list[ExtensionReport] = []
    specs: list[tuple[EntryPointLike, ExtensionSpec]] = []

    for point in points:
        if point.name in duplicate_entry_points:
            reports.append(
                _failure(
                    point.name,
                    "duplicate entry-point name",
                    point.name,
                )
            )
            continue
        try:
            factory = point.load()
            spec = factory()  # type: ignore[operator]
            if not isinstance(spec, ExtensionSpec):
                raise TypeError("factory did not return ExtensionSpec")
            if not isinstance(spec.name, str) or not spec.name:
                raise TypeError("extension name must be a non-empty string")
            if not isinstance(spec.api_version, int):
                raise TypeError("extension API version must be an integer")
            if not isinstance(spec.priority, int):
                raise TypeError("extension priority must be an integer")
            checker = spec.check_health
            if checker is not None:
                target = getattr(checker, "__call__", None)
                if not callable(checker) or any(
                    predicate(value)
                    for predicate in (inspect.iscoroutinefunction, inspect.isasyncgenfunction,
                                      inspect.isgeneratorfunction)
                    for value in (checker, target)
                ):
                    raise TypeError("check_health must be a synchronous callable or None")
                # Optional/future parameters are fine; required positional or
                # keyword-only arguments must fail before owner publication.
                # An uninspectable callable cannot prove this public ABI.
                inspect.signature(checker).bind()
            capabilities = frozenset(spec.capabilities)
            if any(not isinstance(name, str) or not name.strip() for name in capabilities):
                raise TypeError("capability names must be non-empty strings")
        except Exception as exc:
            fatal = is_fatal_extension_error(exc)
            description = _exception_description(exc)
            reports.append(
                _failure(
                    point.name,
                    f"load failed: {description}",
                )
            )
            if fatal:
                raise ExtensionCommitError(
                    f"fatal extension discovery failure at {point.name!r}: {description}",
                    _ordered_reports(reports),
                ) from exc
            continue

        if spec.api_version != EXTENSION_API_VERSION:
            reports.append(
                _failure(
                    point.name,
                    "incompatible extension API version: "
                    f"{spec.api_version} (expected {EXTENSION_API_VERSION})",
                    spec.name,
                    capabilities,
                )
            )
            continue
        specs.append(
            (
                point,
                ExtensionSpec(
                    spec.name,
                    spec.api_version,
                    spec.priority,
                    capabilities,
                    spec.activate,
                    spec.check_health,
                ),
            )
        )

    specs_by_name: dict[str, list[tuple[EntryPointLike, ExtensionSpec]]] = {}
    for point, spec in specs:
        specs_by_name.setdefault(spec.name, []).append((point, spec))

    valid_specs: list[tuple[EntryPointLike, ExtensionSpec]] = []
    for declared_name, declarations in specs_by_name.items():
        if len(declarations) == 1:
            valid_specs.extend(declarations)
            continue
        reason = f"duplicate declared extension name: {declared_name!r}"
        for point, spec in declarations:
            reports.append(
                _failure(
                    point.name,
                    reason,
                    spec.name,
                    spec.capabilities,
                )
            )

    capability_owners: dict[str, list[str]] = {}
    for _point, spec in valid_specs:
        for capability in spec.capabilities:
            capability_owners.setdefault(capability, []).append(spec.name)
    conflicts = {
        capability: tuple(sorted(owners))
        for capability, owners in capability_owners.items()
        if len(owners) > 1
    }
    unambiguous_specs = []
    for point, spec in valid_specs:
        overlapping = sorted(spec.capabilities.intersection(conflicts))
        if overlapping:
            reason = "duplicate capability owners: " + "; ".join(
                f"{name!r}: {', '.join(conflicts[name])}" for name in overlapping
            )
            reports.append(_failure(point.name, reason, spec.name, spec.capabilities))
        else:
            unambiguous_specs.append((point, spec))

    for point, spec in sorted(
        unambiguous_specs,
        key=lambda item: (-item[1].priority, item[1].name),
    ):
        try:
            spec.activate(context)
        except Exception as exc:
            fatal = is_fatal_extension_error(exc)
            description = _exception_description(exc)
            reports.append(
                _failure(
                    point.name,
                    f"activation failed: {description}",
                    spec.name,
                    spec.capabilities,
                )
            )
            if fatal:
                raise ExtensionCommitError(
                    f"fatal extension activation failure at {point.name!r}: {description}",
                    _ordered_reports(reports),
                ) from exc
        else:
            report = ExtensionReport(
                entry_point_name=point.name,
                extension_name=spec.name,
                status=ExtensionStatus.ACTIVE,
                capabilities=spec.capabilities,
                reason="",
            )
            if spec.check_health is not None and health_check_sink is not None:
                try:
                    health_check_sink(report, spec.check_health)
                except BaseException as exc:
                    # Do not append ACTIVE first and then another FAILED report
                    # for the same owner: consumers must see one real outcome.
                    reports.append(_failure(point.name,
                        f"health binding failed after activation: {type(exc).__name__}: {exc}",
                        spec.name, spec.capabilities))
                    raise ExtensionCommitError(
                        f"fatal extension health binding failure at {point.name!r}: {exc}",
                        _ordered_reports(reports),
                    ) from exc
            reports.append(report)

    ordered = _ordered_reports(reports)
    for required in sorted(context.requested_extensions):
        matching = [
            report for report in ordered
            if report.entry_point_name == required
        ]
        if not matching:
            raise RequiredExtensionError(
                f"required extension {required!r} was not discovered",
                ordered,
            )
        if any(
            report.status is not ExtensionStatus.ACTIVE
            for report in matching
        ):
            raise RequiredExtensionError(
                f"required extension {required!r} failed: "
                f"{matching[0].reason}",
                ordered,
            )

    active_capabilities = frozenset(
        capability
        for report in ordered
        if report.status is ExtensionStatus.ACTIVE
        for capability in report.capabilities
    )
    missing = sorted(
        context.requested_capabilities - active_capabilities
    )
    if missing:
        raise RequiredExtensionError(
            "requested capabilities missing from active extensions: "
            + ", ".join(
                f"capability {capability!r}" for capability in missing
            ),
            ordered,
        )
    return ordered


__all__ = [
    "EXTENSION_API_VERSION",
    "ExtensionContext",
    "ExtensionReport",
    "ExtensionSpec",
    "ExtensionStatus",
    "RequiredExtensionError",
    "ExtensionCommitError",
    "is_fatal_extension_error",
    "activate_extensions",
]
