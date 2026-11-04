# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free process-local lifecycle and extension-health transactions.

Host bridges supply their protected action ID and public error type. This
module owns locks, report binding, fork reset and failure publication together;
no framework admission, backend mapping or activation policy lives here.
"""
from __future__ import annotations

from collections.abc import Callable
import os
from threading import RLock, local

from hcu_adapter.extensions import ExtensionCommitError, ExtensionReport, ExtensionStatus


class ProcessLocalLifecycle:
    """Run required callbacks once per PID and latch partial failures."""

    def __init__(self, *, runtime_action: str, error_type: type[RuntimeError] = RuntimeError) -> None:
        if not isinstance(runtime_action, str) or not runtime_action:
            raise TypeError("runtime_action must be a non-empty string")
        if not isinstance(error_type, type) or not issubclass(error_type, RuntimeError):
            raise TypeError("error_type must be a RuntimeError subclass")
        self._runtime_action = runtime_action
        self._error_type = error_type
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
        # Only the configured runtime action inherits failure. Its registry
        # mutations also survive fork. Other actions keep their old reset.
        failure = self._records.get(self._runtime_action)
        if failure is None and self._runtime_action in self._running:
            failure = f"inherited incomplete {self._runtime_action} activation; spawn a fresh process"
        if failure is None and self._health_inflight:
            failure = "inherited incomplete extension health check; spawn a fresh process"
        self._owner_pid = owner_pid
        self._lock = RLock()
        self._records = {} if failure is None else {self._runtime_action: failure}
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
                    raise self._error_type(failure)
                return
            if action_id == self._runtime_action and action_id in self._running:
                raise self._error_type(f"recursive {self._runtime_action} activation")
            self._running.add(action_id)
            try:
                callback()
            except Exception as exc:
                # Diagnostic formatting can raise, re-enter or fork. Publish
                # refusal before it runs and before finally clears _running.
                failure = "runtime action failed; exception details unavailable"
                self._records[action_id] = failure
                try:
                    failure = f"{type(exc).__name__}: {exc}"
                except BaseException:
                    pass
                self._records[action_id] = failure
                raise self._error_type(failure) from exc
            finally:
                self._running.discard(action_id)
            if self._records.get(action_id) is not None:
                raise self._error_type(self._records[action_id])
            self._records[action_id] = None

    def require_runtime_healthy(self) -> None:
        """Read failure only; safe/reentrant inside the runtime callback."""
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            failure = self._records.get(self._runtime_action)
            if failure is not None:
                raise self._error_type(failure)

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
                failure = self._records.get(self._runtime_action)
                if failure is None:
                    # Publish refusal BEFORE arbitrary exception __str__ code.
                    # Formatting can raise, re-enter, block or fork. A child
                    # must inherit a failure even at that exact boundary.
                    failure = "extension health failed; exception details unavailable"
                    self._records[self._runtime_action] = failure
                    published = True
            if published:
                try:
                    detail = f"extension health failed: {type(exc).__name__}: {exc}"
                except BaseException:
                    detail = failure
                with self._lock:
                    self._records[self._runtime_action] = detail
                    failure = detail
            raise self._error_type(failure) from exc
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


