# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Generic health ABI/lifecycle regressions; no Torch, provider or GPU work."""
from dataclasses import asdict, fields, replace
import functools
import os
import select
import signal
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import extensions as ext
from hcu_adapter.bridges import vllm as bridge
from hcu_adapter import runtime_lifecycle as lifecycle


CAP = 'test-capability'


def point(checker=None, callback=lambda _: None, name='owner', priority=0):
    return NS(name=name, load=lambda: lambda: ext.ExtensionSpec(name, 1, priority, {CAP}, callback, checker))


def activate(actions, checker=lambda: None):
    reports = ext.activate_extensions(ext.ExtensionContext('cpu', '0.32.0'), [point(checker)],
                                     health_check_sink=actions.record_extension_health)
    actions.record_extension_reports(reports)
    return reports[0]


def test_additive_api1_fields_and_exact_active_report_sink_without_report_schema_change():
    assert ext.EXTENSION_API_VERSION == 1
    assert [f.name for f in fields(ext.ExtensionSpec)] == [
        'name', 'api_version', 'priority', 'capabilities', 'activate', 'check_health']
    calls = []
    checker = lambda: None
    report, = ext.activate_extensions(ext.ExtensionContext('cpu', '0.32.0'), [point(checker)],
                                     health_check_sink=lambda r, c: calls.append((r, c)))
    assert calls[0][0] is report and calls[0][1] is checker
    assert list(asdict(report)) == ['entry_point_name', 'extension_name', 'status', 'capabilities', 'reason']


async def asynchronous():
    pass


def generator():
    yield None


async def async_generator():
    yield None


class AsyncCallable:
    async def __call__(self):
        pass


@pytest.mark.parametrize('checker', [123, asynchronous, generator, async_generator, AsyncCallable()])
def test_invalid_checker_is_clean_failed_before_activation(checker):
    calls = []
    report, = ext.activate_extensions(ext.ExtensionContext('cpu', '0.32.0'),
                                     [point(checker, lambda _: calls.append('BAD'))])
    assert report.status is ext.ExtensionStatus.FAILED
    assert 'synchronous callable' in report.reason and not calls


class RequiredCallable:
    def __call__(self, required):
        pass


@pytest.mark.parametrize('render_error', [RuntimeError, KeyboardInterrupt])
def test_sink_broken_error_render_stays_typed_fatal_and_host_fork_refused(render_error):
    actions = bridge.ProcessLocalActions()
    calls = []
    class BadString(BaseException):
        def __str__(self):
            raise render_error('diagnostic failed')
    error = BadString()
    def sink(report, checker):
        raise error
    def callback():
        ext.activate_extensions(ext.ExtensionContext('cpu', '0.32.0'),
            [point(lambda: None, priority=10), NS(name='later', load=lambda: lambda:
                ext.ExtensionSpec('later', 1, 0, (), lambda _: calls.append('BAD')))],
            health_check_sink=sink)
    with pytest.raises(bridge.LatchedVllmRuntimeError) as caught:
        actions.run_once('vllm-runtime', callback)
    fatal = caught.value.__cause__
    assert isinstance(fatal, ext.ExtensionCommitError) and fatal.__cause__ is error
    report, = fatal.reports
    assert report.status is ext.ExtensionStatus.FAILED
    assert 'exception details unavailable' in report.reason and not calls
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_runtime_healthy()
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.run_once('vllm-runtime', lambda: calls.append('RETRY'))
    if hasattr(os, 'fork'):
        pid = os.fork()
        if pid == 0:
            try:
                actions.require_runtime_healthy()
            except bridge.LatchedVllmRuntimeError:
                os._exit(0)
            os._exit(3)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    assert not calls


@pytest.mark.parametrize('checker', [lambda required: None, lambda *, required: None,
                                    RequiredCallable(), functools.partial(lambda a, b: None, 1)])
@pytest.mark.parametrize('required', [False, True])
def test_required_checker_arguments_are_rejected_before_owner_activation(checker, required):
    calls = []
    context = ext.ExtensionContext('cpu', '0.32.0', requested_extensions={'owner'} if required else ())
    if required:
        with pytest.raises(ext.RequiredExtensionError) as caught:
            ext.activate_extensions(context, [point(checker, lambda _: calls.append('BAD'))])
        reports = caught.value.reports
    else:
        reports = ext.activate_extensions(context, [point(checker, lambda _: calls.append('BAD'))])
    report, = reports
    assert report.status is ext.ExtensionStatus.FAILED and 'required' in report.reason
    assert not calls


@pytest.mark.parametrize('checker', [lambda optional=None: None, lambda *, optional=None: None,
                                    lambda *args, **kwargs: None,
                                    functools.partial(lambda required: None, 1)])
def test_zero_argument_checker_shape_accepts_optional_future_parameters(checker):
    actions = bridge.ProcessLocalActions()
    activate(actions, checker)
    actions.require_extension_healthy(CAP)


def test_uninspectable_checker_is_rejected_cleanly_before_activation():
    class OpaqueCallable:
        __signature__ = 'invalid inspect signature'
        def __call__(self):
            pass
    calls = []
    report, = ext.activate_extensions(ext.ExtensionContext('cpu', '0.32.0'),
        [point(OpaqueCallable(), lambda _: calls.append('BAD'))])
    assert report.status is ext.ExtensionStatus.FAILED and not calls


def test_absent_checker_keeps_existing_activation_semantics_but_explicit_health_fails_closed():
    actions = bridge.ProcessLocalActions()
    activate(actions, None)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='exact current-process'):
        actions.require_extension_healthy(CAP)


@pytest.mark.parametrize('required', [False, True])
def test_sink_failure_is_fatal_one_report_per_owner_and_stops_later_activation(required):
    calls = []
    def sink(report, checker):
        calls.append(report)
        raise ValueError('binding failed')
    context = ext.ExtensionContext('cpu', '0.32.0', requested_extensions={'owner'} if required else ())
    with pytest.raises(ext.ExtensionCommitError) as caught:
        ext.activate_extensions(context, [point(lambda: None, priority=10),
            NS(name='later', load=lambda: lambda: ext.ExtensionSpec('later', 1, 0, (), lambda _: calls.append('BAD')))],
            health_check_sink=sink)
    report, = caught.value.reports
    assert report.entry_point_name == 'owner' and report.status is ext.ExtensionStatus.FAILED
    assert len(calls) == 1 and calls[0] is not report
    assert 'after activation' in report.reason


@pytest.mark.parametrize('replacement', ['equal', 'failed', 'duplicate', 'empty'])
def test_replaced_failed_or_duplicate_reports_cannot_borrow_checker_or_resurrect_it(replacement):
    actions = bridge.ProcessLocalActions()
    calls = []
    report = activate(actions, lambda: calls.append(1))
    new = {'equal': (replace(report),), 'failed': (replace(report, status=ext.ExtensionStatus.FAILED),),
           'duplicate': (report, report), 'empty': ()}[replacement]
    actions.record_extension_reports(new)
    actions.record_extension_reports((report,))
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='health binding'):
        actions.require_extension_healthy(CAP)
    assert not calls


def test_duplicate_health_binding_is_public_commit_fatal():
    actions = bridge.ProcessLocalActions()
    report = activate(actions)
    with pytest.raises(ext.ExtensionCommitError, match='duplicate'):
        actions.record_extension_health(report, lambda: None)


def test_health_never_invokes_owner_under_outer_activation_lock_even_when_error_is_caught():
    actions = bridge.ProcessLocalActions()
    calls = []
    activate(actions, lambda: calls.append('BAD'))
    def callback():
        with pytest.raises(bridge.LatchedVllmRuntimeError, match='completed host activation'):
            actions.require_extension_healthy(CAP)
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.run_once('vllm-runtime', callback)
    assert not calls


def test_current_pid_mismatch_refuses_even_with_exact_report():
    actions = bridge.ProcessLocalActions()
    report = activate(actions)
    binding = actions._health_bindings[report.entry_point_name]
    actions._health_bindings[report.entry_point_name] = (report, binding[1], os.getpid() + 1)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='current-process'):
        actions.require_extension_healthy(CAP)


@pytest.mark.parametrize('returned', [1, 'not-none', asynchronous])
def test_return_value_must_be_synchronous_none_and_caught_error_cannot_clear_latch(returned):
    actions = bridge.ProcessLocalActions()
    activate(actions, lambda: returned() if returned is asynchronous else returned)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='return None'):
        actions.require_extension_healthy(CAP)
    actions.record_extension_reports(())
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.run_once('vllm-runtime', lambda: pytest.fail('must not run'))


@pytest.mark.parametrize('operation', ['runtime', 'reports', 'activate', 'health'])
@pytest.mark.parametrize('caught', [False, True])
def test_checker_host_reentry_fails_without_acquiring_host_lock_even_if_caught(operation, caught):
    actions = bridge.ProcessLocalActions()
    def checker():
        try:
            {'runtime': actions.require_runtime_healthy, 'reports': actions.extension_reports,
             'activate': lambda: actions.run_once('vllm-runtime', lambda: None),
             'health': lambda: actions.require_extension_healthy(CAP)}[operation]()
        except ext.ExtensionCommitError:
            if not caught:
                raise
    activate(actions, checker)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='reentr|re-enter'):
        actions.require_extension_healthy(CAP)
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_runtime_healthy()


def test_checker_runs_without_host_lock_and_concurrent_report_change_is_latched():
    actions = bridge.ProcessLocalActions()
    entered, changed = threading.Event(), threading.Event()
    def checker():
        entered.set()
        assert changed.wait(3), 'host lock was held while invoking owner checker'
    report = activate(actions, checker)
    def replace_reports():
        assert entered.wait(3)
        actions.record_extension_reports((replace(report),))
        changed.set()
    worker = threading.Thread(target=replace_reports)
    worker.start()
    try:
        with pytest.raises(bridge.LatchedVllmRuntimeError, match='changed during'):
            actions.require_extension_healthy(CAP)
    finally:
        worker.join(4)
    assert not worker.is_alive()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
def test_fork_clears_reports_and_checkers_and_requires_new_activation():
    actions = bridge.ProcessLocalActions()
    report = activate(actions)
    pid = os.fork()
    if pid == 0:
        try:
            assert not actions.extension_reports() and not actions._health_bindings
            activate(actions)
            actions.require_extension_healthy(CAP)
        except BaseException:
            os._exit(2)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert actions.extension_reports()[0] is report
    actions.require_extension_healthy(CAP)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
def test_fork_retains_failed_health_latch_even_after_new_child_report_binding():
    actions = bridge.ProcessLocalActions()
    def fail():
        raise ValueError('observed owner drift')
    activate(actions, fail)
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_extension_healthy(CAP)
    pid = os.fork()
    if pid == 0:
        try:
            assert not actions.extension_reports() and not actions._health_bindings
            activate(actions)
            try:
                actions.require_extension_healthy(CAP)
            except bridge.LatchedVllmRuntimeError:
                os._exit(0)
        except BaseException:
            os._exit(2)
        os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
def test_fork_during_health_retains_incomplete_latch_without_inheriting_locks():
    actions = bridge.ProcessLocalActions()
    entered, release = threading.Event(), threading.Event()
    errors = []
    def checker():
        entered.set()
        assert release.wait(4)
    activate(actions, checker)
    def checking():
        try:
            actions.require_extension_healthy(CAP)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=checking)
    worker.start()
    assert entered.wait(3)
    pid = os.fork()
    if pid == 0:
        try:
            assert not actions.extension_reports() and not actions._health_bindings
            try:
                actions.require_runtime_healthy()
            except bridge.LatchedVllmRuntimeError as error:
                os._exit(0 if 'incomplete extension health' in str(error) else 3)
        except BaseException:
            os._exit(2)
        os._exit(4)
    release.set()
    worker.join(5)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not worker.is_alive() and not errors
    assert actions._health_inflight == 0
    actions.require_runtime_healthy()


def fork_probe_health(actions):
    """Actual child observes health; pipe/kill bounds child even on a lock bug."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            try:
                actions.require_runtime_healthy()
            except bridge.LatchedVllmRuntimeError:
                payload = b'REFUSED'
            else:
                payload = b'HEALTHY'
            os.write(write_fd, payload)
        except BaseException:
            os._exit(2)
        os._exit(0)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 3)
        if not ready:
            os.kill(pid, signal.SIGKILL)
        payload = os.read(read_fd, 32) if ready else b'TIMEOUT'
        _, status = os.waitpid(pid, 0)
        assert ready and os.waitstatus_to_exitcode(status) == 0
        return payload
    finally:
        os.close(read_fd)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
@pytest.mark.parametrize('fault', ['checker', 'snapshot', 'binding'])
def test_exception_formatting_fork_sees_refusal_before_latch_details_and_counter_release(monkeypatch, fault):
    actions = bridge.ProcessLocalActions()
    observations = []
    class ForkOnStrError(ext.ExtensionCommitError):
        def __str__(self):
            observations.append((actions._health_inflight, fork_probe_health(actions)))
            return 'forked during exception formatting'
    monkeypatch.setattr(lifecycle, 'ExtensionCommitError', ForkOnStrError)
    def checker():
        if fault == 'checker':
            raise ForkOnStrError('checker failure')
        if fault == 'snapshot':
            thread = threading.Thread(target=lambda: actions.record_extension_reports((replace(report),)))
            thread.start()
            thread.join(2)
            assert not thread.is_alive()
    report = activate(actions, checker)
    if fault == 'binding':
        actions.record_extension_reports((replace(report),))
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_extension_healthy(CAP)
    assert observations == [(1, b'REFUSED')]
    assert actions._health_inflight == 0
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_runtime_healthy()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
def test_fork_before_failure_record_write_is_covered_by_inflight_until_latched():
    actions = bridge.ProcessLocalActions()
    entered, release = threading.Event(), threading.Event()
    worker_id = []
    parent_pid = os.getpid()
    errors = []
    class BeforeLatch(dict):
        worker_gets = 0
        def get(self, key, default=None):
            if (key == 'vllm-runtime' and os.getpid() == parent_pid
                    and worker_id and threading.get_ident() == worker_id[0]):
                self.worker_gets += 1
                if self.worker_gets == 2:
                    # First get is the initial health query; second is the
                    # exception handler immediately before it stores failure.
                    entered.set()
                    assert release.wait(5)
            return super().get(key, default)
    def checker():
        raise ValueError('checker failed before latch')
    activate(actions, checker)
    actions._records = BeforeLatch(actions._records)
    def checking():
        worker_id.append(threading.get_ident())
        try:
            actions.require_extension_healthy(CAP)
        except bridge.LatchedVllmRuntimeError as error:
            errors.append(error)
    worker = threading.Thread(target=checking)
    worker.start()
    try:
        assert entered.wait(3)
        assert dict.get(actions._records, 'vllm-runtime') is None
        assert actions._health_inflight == 1
        assert fork_probe_health(actions) == b'REFUSED'
    finally:
        release.set()
        worker.join(6)
    assert not worker.is_alive() and len(errors) == 1 and actions._health_inflight == 0
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.require_runtime_healthy()


@pytest.mark.parametrize('error', [ValueError, KeyboardInterrupt])
def test_broken_exception_string_cannot_prevent_fatal_latch(error):
    actions = bridge.ProcessLocalActions()
    class BrokenStrError(Exception):
        def __str__(self):
            raise error('rendering failed')
    def checker():
        raise BrokenStrError()
    activate(actions, checker)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match='details unavailable'):
        actions.require_extension_healthy(CAP)
    assert actions._health_inflight == 0
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        actions.run_once('vllm-runtime', lambda: pytest.fail('must remain refused'))
