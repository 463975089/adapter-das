# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Spawn must select the assigned GPU before GPU-initializing imports."""

from contextlib import contextmanager
import inspect
import json
import os
import sys
import types

import pytest

from hcu_adapter import sglang_spawn_device as subject


def entry(*, selected=None, failure=False):
    namespace = {"events": [], "selected": selected, "failure": failure}
    exec(
        "def reindex(gpu_id):\n"
        "    events.append(('enter', gpu_id))\n"
        "    try:\n"
        "        yield gpu_id if selected is None else selected\n"
        "        if failure: raise ValueError('upstream exit')\n"
        "    finally:\n"
        "        events.append(('exit', gpu_id))\n",
        namespace,
    )
    return contextmanager(namespace["reindex"]), namespace["events"]


@pytest.mark.parametrize("device", [0, 1, 7])
def test_existing_alias_and_original_context_survive(monkeypatch, device):
    monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    original, events = entry()
    alias = original
    signature = inspect.signature(original)
    subject._install_function(original)
    subject._install_function(original)
    assert inspect.signature(alias) == signature
    with alias(device) as selected:
        assert selected == device
        assert json.loads(os.environ[subject._DEVICE_ENV]) == {
            "parent_pid": os.getpid(), "device": device,
        }
        assert events == [("enter", device)]
    assert events == [("enter", device), ("exit", device)]
    assert subject._DEVICE_ENV not in os.environ


@pytest.mark.parametrize("failure", ["body", "upstream", "none"])
@pytest.mark.parametrize("previous", [None, "outer-hint"])
def test_environment_restored_on_normal_and_exceptional_exit(monkeypatch, failure, previous):
    if previous is None:
        monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(subject._DEVICE_ENV, previous)
    original, events = entry(failure=failure == "upstream")
    subject._install_function(original)
    def run():
        with original(2):
            if failure == "body":
                raise ValueError("body")
    if failure == "none":
        run()
    else:
        with pytest.raises(ValueError):
            run()
    assert os.environ.get(subject._DEVICE_ENV) == previous
    assert events == [("enter", 2), ("exit", 2)]


def test_nested_contexts_restore_the_outer_hint(monkeypatch):
    monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    original, _ = entry()
    subject._install_function(original)
    with original(1):
        outer = os.environ[subject._DEVICE_ENV]
        with original(6):
            assert json.loads(os.environ[subject._DEVICE_ENV])["device"] == 6
        assert os.environ[subject._DEVICE_ENV] == outer
    assert subject._DEVICE_ENV not in os.environ


def test_hint_uses_the_yielded_reindexed_device(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    original, events = entry(selected=0)
    subject._install_function(original)
    with original(1) as selected:
        assert selected == 0
        assert json.loads(os.environ[subject._DEVICE_ENV])["device"] == 0
        assert os.environ["HIP_VISIBLE_DEVICES"] == "7,3"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7,3"
    assert events == [("enter", 1), ("exit", 1)]


def test_original_global_rebindings_remain_visible(monkeypatch):
    monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    original, _ = entry(selected=0)
    subject._install_function(original)
    original.__wrapped__.__globals__["selected"] = 3
    with original(7) as selected:
        assert selected == 3
        assert json.loads(os.environ[subject._DEVICE_ENV])["device"] == 3


@pytest.mark.parametrize("selected", [-1, True, "1"])
def test_invalid_upstream_device_does_not_publish_a_hint(monkeypatch, selected):
    monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    original, _ = entry(selected=selected)
    subject._install_function(original)
    with pytest.raises((RuntimeError, TypeError)):
        with original(1):
            pytest.fail("invalid device was accepted")
    assert subject._DEVICE_ENV not in os.environ


def test_changed_contract_and_foreign_binding_are_rejected():
    with pytest.raises(RuntimeError, match="contract changed"):
        subject._install_function(lambda gpu_id: gpu_id)
    original, _ = entry()
    original.__wrapped__.__globals__[subject._DISPATCH] = object()
    with pytest.raises(RuntimeError, match="already exists"):
        subject._install_function(original)
    original, _ = entry()
    subject._install_function(original)
    original.__wrapped__.__globals__[subject._DISPATCH] = object()
    with pytest.raises(RuntimeError, match="binding changed"):
        subject._install_function(original)



def arm_child(monkeypatch, hint, *, argv=("-c", "--multiprocessing-fork")):
    devices = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(set_device=devices.append)))
    monkeypatch.setattr(sys, "argv", list(argv))
    if hint is None:
        monkeypatch.delenv(subject._DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(subject._DEVICE_ENV, hint)
    return devices


@pytest.mark.parametrize("device", [0, 1, 7])
def test_child_selects_once_and_consumes_the_hint(monkeypatch, device):
    devices = arm_child(monkeypatch, json.dumps({
        "parent_pid": os.getppid(), "device": device,
    }))
    subject.select_inherited_device()
    subject.select_inherited_device()
    assert devices == [device]
    assert subject._DEVICE_ENV not in os.environ


@pytest.mark.parametrize("argv,hint", [
    (("-c", "--multiprocessing-fork"), None),
    (("-c",), "resource-tracker-must-not-select-a-gpu"),
    (("sglang", "serve"), "launcher-must-not-select-a-gpu"),
])
def test_unassigned_and_non_spawn_helpers_do_not_select(monkeypatch, argv, hint):
    devices = arm_child(monkeypatch, hint, argv=argv)
    subject.select_inherited_device()
    assert not devices
    assert subject._DEVICE_ENV not in os.environ


@pytest.mark.parametrize("hint", ["invalid-json", "null", "[]", "{}",
    '{"parent_pid": true, "device": 1}',
    '{"parent_pid": 1, "device": -1}',
    '{"parent_pid": 1, "device": "1"}',
    '{"parent_pid": 1, "device": true}',
    '{"parent_pid": -1, "device": 1}',
])
def test_invalid_and_stale_hints_fail_before_any_gpu_selection(monkeypatch, hint):
    devices = arm_child(monkeypatch, hint)
    with pytest.raises(RuntimeError, match="device hint"):
        subject.select_inherited_device()
    assert not devices
    assert subject._DEVICE_ENV not in os.environ
