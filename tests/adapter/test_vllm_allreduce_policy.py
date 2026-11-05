# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free policy gates; no device/provider/model is loaded."""
import importlib

import pytest


@pytest.mark.parametrize("kill,optin,expected", [("aiter_ar", "1", False), ("", "1", True), ("", "0", False)])
def test_existing_kill_precedes_registered_optin(monkeypatch, kill, optin, expected):
    monkeypatch.setenv("HCU_DISABLE_OPS", kill)
    monkeypatch.setenv("HCU_AITER_AR", optin)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1")
    policy = importlib.import_module("hcu_adapter.allreduce").requested_policy()
    assert policy.selected is expected


def test_vllm_auto_does_not_probe_sglang(monkeypatch):
    from hcu_adapter import hcu_env
    monkeypatch.delenv("HCU_AITER_AR", raising=False)
    monkeypatch.setattr(hcu_env, "_sglang_uses_new_aiter_ca", lambda: pytest.fail("vLLM imported SGLang"))
    assert hcu_env.resolve("HCU_AITER_AR", host="vllm")[0].startswith("refused")


def test_existing_config_selection_and_explicit_kill_precedence(monkeypatch, tmp_path):
    from hcu_adapter import allreduce, hcu_config
    config=tmp_path/'hcu.json'
    config.write_text('{"version":1,"ops":{"allreduce":"aiter"}}')
    monkeypatch.setenv('HCU_CONFIG',str(config))
    monkeypatch.delenv('HCU_AITER_AR',raising=False)
    monkeypatch.setenv('HCU_DISABLE_OPS','')
    monkeypatch.setattr(hcu_config,'_STATE',dict(loaded=False,path=None,ops={},overrides={},hosts={}))
    assert allreduce.requested_policy().selected
    assert allreduce.effective_sources()['HCU_AITER_AR']==('1','config')
    monkeypatch.setenv('HCU_DISABLE_OPS','aiter_ar')
    assert not allreduce.requested_policy().selected
