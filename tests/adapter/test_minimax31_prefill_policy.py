# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Public configuration for the external MiniMax prefill provider."""
import json
import pytest
from hcu_adapter.infra import hcu_config
from hcu_adapter.infra import hcu_env
from hcu_adapter.infra.kill_switch import op_disabled


@pytest.mark.parametrize('backend', ['triton', 'stock'])
def test_prefill_choice_survives_real_config_loading(tmp_path, monkeypatch, backend):
    path = tmp_path/'hcu_adapter.json'
    path.write_text(json.dumps({'version': 1, 'ops': {'vllm_minimax_prefill': backend}}))
    monkeypatch.setenv('HCU_CONFIG', str(path))
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': False, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    state = hcu_config.load()
    assert state['ops']['vllm_minimax_prefill'] == backend
    assert not any('MINIMAX_PREFILL' in key for key in state['overrides'])
    assert not any('MINIMAX_PREFILL' in key for key in hcu_env.REGISTRY)


def test_invalid_prefill_choice_is_not_silently_ignored():
    with pytest.raises(hcu_config.HcuConfigError, match='unknown backend'):
        hcu_config._validate({'version': 1, 'ops': {'vllm_minimax_prefill': 'typo'}}, 'input.json')


def test_prefill_veto_uses_existing_master_switch(monkeypatch):
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_minimax_prefill')
    assert op_disabled('vllm_minimax_prefill')


def test_prefill_ui_snapshot_has_the_rendered_status_and_note_fields():
    import re
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2]/'tools/hcu_config_ui.html').read_text()
    snapshot = json.loads(re.search(r'const CATALOG = (\{.*?\});',source,re.S).group(1))
    for backend, entry in hcu_config.BACKEND_CATALOG['vllm_minimax_prefill']['backends'].items():
        ui = snapshot['vllm_minimax_prefill']['backends'][backend]
        assert ui['s'] == entry['status']
        assert ui['n'] == entry['note']
