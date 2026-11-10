# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""High-risk sparse-MLA vendor stress, separate from ordinary MHA/FA matrices.
Run on an idle HCU: pytest tests/vendor/test_flashmla_prefill.py -v
Not executed during CPU-only adaptation; NOT evidence of vendor qualification.
"""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('rows,heads,width,kv_rows', [
    (1,4,128,7), (3,8,256,129), (17,16,128,1023),
    (127,4,256,1024), (129,8,128,2051), (1,64,256,2051),
])
def test_stress_native_prefill(rows, heads, width, kv_rows):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires idle HCU; no CPU substitute for vendor stress')
    from hcu_adapter.kernels import mla_sparse_prefill_dtk as kernel
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in kernel.SUPPORTED_ARCHES:
        pytest.skip('requires gfx936/gfx938')
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd
    path = Path(__file__).parents[1] / 'kernels' / 'test_mla_sparse_prefill.py'
    spec = importlib.util.spec_from_file_location('_mla_prefill_contracts', path)
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    a = contracts.inputs(rows, heads, width, kv_rows, device='cuda')
    for wave in range(5):
        a['q'].add_(.03)
        a['kv'].mul_(-.7)
        a['indices'].add_(wave + 1).remainder_(kv_rows)
        a['topk_length'].copy_((torch.arange(rows, device='cuda') * 7 + wave * 11) % width)
        saved = a['indices'].clone()
        expected = contracts.reference(a)
        assert kernel.try_prefill(**a, provider=flash_mla_sparse_fwd)
        contracts.assert_numerics(a['output'], expected)
        assert torch.equal(a['indices'], saved)
