# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Portable sparse FlashMLA ABI checks; no GPU or model qualification."""
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as kernel


def vendor_sparse(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                  topk_length=None):
    return q[..., :d_v] + 1, 'maximum', 'lse'


def test_sparse_output_buffer_and_sink_reach_vendor():
    calls = []

    def provider(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                 topk_length=None):
        calls.append((q, kv, indices, sm_scale, d_v, attn_sink, topk_length))
        return vendor_sparse(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)

    wrapped = kernel.sparse_entry(provider)
    q = torch.zeros(2, 64, 576, dtype=torch.bfloat16)
    kv, indices = object(), object()
    sinks, lengths = object(), object()
    out = torch.empty(2, 64, 512, dtype=q.dtype)
    result, maximum, lse = wrapped(q, kv, indices, .25,
        attn_sink=sinks, topk_length=lengths, out=out)
    assert result is out
    assert torch.equal(result, torch.ones_like(out))
    assert (maximum, lse) == ('maximum', 'lse')
    assert calls == [(q, kv, indices, .25, 512, sinks, lengths)]


def test_no_output_buffer_keeps_vendor_result_identity():
    expected = (object(), object(), object())
    wrapped = kernel.sparse_entry(lambda *args, **kwargs: expected)
    assert wrapped(None, None, None, .5) is expected


def test_private_bind_preserves_original_globals_and_defaults():
    def dependency():
        return 'original'

    # A real function namespace, not a mutation of the caller's module globals.
    namespace = {'dependency': dependency}
    exec('def native(value=3):\n return dependency(), value\n', namespace)
    original = namespace['native']
    bound = kernel.bind_native(original, {'dependency': lambda: 'vendor'})
    assert bound() == ('vendor', 3)
    assert original() == ('original', 3)
    assert bound.__code__ is original.__code__
    assert bound.__defaults__ is original.__defaults__
    assert namespace['dependency'] is dependency
    with pytest.raises(RuntimeError, match='anchor'):
        kernel.bind_native(original, {'missing': dependency})


def test_built_classes_retain_core_forward_metadata_and_sink_validation():
    """Actual installed Core classes; no constructor or native launch."""
    from vllm.models.hy_v4.nvidia.flashmla_sparse import (
        HYV4FlashMLASparseImpl as Native,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder as Builder,
    )
    vendor = SimpleNamespace(flash_mla_sparse_fwd=vendor_sparse,
        flash_mla_with_kvcache=lambda **kwargs: None,
        get_mla_metadata=lambda *args, **kwargs: None)
    backend = kernel.build_backend(vendor)
    impl, builder = backend.get_impl_cls(), backend.get_builder_cls()
    assert backend.get_name() == 'FLASHMLA_SPARSE'
    assert backend.supports_sink()
    assert not backend.supports_dcp()
    assert impl.forward_mqa is Native.forward_mqa
    assert impl.__init__ is Native.__init__
    assert impl._sinks_for_query is Native._sinks_for_query
    assert builder.build is Builder.build
    assert impl._bf16_flash_mla_kernel.__code__ is Native._bf16_flash_mla_kernel.__code__
    assert impl._fp8_flash_mla_kernel.__code__ is Native._fp8_flash_mla_kernel.__code__
    assert impl._bf16_flash_mla_kernel.__globals__ is not Native._bf16_flash_mla_kernel.__globals__


def test_registration_uses_public_backend_registry_and_checks_owner(monkeypatch):
    import importlib
    from hcu_adapter.bridges import vllm_sparse_flashmla as bridge_module
    from hcu_adapter.bridges import vllm as host
    from hcu_adapter.infra import registry
    bridge = importlib.reload(bridge_module)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx938', ''))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.33.0')
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})

    class Enum:
        current = 'stock.Backend'

        def is_overridden(self):
            return self.current != 'stock.Backend'

        def get_path(self):
            return self.current

        def get_class(self):
            return cls

    enum = Enum()
    def register(key, path):
        assert key is enum
        enum.current = path
    fake_registry = SimpleNamespace(AttentionBackendEnum=SimpleNamespace(FLASHMLA_SPARSE=enum),
                                    register_backend=register)
    monkeypatch.setattr(bridge, '_backend_registry', lambda: fake_registry)
    vendor = SimpleNamespace(flash_mla_sparse_fwd=vendor_sparse,
        flash_mla_with_kvcache=lambda **kwargs: None,
        get_mla_metadata=lambda: None)
    monkeypatch.setattr(bridge, '_native_provider', lambda: vendor)
    cls = type('Backend', (), {'__module__': 'hcu_adapter.kernels.vllm_sparse_flashmla_dtk'})
    monkeypatch.setattr(kernel, 'build_backend', lambda vendor: cls)
    assert bridge.install()
    assert enum.get_path() == bridge.BACKEND
    bridge.require_available()
    assert registry.get(bridge.OP)['mechanism'] == registry.MECH_SEAM
    enum.current = 'foreign.Backend'
    with pytest.raises(RuntimeError, match='owner|publication'):
        bridge.require_healthy()
    enum.current = bridge.BACKEND
    with pytest.raises(RuntimeError):
        bridge.require_healthy()
    importlib.reload(bridge)


def test_killed_registration_does_not_probe_vendor(monkeypatch):
    import importlib
    from hcu_adapter.bridges import vllm_sparse_flashmla as bridge_module
    bridge = importlib.reload(bridge_module)
    monkeypatch.setenv('HCU_DISABLE_OPS', bridge.OP)
    monkeypatch.setattr(bridge, '_native_provider', lambda: pytest.fail('vendor import'))
    assert bridge.install() is False
    with pytest.raises(RuntimeError, match='unavailable'):
        bridge.require_available()
    importlib.reload(bridge)
