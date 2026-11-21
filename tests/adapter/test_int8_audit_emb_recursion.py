# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression: the int8-audit embedding wrapper must call the PRE-patch
forward (PR #142 review P2).

The old wrapper bound ``functools.partial(_Emb.forward, self)`` AFTER
``_Emb.forward = _emb_audited`` had taken effect, so the "original" was the
wrapper itself: the first embedding forward dove into unbounded recursion
(RecursionError). The fix captures the un-patched forward before the class
attribute is replaced -- the same shape the RMSNorm wrapper in that file
already used.
"""

import builtins
import importlib
import io
import sys
import types

import pytest


class _FakeVT:  # minimal tensor face used by the audit-log arm
    def numel(self):
        return 4

    def float(self):
        return self

    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return 0.0


def _install_with_stubs(monkeypatch):
    calls = []

    class VocabParallelEmbedding:
        def forward(self, x):
            calls.append("orig")
            return _FakeVT()

    class RMSNorm:
        def forward(self, x, residual=None):
            return x

    class _Scheme:
        def apply_weights(self, layer, x, bias):
            return x

        def process_weights_after_loading(self, layer):
            return layer

    class _Q35:
        @staticmethod
        def load_weights(weights):
            return None

    def _pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []  # importable package face for parent resolution
        return m

    leaf = {
        "sglang.srt.layers.vocab_parallel": dict(
            VocabParallelEmbedding=VocabParallelEmbedding
        ),
        "sglang.srt.layers.layernorm": dict(RMSNorm=RMSNorm),
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_int8": dict(CompressedTensorsW8A8Int8=_Scheme),
        "sglang.srt.models.qwen3_5": dict(Qwen3_5ForConditionalGeneration=_Q35),
    }
    stubs = {name: _pkg(name) for name in (
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.compressed_tensors",
        "sglang.srt.layers.quantization.compressed_tensors.schemes",
        "sglang.srt.models",
    )}
    for name, attrs in leaf.items():
        m = _pkg(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        stubs[name] = m
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    captured = io.StringIO()
    real_open = builtins.open

    def _fake_open(path, *args, **kwargs):
        if isinstance(path, str) and "int8_apply_dump.log" in path:
            return captured
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _fake_open)
    monkeypatch.setenv("HCU_DUMP_INT8", "1")

    audit = importlib.import_module("hcu_adapter.bridges.sglang_int8_audit")
    audit.install()
    return stubs["sglang.srt.layers.vocab_parallel"].VocabParallelEmbedding, calls, captured


def test_emb_wrapper_calls_pre_patch_forward(monkeypatch):
    emb_cls, calls, _ = _install_with_stubs(monkeypatch)

    x = _FakeVT()
    out1 = emb_cls.forward(emb_cls(), x)  # == _emb_audited(self, x)
    out2 = emb_cls.forward(emb_cls(), x)

    assert calls == ["orig", "orig"]
    assert out1 is not None and out2 is not None


def test_emb_wrapper_audits_once_per_instance(monkeypatch):
    emb_cls, _, captured = _install_with_stubs(monkeypatch)

    emb = emb_cls()
    emb_cls.forward(emb, _FakeVT())
    first = captured.getvalue().count("EMB xamax=")
    emb_cls.forward(emb, _FakeVT())
    second = captured.getvalue().count("EMB xamax=")

    assert first == 1
    assert second == 1  # no re-audit on subsequent calls
