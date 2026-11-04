# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU dtype diagnostics preserve fixed prefixes and cannot claim acceptance."""

from types import SimpleNamespace
import json
import sys

import pytest
import torch

from tools.l15_checkpoint_dtype_probe import compare_dtypes, probabilities, run
from tools import l15_checkpoint_dtype_probe as probe


def head(lp):
    values, tokens = lp.topk(20, dim=-1)
    return [[[score, token, None] for score, token in zip(scores, ids)]
            for scores, ids in zip(values.tolist(), tokens.tolist())]


def test_dtype_and_device_errors_have_independent_token_locations():
    single = torch.linspace(2, -2, 32).repeat(2, 1).log_softmax(-1)
    reduced_logits = single.clone()
    reduced_logits[1, 7] += .5
    reduced = reduced_logits.log_softmax(-1)
    observed_logits = reduced.clone()
    observed_logits[0, 3] -= .8
    observed = observed_logits.log_softmax(-1)
    result = compare_dtypes(single, reduced, head(observed))
    dtype = result["worst_cpu_dtype_on_hcu_labels"]
    device = result["worst_hcu_vs_fp32"]
    assert (dtype["output_position"], dtype["token_id"]) == (1, 7)
    assert (device["output_position"], device["token_id"]) == (0, 3)
    assert dtype["absolute_error"] == pytest.approx(
        abs(reduced[1, 7].item() - single[1, 7].item()))
    assert device["cpu_fp32_logprob"] == pytest.approx(single[0, 3].item())
    assert device["cpu_bf16_logprob"] == pytest.approx(reduced[0, 3].item())
    assert result["hcu_vs_cpu_bf16"]["max_top20_logprob_error"] > .7
    assert result["cpu_bf16_vs_cpu_fp32"]["max_top20_logprob_error"] > .4
    assert "passed" not in result and "policy" not in result


def test_equal_references_do_not_invent_dtype_noise_or_hide_hcu_error():
    lp = torch.linspace(2, -2, 32)[None].log_softmax(-1)
    observed = head(lp)
    observed[0][2][0] -= .4
    result = compare_dtypes(lp, lp.clone(), observed)
    assert result["cpu_bf16_vs_cpu_fp32"]["max_top20_logprob_error"] == 0
    assert result["worst_cpu_dtype_on_hcu_labels"]["absolute_error"] == 0
    assert result["hcu_vs_cpu_fp32"]["max_top20_logprob_error"] == pytest.approx(.4)
    with pytest.raises(ValueError, match="shapes"):
        compare_dtypes(lp, lp.repeat(2, 1), observed)
    observed[0][1][1] = observed[0][0][1]
    with pytest.raises(ValueError, match="identities"):
        compare_dtypes(lp, lp, observed)


@pytest.mark.parametrize("keep", [False, True])
def test_teacher_forcing_includes_exact_prefix_and_excludes_final_extra_logit(keep):
    logits = torch.zeros(1, 5, 32)
    for position in range(5):
        logits[0, position, position] = position + 1
    seen = []

    class Full:
        def forward(self, input_ids, use_cache):
            seen.append((input_ids.tolist(), use_cache))
            return SimpleNamespace(logits=logits)

        __call__ = forward

    class Keep:
        def forward(self, input_ids, use_cache, logits_to_keep):
            seen.append((input_ids.tolist(), use_cache))
            assert logits_to_keep == 3
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        __call__ = forward

    row = {"prompt_ids": [2, 3, 4], "output_ids": [5, 6]}
    actual = probabilities(Keep() if keep else Full(), row, 32)
    assert seen == [([[2, 3, 4, 5, 6]], False)]
    torch.testing.assert_close(actual, logits[0, 2:4].log_softmax(-1))
    with pytest.raises(ValueError, match="vocabulary"):
        probabilities(Full(), row, 4)


def test_diagnostic_cannot_replace_original_evidence(tmp_path):
    report = tmp_path / "smoke.json"
    report.write_text("original")
    with pytest.raises(ValueError, match="preserve"):
        run(SimpleNamespace(out=report, smoke_report=report))
    assert report.read_text() == "original"


@pytest.mark.parametrize("fault", ["clean", "unbound", "changed", "explicit-quantization"])
def test_run_records_diagnostic_without_gate_and_checks_final_bytes(tmp_path, monkeypatch, fault, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"vocab_size":32,"model_type":"test"}')
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps({
        "generation_profile": "accuracy",
        "quantization": "w8a8_int8" if fault == "explicit-quantization" else None,
    }))
    bound = {"asset": "original"}
    row = {"prompt_ids": [2, 3], "output_ids": [4, 5], "mode": "eager", "case": "test",
           "observed_top_logprobs": head(torch.zeros(2, 32).log_softmax(-1))}
    monkeypatch.setattr(probe, "requests", lambda _: (model_path, [row, dict(row)]))
    monkeypatch.setattr(probe, "execution_binding",
                        lambda *_: None if fault == "unbound" else bound)
    monkeypatch.setattr(probe, "fingerprint", lambda _: bound)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test"))
    from hcu_adapter import hcu_env
    monkeypatch.setattr(hcu_env, "resolve", lambda _: ("none", "test"))
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(32, dtype=dtype))
            # Official loaders retain FP32 RoPE frequencies even with BF16
            # parameters. A whole-model cast must fail this regression.
            self.register_buffer("inv_freq", torch.tensor([.1234567], dtype=torch.float32))

        def forward(self, input_ids, use_cache, logits_to_keep):
            calls.append((self.weight.dtype, input_ids.tolist()))
            assert self.inv_freq.dtype == torch.float32
            if fault == "changed":
                source.write_text("changed during inference")
            return SimpleNamespace(logits=self.weight.expand(1, logits_to_keep, 32))

    loaded = []

    def load(*_, **kwargs):
        assert kwargs["attn_implementation"] == "eager" and kwargs["local_files_only"]
        loaded.append(kwargs["dtype"])
        return Model(kwargs["dtype"])

    loader = SimpleNamespace(from_pretrained=load)
    monkeypatch.setattr(probe, "reference_loader", lambda *_: loader)
    args = SimpleNamespace(out=tmp_path / "diagnostic.json", smoke_report=source, threads=1)
    if fault != "clean":
        with pytest.raises(ValueError, match="byte-bound|changed during|quantized checkpoints"):
            run(args)
        assert not args.out.exists()
        return
    assert run(args)
    result = json.loads(args.out.read_text())
    assert result["diagnostic_only"] is True
    assert "passed" not in result and "policy" not in result
    assert result["checkpoint_binding"] == "matched-hcu-execution"
    assert len(result["rows"]) == 2
    assert result["reference"]["dtypes"] == ["float32", "bfloat16"]
    assert loaded == [torch.float32, torch.bfloat16]
    buffers = result["reference"]["floating_buffers"]
    assert buffers["torch.float32"]["inv_freq"] == buffers["torch.bfloat16"]["inv_freq"]
    assert buffers["torch.bfloat16"]["inv_freq"]["dtype"] == "torch.float32"
    assert calls == [(torch.float32, [[2, 3, 4, 5]]), (torch.bfloat16, [[2, 3, 4, 5]])]
