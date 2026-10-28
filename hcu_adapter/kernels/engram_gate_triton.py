# SPDX-License-Identifier: Apache-2.0
"""SGLang HCU Engram inference gate, with DeepInfer's pre-fused FP32 weight.

Adapted from HYGON-AI/sglang-das at 7b91060ae3ef5e112048988da6f4f673cf4b6a89,
python/sglang/kernels/ops/embeddings/engram_gate.py. Arithmetic, reduction order,
launch geometry and disabled FMA match that source. The interface additionally
supports upstream DeepInfer's strides, image mask and in-place output.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_CALLS = 0


def usage():
    return {'sglang_fused_inference': _CALLS}


@triton.jit
def _gate(X, KV, W, IMAGE, O,
          XT: tl.constexpr, XH: tl.constexpr, XD: tl.constexpr,
          KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
          WH: tl.constexpr, WD: tl.constexpr,
          OT: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
          D: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr,
          CLAMP: tl.constexpr, HAS_IMAGE: tl.constexpr, IMAGE_STRIDE: tl.constexpr,
          B: tl.constexpr):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + token * XT + hc * XH + col * XD, mask, 0).to(tl.float32)
    key = tl.load(KV + token * KT + hc * KH + col * KD, mask, 0).to(tl.float32)
    # DeepInfer computes q_weight.float() * k_weight.float() once at loading.
    weight = tl.load(W + hc * WH + col * WD, mask, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS)
    dot = tl.sum((x * weight) * key, 0) * rstd * (D ** -0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * KT + HC * KH + col * KD, mask, 0).to(tl.float32)
    result = x + gate * value
    if HAS_IMAGE:
        result = tl.where(tl.load(IMAGE + token * IMAGE_STRIDE), x, result)
    tl.store(O + token * OT + hc * OH + col * OD, result, mask)


def inference_gate(hidden_states, kv, weight_fused, eps, clamp_value,
                   image_token_mask=None, inplace=False):
    """Called after the public Engram provider validates tensor contracts."""
    global _CALLS
    tokens, hc, width = hidden_states.shape
    output = hidden_states if inplace else torch.empty_like(hidden_states)
    if tokens:
        _gate[(tokens * hc,)](
            hidden_states, kv, weight_fused,
            hidden_states if image_token_mask is None else image_token_mask, output,
            *hidden_states.stride(), *kv.stride(), *weight_fused.stride(),
            *output.stride(), width, hc, eps, clamp_value,
            image_token_mask is not None,
            0 if image_token_mask is None else image_token_mask.stride(0),
            triton.next_power_of_2(width),
            num_warps=4, enable_fp_fusion=False)
    _CALLS += 1
    return output
