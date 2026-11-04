# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Greedy 12-token generation with the pure-torch reference forward.
Reuses _ref_forward.py's machinery by importing it in a controlled way.
"""

import sys
import torch

# import _ref_forward as a module: it executes to `lnf = read(...)` with ids/hs
# in _ref_forward; simpler to duplicate the forward as a function here.

import importlib.util

spec = importlib.util.spec_from_file_location("_rf", "/adapter/tests/tools/_ref_forward.py")
_rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_rf)  # runs the single prompt; prints REFERENCE top8

read = _rf.read
load_linear = _rf.load_linear
rms = _rf.rms
silu_mul = _rf.silu_mul
rope_inplace = _rf.rope_inplace
fused_experts_w4a8 = _rf.fused_experts_w4a8
dev = _rf.dev
tok = _rf.tok
H, NHEAD, QK_HEAD = _rf.H, _rf.NHEAD, _rf.QK_HEAD
QK_NOPE, QK_ROPE, V_HEAD = _rf.QK_NOPE, _rf.QK_ROPE, _rf.V_HEAD
KV_LORA, Q_LORA = _rf.KV_LORA, _rf.Q_LORA
NL, NE, TOPK, IE = _rf.NL, _rf.NE, _rf.TOPK, _rf.IE
FIRST_DENSE, RS, EPS = _rf.FIRST_DENSE, _rf.RS, _rf.EPS
rope_theta = _rf.rope_theta
emb, lm = _rf.emb, _rf.lm

cur = _rf.ids.clone()

with torch.no_grad():
    for step in range(12):
        hs = emb[cur].float().unsqueeze(0)
        Tc = cur.shape[0]
        pos = torch.arange(Tc)
        inv = 1.0 / (rope_theta ** (torch.arange(0, QK_ROPE, 2).float() / QK_ROPE))
        ANGLE = pos.float()[:, None] * inv[None, :]

        def rope_inplace2(t):
            c, s = ANGLE.cos().to(dev), ANGLE.sin().to(dev)
            x1, x2 = t[..., 0::2].float(), t[..., 1::2].float()
            y1 = x1 * c[None, :, None] - x2 * s[None, :, None]
            y2 = x1 * s[None, :, None] + x2 * c[None, :, None]
            o = t.clone()
            o[..., 0::2] = y1.to(t.dtype)
            o[..., 1::2] = y2.to(t.dtype)
            return o

        for L in range(NL):
            p = f"model.layers.{L}"
            w = lambda n, N, K: load_linear(f"{p}.{n}", N, K)

            def bf(n):
                t = read(f"{p}.{n}")
                if t.dtype == torch.uint16:
                    return t.view(torch.bfloat16).to(dev).float()
                return t.to(dev).float()

            xa = rms(hs.view(-1, H), bf("input_layernorm.weight")).view(1, Tc, H)
            qa_ln, kva_ln = bf("self_attn.q_a_layernorm.weight"), bf(
                "self_attn.kv_a_layernorm.weight")
            q_a = w("self_attn.q_a_proj.weight", Q_LORA, H)
            kv_a = w("self_attn.kv_a_proj_with_mqa.weight", KV_LORA + QK_ROPE, H)
            q_b = w("self_attn.q_b_proj.weight", NHEAD * QK_HEAD, Q_LORA)
            kv_b = w("self_attn.kv_b_proj.weight", NHEAD * (QK_NOPE + V_HEAD), KV_LORA)
            o_p = w("self_attn.o_proj.weight", H, NHEAD * V_HEAD)

            q_lat = rms(xa.view(-1, H) @ q_a.t(), qa_ln).to(torch.bfloat16)
            kva = xa.view(-1, H) @ kv_a.t()
            c_kv = rms(kva[:, :KV_LORA], kva_ln).to(torch.bfloat16)
            del q_a, kv_a

            q = (q_lat.float() @ q_b.t()).view(Tc, NHEAD, QK_HEAD)
            q_rope = rope_inplace2(q[..., QK_NOPE:].contiguous())
            kva_u = c_kv.float() @ kv_b.t()
            kv = kva_u.view(Tc, NHEAD, QK_NOPE + V_HEAD)
            k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
            k_rope = kva[:, KV_LORA:].view(Tc, 1, QK_ROPE).expand(
                Tc, NHEAD, QK_ROPE)
            qk = torch.cat([q[..., :QK_NOPE].float(), q_rope.float()], -1)
            kk = torch.cat([k_nope.float(), k_rope.float()], -1)
            del kv_b, q_lat, q, kva_u, kva, q_rope, k_rope

            sc = torch.einsum("thd,shd->hts", qk, kk) / (QK_HEAD ** 0.5)
            msk = torch.full((Tc, Tc), float("-inf"), device=dev).triu(1)
            pr = (sc + msk).softmax(-1)
            att = (torch.einsum("hts,shd->thd", pr, v.float()).reshape(Tc, -1)
                   @ o_p.float().t()).view(1, Tc, H)
            del qk, kk, sc, pr, k_nope, v, o_p
            hs = hs + att

            xb = rms(hs.view(-1, H), bf("post_attention_layernorm.weight")
                     ).view(-1, H)
            if L < FIRST_DENSE:
                gshape = read(f"{p}.mlp.gate_proj.weight").shape
                ID = gshape[0]
                gw = load_linear(f"{p}.mlp.gate_proj.weight", ID, H).to(torch.bfloat16)
                uw = load_linear(f"{p}.mlp.up_proj.weight", ID, H).to(torch.bfloat16)
                dw = load_linear(f"{p}.mlp.down_proj.weight", H, ID).to(torch.bfloat16)
                ff = (silu_mul(xb, gw, uw).float() @ dw.float().t()).view(1, Tc, H)
                del gw, uw, dw
                hs = hs + ff
            else:
                gate, bias = bf("mlp.gate.weight"), bf(
                    "mlp.gate.e_score_correction_bias")
                scores = (xb.float() @ gate.t()).sigmoid()
                _, ti = (scores + bias).topk(TOPK, dim=-1)
                tw = scores.gather(1, ti)
                tw = tw / tw.sum(-1, keepdim=True)

                def stack(nm):
                    ws = [read(f"model.layers.{L}.mlp.experts.{e}.{nm}.weight")
                          for e in range(NE)]
                    return torch.stack(ws).contiguous()

                w13 = torch.cat([stack("gate_proj"), stack("up_proj")], 1).to(dev)
                w2 = stack("down_proj").to(dev)

                def scales(nm):
                    return torch.cat([
                        read(f"model.layers.{L}.mlp.experts.{e}.{nm}.weight_scale")
                        .float().view(-1) * 16.0 for e in range(NE)]).to(dev)

                sg, su, sd = scales("gate_proj"), scales("up_proj"), scales(
                    "down_proj")
                moe = fused_experts_w4a8(
                    xb.to(torch.bfloat16), w13, w2,
                    torch.cat([sg, su]).view(NE, 2 * IE).contiguous(),
                    sd.view(NE, H).contiguous(), tw, ti,
                ).float()
                del w13, w2
                torch.cuda.empty_cache()
                se_g = w("mlp.shared_experts.gate_proj.weight", IE, H).to(
                    torch.bfloat16)
                se_u = w("mlp.shared_experts.up_proj.weight", IE, H).to(
                    torch.bfloat16)
                se_d = w("mlp.shared_experts.down_proj.weight", H, IE).to(
                    torch.bfloat16)
                se = silu_mul(xb, se_g, se_u).float() @ se_d.float().t()
                del se_g, se_u, se_d
                hs = hs + (moe * RS + se).view(1, Tc, H)
                del se, moe, gate, sg, su, sd
            torch.cuda.empty_cache()

            # rebind global ANGLE for next step's rope (module-level helper used
            # by rope closure below next iteration)
            _rf.ANGLE = ANGLE

        lnf_ = read("model.norm.weight").view(torch.bfloat16).to(dev).float()
        h_last = rms(hs[0, -1], lnf_).to(torch.bfloat16)
        logits = h_last.float() @ lm.t()
        nxt = logits.argmax().item()
        cur = torch.cat([cur, torch.tensor([nxt])])
        print("gen:", repr(tok.decode(cur.tolist())))
