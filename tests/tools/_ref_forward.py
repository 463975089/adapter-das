# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference forward of GLM-5-W4A8_V2_6 on ONE GPU (TP1).

Streams weights from NFS per layer, dequant on the fly:
  int4-packed (packed K = true K/2) -> unpack * scale*16 ; W8 -> w8*scale ; bf16 raw.
MoE: adapter w4a8 kernel (same as server), sigmoid noaux_tc routing + shared expert.
Prints last-token top-8 logits; goal: compare vs server output.
"""

import json
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/adapter")

CKPT = "/public/share/GLM-5-W4A8_V2_6/"
PROMPT = "The capital of France is"

cfg = json.load(open(CKPT + "config.json"))
H = cfg["hidden_size"]
NHEAD = cfg["num_attention_heads"]
QK_HEAD = cfg["qk_head_dim"]
QK_NOPE = cfg["qk_nope_head_dim"]
QK_ROPE = cfg["qk_rope_head_dim"]
V_HEAD = cfg["v_head_dim"]
KV_LORA = cfg["kv_lora_rank"]
Q_LORA = cfg["q_lora_rank"]
NL = cfg["num_hidden_layers"]
NE = cfg["n_routed_experts"]
TOPK = cfg["num_experts_per_tok"]
IE = cfg["moe_intermediate_size"]
FIRST_DENSE = cfg["first_k_dense_replace"]
RS = cfg["routed_scaling_factor"]
EPS = cfg["rms_norm_eps"]
rope_theta = cfg["rope_parameters"]["rope_theta"]

wm = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]
_fh = {}


def read(key):
    path = CKPT + wm[key]
    if path not in _fh:
        fp = open(path, "rb")
        n = struct.unpack("<Q", fp.read(8))[0]
        _fh[path] = (fp, fp.tell() + n, json.loads(fp.read(n)))
    fp, base, hdr = _fh[path]
    v = hdr[key]
    s, e = v["data_offsets"]
    fp.seek(base + s)
    raw = fp.read(e - s)
    npdt = {"I8": np.int8, "F32": np.float32, "BF16": np.uint16, "U8": np.uint8}[
        v["dtype"]
    ]
    return torch.from_numpy(np.frombuffer(raw, dtype=npdt).reshape(v["shape"]).copy())


def unpack_signed(w):
    u = w.view(torch.uint8).to(dev)
    hi = (u >> 4).to(torch.int16)
    lo = (u & 0xF).to(torch.int16)
    hi = torch.where(hi >= 8, hi - 16, hi)
    lo = torch.where(lo >= 8, lo - 16, lo)
    return torch.stack((hi, lo), -1).reshape(w.shape[0], w.shape[1] * 2).to(torch.int8)


def load_linear(key, N, K):
    """Return fp32 GPU weight [N, K]. int4-packed iff stored K == true K/2."""
    w = read(key)
    assert w.shape[0] == N, (key, w.shape, N)
    if w.shape[1] == K:  # W8
        s = read(key + "_scale").float().to(dev).view(-1, 1)
        return w.to(dev).float() * s
    assert w.shape[1] * 2 == K, (key, w.shape, K)
    s = read(key + "_scale").float().to(dev).view(-1, 1) * 16.0
    return unpack_signed(w).float() * s


dev = "cuda:0"

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
ids = tok(PROMPT, return_tensors="pt").input_ids[0]
T = ids.shape[0]
print("tokens:", tok.convert_ids_to_tokens(ids))

pos = torch.arange(T)
inv = 1.0 / (rope_theta ** (torch.arange(0, QK_ROPE, 2).float() / QK_ROPE))
ANGLE = pos.float()[:, None] * inv[None, :]          # [T, 32]


def rope_inplace(t):  # t [.., NH, 64]
    c, s = ANGLE.cos().to(dev), ANGLE.sin().to(dev)
    x1, x2 = t[..., 0::2].float(), t[..., 1::2].float()
    y1 = x1 * c[None, :, None] - x2 * s[None, :, None]
    y2 = x1 * s[None, :, None] + x2 * c[None, :, None]
    o = t.clone()
    o[..., 0::2] = y1.to(t.dtype)
    o[..., 1::2] = y2.to(t.dtype)
    return o


def rms(x, w):
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), EPS)


def silu_mul(x, gw, uw):
    gw = gw.float() if gw.dtype != torch.float32 else gw
    uw = uw.float() if uw.dtype != torch.float32 else uw
    return (F.silu(x.float() @ gw.t()) * (x.float() @ uw.t())).to(torch.bfloat16)


from hcu_adapter.slimquant import fused_experts_w4a8

with torch.no_grad():
    # embed + lm_head are bf16
    emb = read("model.embed_tokens.weight").view(torch.bfloat16).to(dev)
    lm = read("lm_head.weight").view(torch.bfloat16).to(dev).float()
    hs = emb[ids].float().unsqueeze(0)                    # [1,T,H] fp32 flow

    for L in range(NL):
        p = f"model.layers.{L}"
        w = lambda n, N, K: load_linear(f"{p}.{n}", N, K)

        def bf(n):
            t = read(f"{p}.{n}")
            if t.dtype == torch.uint16:
                return t.view(torch.bfloat16).to(dev).float()
            return t.to(dev).float()  # F32 (e_score_correction_bias) stays fp32

        xa = rms(hs.view(-1, H), bf("input_layernorm.weight")).view(1, T, H)
        # MLA
        qa_ln, kva_ln = bf(f"self_attn.q_a_layernorm.weight"), bf(
            f"self_attn.kv_a_layernorm.weight")
        q_a = w("self_attn.q_a_proj.weight", Q_LORA, H)
        kv_a = w("self_attn.kv_a_proj_with_mqa.weight", KV_LORA + QK_ROPE, H)
        q_b = w("self_attn.q_b_proj.weight", NHEAD * QK_HEAD, Q_LORA)
        kv_b = w("self_attn.kv_b_proj.weight", NHEAD * (QK_NOPE + V_HEAD), KV_LORA)
        o_p = w("self_attn.o_proj.weight", H, NHEAD * V_HEAD)

        q_lat = rms(xa.view(-1, H) @ q_a.t(), qa_ln).to(torch.bfloat16)
        kva = xa.view(-1, H) @ kv_a.t()
        c_kv = rms(kva[:, :KV_LORA], kva_ln).to(torch.bfloat16)
        del kv_a, q_a

        q = (q_lat.float() @ q_b.t()).view(T, NHEAD, QK_HEAD)
        q_rope = rope_inplace(q[..., QK_NOPE:].contiguous())
        kva_u = c_kv.float() @ kv_b.t()
        kv = kva_u.view(T, NHEAD, QK_NOPE + V_HEAD)
        k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
        k_rope = kva[:, KV_LORA:].view(T, 1, QK_ROPE).expand(T, NHEAD, QK_ROPE)

        qk = torch.cat([q[..., :QK_NOPE].float(), q_rope.float()], -1)
        kk = torch.cat([k_nope.float(), k_rope.float()], -1)
        del kv_b, q_lat, q, kva_u, kva, q_rope, k_rope
        sc = torch.einsum("thd,shd->hts", qk, kk) / (QK_HEAD ** 0.5)
        msk = torch.full((T, T), float("-inf"), device=dev).triu(1)
        pr = (sc + msk).softmax(-1)
        att = (torch.einsum("hts,shd->thd", pr, v.float()).reshape(T, -1)
               @ o_p.float().t()).view(1, T, H)
        del qk, kk, sc, pr, k_nope, v, o_p
        hs = hs + att

        xb = rms(hs.view(-1, H), bf("post_attention_layernorm.weight")).view(-1, H)
        if L < FIRST_DENSE:
            # dense MLP: separate gate/up/down; fp32 weights used then freed
            gshape = read(f"{p}.mlp.gate_proj.weight").shape
            ID = gshape[0]
            gw = load_linear(f"{p}.mlp.gate_proj.weight", ID, H).to(torch.bfloat16)
            uw = load_linear(f"{p}.mlp.up_proj.weight", ID, H).to(torch.bfloat16)
            dw = load_linear(f"{p}.mlp.down_proj.weight", H, ID).to(torch.bfloat16)
            ff = (silu_mul(xb, gw, uw).float() @ dw.float().t()).view(1, T, H)
            del gw, uw, dw
            hs = hs + ff
        else:
            gate = bf("mlp.gate.weight")
            bias = bf("mlp.gate.e_score_correction_bias")
            rl = xb.float() @ gate.t()
            # noaux_tc: choose by sigmoid+bias; weight = raw sigmoid, renorm
            scores = rl.sigmoid()
            sfc = scores + bias
            _, ti = sfc.topk(TOPK, dim=-1)
            tw = scores.gather(1, ti)
            tw = tw / tw.sum(-1, keepdim=True)

            def stack(nm):
                ws = [read(f"model.layers.{L}.mlp.experts.{e}.{nm}.weight")
                      for e in range(NE)]
                return torch.stack(ws).contiguous()

            w13 = torch.cat([stack("gate_proj"), stack("up_proj")], 1).to(dev)
            w2 = stack("down_proj").to(dev)
            _s13g = torch.cat([
                read(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale"
                     ).float().view(-1) * 16.0 for e in range(NE)]).to(dev)
            _s13u = torch.cat([
                read(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight_scale"
                     ).float().view(-1) * 16.0 for e in range(NE)]).to(dev)
            _s2 = torch.cat([
                read(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight_scale"
                     ).float().view(-1) * 16.0 for e in range(NE)]).to(dev)
            moe = fused_experts_w4a8(
                xb.to(torch.bfloat16), w13, w2,
                torch.cat([_s13g, _s13u]).view(NE, 2 * IE).contiguous(),
                _s2.view(NE, H).contiguous(),
                tw, ti,
            ).float()
            del w13, w2
            torch.cuda.empty_cache()
            # shared expert (W8): compute then free
            se_g = w(f"mlp.shared_experts.gate_proj.weight", IE, H).to(torch.bfloat16)
            se_u = w(f"mlp.shared_experts.up_proj.weight", IE, H).to(torch.bfloat16)
            se_d = w(f"mlp.shared_experts.down_proj.weight", H, IE).to(torch.bfloat16)
            se = silu_mul(xb, se_g, se_u).float() @ se_d.float().t()
            del se_g, se_u, se_d
            hs = hs + (moe * RS + se).view(1, T, H)
            del se, moe, gate, _s13g, _s13u, _s2
        torch.cuda.empty_cache()

        if L % 10 == 0 or L == NL - 1:
            print(f"L{L}: hid-norm(last)={hs[0,-1].float().norm():.3f}")

    lnf = read("model.norm.weight").view(torch.bfloat16).to(dev).float()
    h_last = rms(hs[0, -1], lnf).to(torch.bfloat16)
    logits = h_last.float() @ lm.t()
    topv, topi = logits.topk(8)
    print("REFERENCE top8:",
          list(zip(tok.convert_ids_to_tokens(topi.tolist()),
                   [round(v, 2) for v in topv.tolist()])))
