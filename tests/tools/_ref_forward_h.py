"""Reference forward with per-layer hidden-state dump (for server bisect).

NL_OVERWRITE env: truncate num_hidden_layers (user-endorsed fast iteration).
Saves list of last-token hidden states [L0..L{NL-1}, final_normed] to
/adapter/tests/tools/_ref_hiddens.pt and prints top8 logits.
Rest identical to _ref_forward.py.
"""
import json
import os
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
NL = int(os.environ.get("NL_OVERWRITE", cfg["num_hidden_layers"]))
NE = cfg["n_routed_experts"]
TOPK = cfg["num_experts_per_tok"]
IE = cfg["moe_intermediate_size"]
FIRST_DENSE = cfg["first_k_dense_replace"]
RS = cfg["routed_scaling_factor"]
EPS = cfg["rms_norm_eps"]
rope_theta = cfg["rope_parameters"]["rope_theta"]
print(f"NL={NL} (cfg {cfg['num_hidden_layers']})")

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
    w = read(key)
    assert w.shape[0] == N, (key, w.shape, N)
    if w.shape[1] == K:  # W8
        s = read(key + "_scale").float().to(dev).view(-1, 1)
        return w.to(dev).float() * s
    assert w.shape[1] * 2 == K, (key, w.shape, K)
    s = read(key + "_scale").float().to(dev).view(-1, 1) * 16.0
    return unpack_signed(w).float() * s


def bf(n):
    t = read(f"{p}.{n}")
    if t.dtype == torch.float32:
        return t.to(dev).float()
    return t.view(torch.bfloat16).to(dev).float()


dev = "cuda:0"

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
_ids_env = os.environ.get("REF_IDS")
PROMPT = os.environ.get("REF_PROMPT", PROMPT)
if _ids_env:
    ids = torch.tensor([int(v) for v in _ids_env.split(",")], dtype=torch.long)
    print("ids from REF_IDS:", ids.tolist())
else:
    ids = tok(PROMPT, return_tensors="pt").input_ids[0]
T = ids.shape[0]
print("tokens:", tok.convert_ids_to_tokens(ids))

pos = torch.arange(T)
inv = 1.0 / (rope_theta ** (torch.arange(0, QK_ROPE, 2).float() / QK_ROPE))
ANGLE = pos.float()[:, None] * inv[None, :]


def rope_inplace(t):
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

hiddens = []  # last-token hidden state after each block (input of next)

with torch.no_grad():
    emb = read("model.embed_tokens.weight").view(torch.bfloat16).to(dev)
    lm = read("lm_head.weight").view(torch.bfloat16).to(dev).float()
    hs = emb[ids].float().unsqueeze(0)
    hiddens.append(hs[0, -1].cpu())

    for L in range(NL):
        p = f"model.layers.{L}"
        w = lambda n, N, K: load_linear(f"{p}.{n}", N, K)
        bfl = lambda n: bf(n)

        xa = rms(hs.view(-1, H), bfl("input_layernorm.weight")).view(1, T, H)
        qa_ln, kva_ln = bfl("self_attn.q_a_layernorm.weight"), bfl(
            "self_attn.kv_a_layernorm.weight")
        q_a = w("self_attn.q_a_proj.weight", Q_LORA, H)
        kv_a = w("self_attn.kv_a_proj_with_mqa.weight", KV_LORA + QK_ROPE, H)
        q_b = w("self_attn.q_b_proj.weight", NHEAD * QK_HEAD, Q_LORA)
        kv_b = w("self_attn.kv_b_proj.weight", NHEAD * (QK_NOPE + V_HEAD), KV_LORA)
        o_p = w("self_attn.o_proj.weight", H, NHEAD * V_HEAD)

        q_lat = rms(xa.view(-1, H) @ q_a.t(), qa_ln).to(torch.bfloat16)
        kva = xa.view(-1, H) @ kv_a.t()
        c_kv = rms(kva[:, :KV_LORA], kva_ln).to(torch.bfloat16)

        q = (q_lat.float() @ q_b.t()).view(T, NHEAD, QK_HEAD)
        q_rope = rope_inplace(q[..., QK_NOPE:].contiguous())
        kva_u = c_kv.float() @ kv_b.t()
        kv = kva_u.view(T, NHEAD, QK_NOPE + V_HEAD)
        k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
        k_rope = kva[:, KV_LORA:].view(T, 1, QK_ROPE).expand(T, NHEAD, QK_ROPE)

        qk = torch.cat([q[..., :QK_NOPE].float(), q_rope.float()], -1)
        kk = torch.cat([k_nope.float(), k_rope.float()], -1)
        sc = torch.einsum("thd,shd->hts", qk, kk) / (QK_HEAD ** 0.5)
        msk = torch.full((T, T), float("-inf"), device=dev).triu(1)
        pr = (sc + msk).softmax(-1)
        att = (torch.einsum("hts,shd->thd", pr, v.float()).reshape(T, -1)
               @ o_p.t()).view(1, T, H)
        hs = hs + att
        if L == 0 and os.environ.get("REF_DUMP_INNER") == "1":
            torch.save(
                {
                    "attn0": att[0, -1].float().cpu(),
                    "xa0": xa[0, -1].float().cpu(),
                    "q0": q[-1].float().cpu(),
                    "k0": kk[-1].float().cpu(),
                    "v0": v[-1].float().cpu(),
                    "pr0": pr[:, :, -1].float().cpu(),
                },
                "/adapter/tests/tools/_ref_inner.pt",
            )
            print("saved _ref_inner.pt")

        xb = rms(hs.view(-1, H), bfl("post_attention_layernorm.weight")).view(-1, H)
        if L < FIRST_DENSE:
            gshape = read(f"{p}.mlp.gate_proj.weight").shape
            ID = gshape[0]
            gw = load_linear(f"{p}.mlp.gate_proj.weight", ID, H)
            uw = load_linear(f"{p}.mlp.up_proj.weight", ID, H)
            dw = load_linear(f"{p}.mlp.down_proj.weight", H, ID)
            ff = (silu_mul(xb, gw, uw).float() @ dw.t()).view(1, T, H)
            hs = hs + ff
            if L == 0 and os.environ.get("REF_DUMP_INNER") == "1":
                d = torch.load("/adapter/tests/tools/_ref_inner.pt")
                d["mlp0"] = ff[0, -1].float().cpu()
                d["xb0"] = xb[-1].float().cpu()
                torch.save(d, "/adapter/tests/tools/_ref_inner.pt")
        else:
            gate = bfl("mlp.gate.weight")
            rl = xb.float() @ gate.t()
            tw, ti = rl.sigmoid().topk(TOPK, dim=-1)
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
            se_g = w(f"mlp.shared_experts.gate_proj.weight", IE, H)
            se_u = w(f"mlp.shared_experts.up_proj.weight", IE, H)
            se_d = w(f"mlp.shared_experts.down_proj.weight", H, IE)
            se = silu_mul(xb, se_g, se_u).float() @ se_d.t()
            hs = hs + (moe * RS + se).view(1, T, H)

        hiddens.append(hs[0, -1].cpu())
        print(f"L{L}: hid-norm(last)={hs[0,-1].float().norm():.3f}")

    lnf = read("model.norm.weight").view(torch.bfloat16).to(dev).float()
    h_last = rms(hs[0, -1], lnf).to(torch.bfloat16)
    hiddens.append(h_last.cpu())
    logits = h_last.float() @ lm.t()
    topv, topi = logits.topk(8)
    print("REFERENCE top8:",
          list(zip(tok.convert_ids_to_tokens(topi.tolist()),
                   [round(v, 2) for v in topv.tolist()])))

torch.save({"hiddens": hiddens, "ids": ids.tolist(), "NL": NL},
           "/adapter/tests/tools/_ref_hiddens.pt")
print("saved _ref_hiddens.pt", len(hiddens), "entries")
