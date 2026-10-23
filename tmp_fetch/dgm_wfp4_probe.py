import torch, json, time
import deepgemm
from safetensors import safe_open

torch.manual_seed(0)
E_SUB, K, N = 32, 5120, 2304
root = "/models/dsv41f"
idx = json.load(open(f"{root}/model.safetensors.index.json"))
wm = idx["weight_map"]

# load experts 0..E_SUB-1 w1 (+scales) for layer 6
ws, wsc = [], []
for e in range(E_SUB):
    qw = torch.empty(0)
    for name, getter in ((f"layers.6.ffn.experts.{e}.w1.weight", "w"),
                         (f"layers.6.ffn.experts.{e}.w1.scale", "s")):
        f = wm[name]
        with safe_open(f"{root}/{f}", framework="pt") as sf:
            t = sf.get_tensor(name)
        ws.append(t) if getter == "w" else wsc.append(t)
w = torch.stack(ws).cuda().view(torch.uint8).contiguous()
wsc_t = torch.stack(wsc).cuda().view(torch.uint8).contiguous()
print("w", w.shape, w.dtype, "scale", wsc_t.shape, wsc_t.dtype, flush=True)

# torch dequant reference (e2m1 LUT x e8m0 group scale)
lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
def dequant(w_u8, sc_u8):
    E_, N_, Kp = w_u8.shape
    lo = (w_u8 & 0xF).long()
    hi = (w_u8 >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(E_, N_, Kp * 2)  # [E,N,K]
    e8m0 = torch.pow(2.0, sc_u8.view(torch.uint8).long().to(torch.float32) - 127.0)  # [E,N,K/32]
    e8m0 = e8m0.repeat_interleave(32, dim=2)
    return (vals * e8m0).to(torch.bfloat16)

w_ref = dequant(w, wsc_t)  # [E, N, K]
print("dequant done", w_ref.shape, w_ref.abs().mean().item(), flush=True)

# grouped contiguous layout: 64 tokens x topk6 = 384 assignments over E_SUB experts
M_rows, topk = 384, 6
assign = torch.stack([torch.randperm(E_SUB)[:topk] for _ in range(64)])
m_idx = torch.full((M_rows,), -1, dtype=torch.int32)
a = torch.randn(M_rows, K, device="cuda", dtype=torch.bfloat16) * 0.05
# group rows by expert (row order: expert-major), pad already exact via -1 fill
row = 0
rows_per_e = torch.bincount(assign.flatten(), minlength=E_SUB)
starts = torch.cumsum(rows_per_e, 0) - rows_per_e
fills = []
flat_e = assign.flatten()
flat_t = torch.arange(64).repeat_interleave(topk)
order = torch.argsort(flat_e, stable=True)
sorted_t = flat_t[order]
sorted_e = flat_e[order]
m_idx = torch.full((M_rows,), -1, dtype=torch.int32)
pos = 0
a_rows = torch.zeros(M_rows, K, device="cuda", dtype=torch.bfloat16)
# generate per-token activation once; duplicated rows per (token,expert)
a_tok = torch.randn(64, K, device="cuda", dtype=torch.bfloat16) * 0.05
for e in range(E_SUB):
    ids = sorted_t[sorted_e == e]
    if len(ids) == 0:
        continue
    m_idx[pos:pos + len(ids)] = e
    a_rows[pos:pos + len(ids)] = a_tok[ids]
    pos += len(ids)
print("layout rows used", pos, flush=True)

out = torch.zeros(M_rows, N, device="cuda", dtype=torch.bfloat16)
m_idx_d = m_idx.cuda()
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a_rows, (w, wsc_t), out, m_idx_d)
torch.cuda.synchronize()
print("kernel ran, nonzero frac", (out != 0).float().mean().item(), flush=True)

# reference rows
ref = torch.zeros(M_rows, N, device="cuda", dtype=torch.float32)
for e in range(E_SUB):
    m = m_idx == e
    if m.any():
        rows = m.nonzero().squeeze(1)
        ref[rows] = a_rows[rows].float() @ w_ref[e].float().T
d = (out.float() - ref.float()).abs()
denom = ref.float().abs().clamp_min(1e-3)
print(f"parity: maxabs={d.max().item():.5f} maxrel={(d/denom).max().item():.4f} refmax={ref.abs().max().item():.3f}", flush=True)

# --- binary search the convention: single expert, no padding ambiguity ---
E1, M1 = 1, 128
w1_ = w[:1].clone(); sc1_ = wsc_t[:1].clone()
wref1 = w_ref[:1]
a1 = torch.randn(M1, K, device="cuda", dtype=torch.bfloat16) * 0.05
mi1 = torch.zeros(M1, dtype=torch.int32, device="cuda")
out1 = torch.zeros(M1, N, device="cuda", dtype=torch.bfloat16)
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a1, (w1_, sc1_), out1, mi1)
torch.cuda.synchronize()
r1 = a1.float() @ wref1[0].float().T
def score(o, r):
    return (o.float() - r).abs().max().item(), float(torch.nn.functional.cosine_similarity(o.float().flatten().unsqueeze(0), r.flatten().unsqueeze(0)).item())
print("single-expert as-is:", score(out1, r1), flush=True)
# hi/lo nibble swap reference
lo = (w1_ & 0xF).long(); hi = (w1_ >> 4).long()
vals_sw = torch.stack([lut[hi], lut[lo]], dim=-1).reshape(1, N, K)
e8 = torch.pow(2.0, sc1_.long().to(torch.float32) - 127.0).repeat_interleave(32, dim=2)
wref_sw = (vals_sw * e8).to(torch.bfloat16)
r_sw = a1.float() @ wref_sw[0].float().T
print("single-expert nibble-swapped:", score(out1, r_sw), flush=True)
# bias 64
e8b = torch.pow(2.0, sc1_.long().to(torch.float32) - 64.0).repeat_interleave(32, dim=2)
wref_b = (vals_sw * e8b).to(torch.bfloat16)
print("nibble-swap + bias64:", score(out1, a1.float() @ wref_b[0].float().T), flush=True)
vals_orig = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(1, N, K)
wref_b2 = (vals_orig * e8b).to(torch.bfloat16)
print("orig-nibble + bias64:", score(out1, a1.float() @ wref_b2[0].float().T), flush=True)
# one-hot probe: a1[m] = e_m  -> out1[m, n] = W_dec[m, n] (scaled by nothing)
KS = 16
oh = torch.zeros(KS, K, device="cuda", dtype=torch.bfloat16)
for m in range(KS):
    oh[m, m] = 1.0
mi_oh = torch.zeros(KS, dtype=torch.int32, device="cuda")
out_oh = torch.zeros(KS, N, device="cuda", dtype=torch.bfloat16)
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(oh, (w1_, sc1_), out_oh, mi_oh)
torch.cuda.synchronize()
# my reference decode for first KS k of expert0 row n=0..7
sc_e0 = sc1_[0].long().float()  # [N, 160]
w_e0 = w1_[0]  # [N, 2560] uint8
NS = 64
lo = (w_e0[:NS, :16] & 0xF).long()   # [NS,16] bytes -> k0..31 lo/hi
hi = (w_e0[:NS, :16] >> 4).long()
sc0 = torch.pow(2.0, sc_e0[:NS, 0] - 127.0)  # [NS] group0 scale
kern = out_oh[:KS, :NS].float()            # [KS(16), NS] kernel decode (k in 0..15)
import torch.nn.functional as F
for k in range(16):
    byte_i, half = k // 2, k % 2
    nib = (lo if half == 0 else hi)[:, byte_i].long()
    mine_k = lut[nib] * sc0  # [NS]
    c = F.cosine_similarity(kern[k], mine_k, dim=0).item()
    # also try matching against ALL kernel rows to find permutation
    row = kern[k]
    best = max(range(16), key=lambda j: F.cosine_similarity(row, lut[(lo if j % 2 == 0 else hi)[:, j // 2].long()] * sc0, dim=0).item())
    cb = F.cosine_similarity(row, lut[(lo if best % 2 == 0 else hi)[:, best // 2].long()] * sc0, dim=0).item()
    print(f"k={k:2d}: cos(mine)={c:+.3f}  best_match=k'{best} cos={cb:+.3f}")
import sys; sys.exit(0)

# timing via graph replay
def call():
    deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a_rows, (w, wsc_t), out, m_idx_d)
for _ in range(3):
    call()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    call()
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
s, ev = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(100):
    g.replay()
ev.record(); torch.cuda.synchronize()
print(f"wfp4a16 contiguous: {s.elapsed_time(ev) * 1000 / 100:.1f} us (M{M_rows} K{K} N{N} E{E_SUB})", flush=True)
