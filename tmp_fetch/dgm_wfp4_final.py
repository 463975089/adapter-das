import torch, json
import deepgemm
from safetensors import safe_open

torch.manual_seed(0)
E, K, N = 384, 5120, 2304
root = "/models/dsv41f"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
ws, wsc = [], []
for e in range(E):
    for name, dst in ((f"layers.6.ffn.experts.{e}.w1.weight", ws),
                      (f"layers.6.ffn.experts.{e}.w1.scale", wsc)):
        with safe_open(f"{root}/{wm[name]}", framework="pt") as sf:
            dst.append(sf.get_tensor(name))
w_raw = torch.stack(ws).cuda().view(torch.uint8).contiguous()
wsc_t = torch.stack(wsc).cuda().view(torch.uint8).contiguous()

b = w_raw.reshape(-1, 4)
lo, hi = b & 0xF, b >> 4
w_k = torch.stack((lo[:, 0] | (lo[:, 2] << 4), hi[:, 0] | (hi[:, 2] << 4),
                   lo[:, 1] | (lo[:, 3] << 4), hi[:, 1] | (hi[:, 3] << 4)), dim=1).reshape(E, N, K // 2).contiguous()
del b, lo, hi
print("loaded+repacked", w_k.shape, flush=True)

lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
def dequant_ref(e):
    w0 = w_raw[e].long()
    vals = torch.stack([lut[w0 & 0xF], lut[w0 >> 4]], -1).reshape(N, K)
    e8 = torch.pow(2.0, wsc_t[e].view(torch.uint8).long().float() - 127.0).repeat_interleave(32, 1)
    return (vals * e8).to(torch.bfloat16)

def run_case(tag, ntok, E_act, ALIGN):
    topk = 6
    a_tok = torch.randn(ntok, K, device="cuda", dtype=torch.bfloat16) * 0.05
    assign = torch.stack([torch.randperm(E_act)[:topk] for _ in range(ntok)])
    flat_e = assign.flatten()
    flat_t = torch.arange(ntok).repeat_interleave(topk)
    order = torch.argsort(flat_e, stable=True)
    st_, se_ = flat_t[order], flat_e[order]
    groups = [(e, st_[se_ == e]) for e in range(E_act)]
    groups = [(e, ids) for e, ids in groups if len(ids)]
    npad = sum(int(-(-len(ids) // ALIGN)) * ALIGN for _, ids in groups)
    m_idx = torch.full((npad,), -1, dtype=torch.int32)
    a_rows = torch.zeros(npad, K, device="cuda", dtype=torch.bfloat16)
    row_of = {}
    pos = 0
    for e, ids in groups:
        m_idx[pos:pos + len(ids)] = e
        a_rows[pos:pos + len(ids)] = a_tok[ids]
        row_of[e] = (pos, pos + len(ids))
        pos += int(-(-len(ids) // ALIGN)) * ALIGN
    out = torch.zeros(npad, N, device="cuda", dtype=torch.bfloat16)
    mi = m_idx.cuda()
    deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a_rows, (w_k, wsc_t), out, mi)
    torch.cuda.synchronize()
    dmax, rmax = 0.0, 1e-9
    for e, _ in groups[:8]:
        p0, p1 = row_of[e]
        rows = torch.arange(p0, p1, device="cuda")
        r = a_rows[rows].float() @ dequant_ref(e).float().T
        dmax = max(dmax, (out[rows].float() - r).abs().max().item())
        rmax = max(rmax, r.abs().max().item())
    def call():
        deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a_rows, (w_k, wsc_t), out, mi)
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
    for _ in range(50):
        g.replay()
    ev.record(); torch.cuda.synchronize()
    us = s.elapsed_time(ev) * 1000 / 50
    print(f"{tag} align={ALIGN} rows={npad}: maxabs={dmax:.4f} refmax={rmax:.3f} rel={dmax/rmax:.4f} | {us:.1f} us", flush=True)

for al in (32, 64, 128):
    run_case("decode B64x6", 64, E, al)
run_case("prefill B512x6", 512, E, 128)
run_case("prefill B512x6", 512, E, 32)
print("done", flush=True)
