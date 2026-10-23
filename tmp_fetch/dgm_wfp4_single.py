import torch, json
import deepgemm
from safetensors import safe_open

torch.manual_seed(0)
K, N = 5120, 2304
root = "/models/dsv41f"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{root}/{wm['layers.6.ffn.experts.0.w1.weight']}", framework="pt") as sf:
    w_raw = sf.get_tensor("layers.6.ffn.experts.0.w1.weight").cuda().view(torch.uint8).contiguous()
with safe_open(f"{root}/{wm['layers.6.ffn.experts.0.w1.scale']}", framework="pt") as sf:
    sc = sf.get_tensor("layers.6.ffn.experts.0.w1.scale").cuda().view(torch.uint8).contiguous()

def repack(w):
    b = w.reshape(-1, 4)
    lo, hi = b & 0xF, b >> 4
    return torch.stack((lo[:, 0] | (lo[:, 1] << 4), lo[:, 2] | (lo[:, 3] << 4),
                        hi[:, 0] | (hi[:, 1] << 4), hi[:, 2] | (hi[:, 3] << 4)), dim=1).reshape(w.shape).contiguous()

lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
def ref_dequant(w, s):
    lo_, hi_ = w.long() & 0xF, w.long() >> 4
    vals = torch.stack([lut[lo_], lut[hi_]], -1).reshape(N, K)
    e8 = torch.pow(2.0, s.view(torch.uint8).long().float() - 127.0).repeat_interleave(32, 1)
    return (vals * e8).to(torch.bfloat16)

M = 128
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
r = a.float() @ ref_dequant(w_raw, sc).float().T

w_k = repack(w_raw)
out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a, (w_k.unsqueeze(0), sc.unsqueeze(0)), out, torch.zeros(M, dtype=torch.int32, device="cuda"))
torch.cuda.synchronize()
import torch.nn.functional as F
d = (out.float() - r).abs()
print(f"repacked single-expert: maxabs={d.max().item():.4f} refmax={r.abs().max().item():.3f} cos={F.cosine_similarity(out.float().flatten().unsqueeze(0), r.flatten().unsqueeze(0)).item():.4f}", flush=True)

# also raw for reference
out2 = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(a, (w_raw.unsqueeze(0), sc.unsqueeze(0)), out2, torch.zeros(M, dtype=torch.int32, device="cuda"))
torch.cuda.synchronize()
print(f"raw single-expert: cos={F.cosine_similarity(out2.float().flatten().unsqueeze(0), r.flatten().unsqueeze(0)).item():.4f}", flush=True)

# one-hot over k=0..31 to re-verify permutation on THIS expert (full group0)
oh = torch.zeros(32, K, device="cuda", dtype=torch.bfloat16)
for m in range(32):
    oh[m, m] = 1.0
o3 = torch.zeros(32, N, device="cuda", dtype=torch.bfloat16)
deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(oh, (w_k.unsqueeze(0), sc.unsqueeze(0)), o3, torch.zeros(32, dtype=torch.int32, device="cuda"))
torch.cuda.synchronize()
sc0 = torch.pow(2.0, sc[:64, 0].long().float() - 127.0)
braw = w_raw[:64, :16].long()
for k in range(32):
    byte_i, half = k // 2, k % 2
    nib = (braw[:, byte_i] & 0xF if half == 0 else braw[:, byte_i] >> 4).float()
    mine = lut[(braw[:, byte_i] & 0xF if half == 0 else braw[:, byte_i] >> 4)] * sc0
    best = max(range(32), key=lambda j: F.cosine_similarity(o3[j, :64].float(), lut[(braw[:, j // 2] & 0xF if j % 2 == 0 else braw[:, j // 2] >> 4)] * sc0, dim=0).item())
    cb = F.cosine_similarity(o3[k, :64].float(), mine.float(), dim=0).item()
    if k < 8 or abs(cb) < 0.99:
        print(f"k={k:2d} cos(mine)={cb:+.3f} best=k'{best}", flush=True)
