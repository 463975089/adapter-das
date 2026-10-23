import torch, json
import deepgemm
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)
K, N = 5120, 2304
root = "/models/dsv41f"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{root}/{wm['layers.6.ffn.experts.0.w1.weight']}", framework="pt") as sf:
    w_raw = sf.get_tensor("layers.6.ffn.experts.0.w1.weight").cuda().view(torch.uint8).contiguous()
with safe_open(f"{root}/{wm['layers.6.ffn.experts.0.w1.scale']}", framework="pt") as sf:
    sc = sf.get_tensor("layers.6.ffn.experts.0.w1.scale").cuda().view(torch.uint8).contiguous()

# extract kernel decode: one-hot chunks
W_dec = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
CH = 256
for s0 in range(0, K, CH):
    m = min(CH, K - s0)
    oh = torch.zeros(m, K, device="cuda", dtype=torch.bfloat16)
    oh[torch.arange(m), torch.arange(s0, s0 + m)] = 1.0
    o = torch.zeros(m, N, device="cuda", dtype=torch.bfloat16)
    deepgemm.m_grouped_wfp4a16_gemm_nt_contiguous(oh, (w_raw.unsqueeze(0), sc.unsqueeze(0)), o, torch.zeros(m, dtype=torch.int32, device="cuda"))
    W_dec[s0:s0 + m] = o
torch.cuda.synchronize()
print("W_dec extracted", W_dec.shape, flush=True)

# reference dequant matrix [K, N]
lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
lo_, hi_ = w_raw.long() & 0xF, w_raw.long() >> 4
W_ref = torch.stack([lut[lo_], lut[hi_]], -1).reshape(N, K).T  # [K, N]
e8 = torch.pow(2.0, sc.view(torch.uint8).long().float() - 127.0)  # [N, K/32]
W_ref = (W_ref * e8.repeat_interleave(32, 1).T).float()

A = F.normalize(W_dec.float(), dim=1)
B = F.normalize(W_ref, dim=1)
sim = A @ B.T  # [K, K]
perm = sim.argmax(dim=1)
conf = sim.max(dim=1).values
print(f"perm fit: mean conf={conf.mean().item():.4f} frac>0.99={float((conf > 0.99).float().mean()):.3f}", flush=True)
ok = (W_dec.float()[torch.arange(K)] - W_ref[perm]).abs().max().item()
print(f"check maxabs after perm (first rows): {(W_dec.float()[:8] - W_ref[perm[:8]]).abs().max().item():.5f}", flush=True)
# dump perm structure: does k map within locality?
loc = (perm - torch.arange(K, device="cuda")).abs()
print(f"displacement: max={int(loc.max())} p50={int(loc.float().median())}", flush=True)
torch.save(perm.cpu(), "/root/wfp4_perm.pt")
# print first 64 mappings compactly
p = perm[:64].cpu().tolist()
print("perm[0..63]:", p, flush=True)
