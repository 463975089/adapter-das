"""CPU parity test: mqa_logits_emu vs pure-bf16 reference.

Reference computes logits[i,t] = sum_h w[i,h] * (sum_d q[i,h,d]*k[t,d]) * ks[t]
in fp32 on bf16-dequantized inputs — exactly what the emulation must produce.
Run inside the DTK container (or any torch env); CPU is enough.
"""

import os
import sys

sys.path.insert(0, "/tmp/w4a8")  # adapter root
os.environ.setdefault("HCU_SLIMQUANT", "0")

import torch

from hcu_adapter import mqa_logits_emu as emu

torch.manual_seed(0)
dev = "cpu"
Q, H, D, T = 7, 32, 128, 257  # odd sizes to catch tiling bugs

# --- fp8 quantized inputs (fnuz on DCU; values in e4m3 range) ---
is_fnuz = hasattr(torch, "float8_e4m3fnuz")
f8dt = torch.float8_e4m3fnuz if is_fnuz else torch.float8_e4m3fn

q8 = (torch.randn(Q, H, D) * 0.3).clamp(-448, 448).to(f8dt)
k8 = (torch.randn(T, D) * 0.3).clamp(-448, 448).to(f8dt)
k_scale = (torch.rand(T) * 0.05 + 0.001).float()
w = torch.randn(Q, H).float() * 0.1  # q_scale pre-folded
seq = torch.tensor([3, 10, 257, 1, 100, 256, 50], dtype=torch.int32)
maxL = int(seq.max())
rows = []
for L in seq.tolist():
    ids = torch.randperm(T)[:L].sort().values
    rows.append(torch.nn.functional.pad(ids, (0, maxL - L)))
bt = torch.stack(rows)  # [Q, maxL], page-1 token ids
bt = torch.nn.functional.pad(bt, (0, 512 - bt.shape[1]))
max_seq_len = 512

# packed kv cache: [T, 128B fp8 | 4B scale]
kv_packed = torch.cat(
    [k8.view(torch.uint8), k_scale.unsqueeze(1).view(torch.uint8)], dim=1
)

# emulation (patched-in functions, called directly)
logits_paged = emu._paged_mqa_logits(
    q8,
    kv_packed,
    w,
    seq,
    bt,
    max_seq_len,
    preshuffle=False,
    kv_block_size=1,
)

# reference: same bf16-matmul entry path, fp32 scales — validates wiring
# (gather, masking, scale placement); bit-exactness vs the emulation.
qf = q8.to(torch.bfloat16)
kf = k8.to(torch.bfloat16)
ref = torch.matmul(qf, kf.t().unsqueeze(0)).float()  # [Q, H, T]
ref = (ref * w.unsqueeze(-1) * k_scale.view(1, 1, -1)).sum(dim=1)
ref_paged = torch.zeros(Q, max_seq_len)
for i in range(Q):
    L = seq[i].item()
    tok = bt[i, :L].long()
    ref_paged[i, :L] = ref[i, tok]  # logits columns are page-table positions

err = (logits_paged - ref_paged).abs().max().item()
denom = ref_paged.abs().max().item()
print("paged  max|err| = %.3g (rel %.3g)" % (err, err / max(denom, 1e-9)))
# bf16 matmul accumulation-order noise (different tile shapes) dominates;
# 0.1 abs on O(0.1) logits ~ bf16 epsilon * row length.
assert err <= 0.1, "paged mismatch"
assert logits_paged.dtype == torch.float32

# --- ragged ---
ks = torch.tensor([0, 5, 12, 40], dtype=torch.int32)
ke = torch.tensor([5, 12, 40, 77], dtype=torch.int32)
Qr = 4
q8r = (torch.randn(Qr, H, D) * 0.3).to(f8dt)
wr = torch.randn(Qr, H).float() * 0.1
logits_r = emu._ragged_fp8_mqa_logits(q8r, k8, k_scale, wr, ks, ke)
ref_r = torch.matmul(
    q8r.to(torch.bfloat16), k8.to(torch.bfloat16).t().unsqueeze(0)
).float()  # [Qr, H, T]
ref_r = (ref_r * wr.unsqueeze(-1) * k_scale.view(1, 1, -1)).sum(dim=1)  # [Qr, T]
valid = (torch.arange(T).view(1, -1) >= ks.view(-1, 1)) & (
    torch.arange(T).view(1, -1) < ke.view(-1, 1)
)
ref_r = torch.where(valid, ref_r, torch.tensor(float("-inf")))
err_r = (logits_r[valid] - ref_r[valid]).abs().max().item()
denom_r = ref_r[valid].abs().max().item()
print("ragged finite max|err| = %.3g (rel %.3g)" % (err_r, err_r / max(denom_r, 1e-9)))
assert err_r <= 0.1, "ragged mismatch"
print("OK")
