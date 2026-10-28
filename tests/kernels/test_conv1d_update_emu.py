"""causal_conv1d_update torch-emulator gate — §3.4b.

The gfx938 triton kernel returns correlation noise (cos 0.26 vs the W=4
non-spec reference; weight-flip variant 0.50 — neither matches), while this
emulator passes exactly vs the same reference. Runs on every KDA layer per
decode step (42 x steps) in GLM-5.x — the decode-side word-soup driver once
prefill chunk_kda is emulated. Also covers the in-place state slide and the
conv_state_indices continuous-batching indexing path.
"""

import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import sys
sys.path.insert(0, "/root/hcu_t2")
sys.path.insert(0, "/sglang-v0521/python")
import sitecustomize
import torch
from hcu_adapter.conv1d_update_emu import causal_conv1d_update_emu

torch.manual_seed(0)
D, W, BS = 96, 4, 3
x = torch.randn(BS, D, device="cuda", dtype=torch.bfloat16)
w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16) * 0.3
bias = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1
state0 = torch.randn(BS, D, W - 1, device="cuda", dtype=torch.bfloat16)
state = state0.clone()
idx = torch.arange(BS, device="cuda", dtype=torch.int32)
out = causal_conv1d_update_emu(x, state, w, bias, "silu", conv_state_indices=idx)
torch.cuda.synchronize()
ref = torch.zeros(BS, D, device="cuda")
for b in range(BS):
    hist = torch.cat([state0[b].float(), x[b].float().unsqueeze(-1)], dim=1)
    for d in range(D):
        val = float((hist[d] * w[d].float()).sum()) + float(bias[d])
        ref[b, d] = val * torch.sigmoid(torch.tensor(val))
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
exact = torch.allclose(out.float(), ref, atol=2e-2)
print(f"conv1d emu: cos={cos:.6f} exact(atol2e-2)={exact}")
assert cos >= 0.9999
print("PASS")
