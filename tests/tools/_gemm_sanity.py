# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GEMM accuracy sanity on current DTK/torch stack.

Checks bf16 and fp32 matmul rel error vs fp64 CPU reference for shapes
matching GLM-5-W4A8 linears (K=6144). A bf16 accumulation regression
(worse than ~1e-2 output rel) would poison all 78 layers.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
M, N, K = 64, 2048, 6144

a = torch.randn(M, K, dtype=torch.float64)
b = torch.randn(N, K, dtype=torch.float64)
ref = a @ b.t()  # cpu fp64

a32, b32 = a.float().to(dev), b.float().to(dev)
a16, b16 = a.bfloat16().to(dev), b.bfloat16().to(dev)

y32 = (a32 @ b32.t()).cpu().double()
# input-rounding floor: what pure rounding of inputs costs (fp32 accum ideal)
y16_floor = (a.bfloat16().double() @ b.bfloat16().double().t())
y16 = (a16 @ b16.t()).cpu().double()

def rel(y):
    return ((y - ref).abs() / (ref.abs() + 1e-6)).mean().item()

print(f"torch {torch.__version__} dev {torch.cuda.get_device_name(0)}")
print(f"fp32 matmul rel:      {rel(y32):.3e}")
print(f"bf16 input floor rel: {rel(y16_floor):.3e}")
print(f"bf16 matmul rel:      {rel(y16):.3e}")
print(f"bf16 excess vs floor: {rel(y16) / max(rel(y16_floor), 1e-12):.2f}x")

# split-consistency: same op, different M padding (catches algo-dependent bugs)
a16_pad = torch.cat([a16, torch.zeros(1, K, device=dev, dtype=torch.bfloat16)], 0)
y_pad = (a16_pad @ b16.t())[:M].cpu().double()
print(f"bf16 pad-consistency max|d|: {(y_pad - y16).abs().max().item():.4e}")

# rms norm chain stability quick check (fp32)
x = torch.randn(5, 6144, device=dev)
w = torch.randn(6144, device=dev)
r1 = F.rms_norm(x.float(), (6144,), w.float(), 1e-6)
print("rms ok", float(r1.sum()))
