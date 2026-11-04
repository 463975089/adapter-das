# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import torch

sys.path.insert(0, "/root/hcu_t2")

from safetensors import safe_open

MODEL = "/public/opendas/DL_DATA/llm-models/Qwen3.6-27B-Channel-INT8-w8a8"
import os

# find a Linear layer
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
wkey = next(k for k in wm if k.endswith("mlp.down_proj.weight"))
skey = wkey.replace(".weight", ".weight_scale")
wfile = wm[wkey]
with safe_open(os.path.join(MODEL, wfile), framework="pt") as f:
    w = f.get_tensor(wkey)
    ws = f.get_tensor(skey) if skey in wm and skey in f.keys() else None
print("w:", w.shape, w.dtype, "| scale:", None if ws is None else (ws.shape, ws.dtype))
if ws is not None:
    print("scale sample:", ws[:3, 0].tolist())
    # dequant sanity: per-channel (N,1) scale; w int8 [N,K]
    deq = w.float() * ws
    print("deq row0[:6]:", deq[0, :6].tolist())
    print("std:", deq.std().item(), "mean:", deq.mean().item())

# now run the CT scheme path end to end
import sglang.srt.layers.quantization.compressed_tensors.schemes.\
    compressed_tensors_w8a8_int8 as m

print("module int8_scaled_mm:", getattr(m, "int8_scaled_mm", None))
