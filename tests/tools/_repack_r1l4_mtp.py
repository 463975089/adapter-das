# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json, os, shutil
from safetensors.torch import save_file
from safetensors import safe_open

SRC = "/public/opendas/DL_DATA/llm-models/DeepSeek-R1"
BASE = "/root/models/R1-L4"          # already-patched 4-layer build (config + tokenizer)
DST = "/root/models/R1-L4-MTP"
os.makedirs(DST, exist_ok=True)

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]

keep = {}  # shard -> [(ckpt_name, out_name)]
for name, shard in wm.items():
    out = None
    if any(f".layers.{i}." in name for i in range(4)) or name in (
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        out = name
    elif ".layers.61." in name:
        # MTP layer: NextN loader reads model.layers.{num_hidden_layers} = 4
        out = name.replace(".layers.61.", ".layers.4.")
    if out:
        keep.setdefault(shard, []).append((name, out))

print("shards needed:", len(keep), flush=True)

for f in os.listdir(BASE):
    if f.endswith((".json", ".py")) and "safetensors" not in f:
        shutil.copy(f"{BASE}/{f}", f"{DST}/{f}")

tensors = {}
for shard, pairs in sorted(keep.items()):
    with safe_open(f"{SRC}/{shard}", framework="pt", device="cpu") as f:
        for n, out in pairs:
            tensors[out] = f.get_tensor(n)
    print("read", shard, flush=True)

save_file(tensors, f"{DST}/model.safetensors", metadata={"format": "pt"})
json.dump({"metadata": idx.get("metadata", {}),
           "weight_map": {n: "model.safetensors" for n in tensors}},
          open(f"{DST}/model.safetensors.index.json", "w"))
print("DONE", len(tensors), "tensors,", os.path.getsize(f"{DST}/model.safetensors") / 1e9, "GB", flush=True)
