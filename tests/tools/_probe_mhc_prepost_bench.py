# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""mHC pre/post 链 serving 形状计时 + dispatch 路径计数（HCU_MHC_UNFUSE 配套）。

背景（2026-10-16）：mHC 融合已被 HCU_MHC_UNFUSE 关掉（gfx938 上 aiter 融合
mhc_post_pre 的 FFN 段 z 偏移 +0.5~+5 = 词汤主病灶），unfuse 后走 hc_pre +
hc_post 纯 torch 链。本探针量这两段在 decode（T=1）/prefill（T=128）形状
的耗时并计数实际落哪条 dispatch（torch/tilelang/aiter）。

**standalone 坑（已踩）**：
- 脚本前置的 ServerArgs+publish、get_parallel/tp_group 伪造、
  use_symmetric_memory 打桩是 standalone 跑 mhc 模块的必要条件（serving 里
  由框架提供）；
- tilelang mhc_pre 在 standalone 编译失败（"Could not normalize iterators"）
  而 serving 能编译——环境差异未解，standalone 计数落到 torch 臂属预期。

跑法：容器内 `python tests/tools/_probe_mhc_prepost_bench.py`。
"""
import os
import time

os.environ["SGLANG_OPT_FUSE_MHC_POST_PRE"] = "false"  # serving sets this (mhcunfuse)

import torch

torch.manual_seed(7)
dev = "cuda"

import sglang.kernels.ops.layernorm.mhc as M
from sglang.srt.server_args import ServerArgs
import sglang.srt.runtime_context as _rtc
_sa = ServerArgs(model_path='/models/glm53f', trust_remote_code=True, tp_size=4)
_rtc.publish(_sa, role='scheduler')
import contextlib
M.use_symmetric_memory = lambda *a, **kw: contextlib.nullcontext()
M.is_allocation_symmetric = lambda: False
class _FP:
    tp_group = None
    tp_rank = 0
    tp_size = 1
M.get_parallel = lambda: _FP()

counts = {"pre_torch": 0, "pre_tilelang": 0, "pre_aiter": 0,
          "post_torch": 0, "post_tilelang": 0, "post_aiter": 0}
_o_pre_torch = M._mhc_pre_torch
_o_post_torch = M._mhc_post_torch
M._mhc_pre_torch = lambda **kw: (counts.__setitem__("pre_torch", counts["pre_torch"] + 1) or _o_pre_torch(**kw))
M._mhc_post_torch = lambda *a, **kw: (counts.__setitem__("post_torch", counts["post_torch"] + 1) or _o_post_torch(*a, **kw))
print("use_aiter_mhc:", M._use_aiter_mhc())
print("use_tilelang_pre:", M._use_tilelang_mhc_pre(), "use_tilelang_post:", M._use_tilelang_mhc_post())

HC, HID = 4, 4096
hc_scale = torch.randn(3, device=dev, dtype=torch.float32)
hc_base = torch.randn((2 + HC) * HC, device=dev, dtype=torch.float32)

def bench(T, iters=50):
    x = torch.randn(T, HC * HID, device=dev, dtype=torch.bfloat16)
    fn = torch.randn((2 + HC) * HC, HC * HID, device=dev, dtype=torch.float32)
    blk = torch.randn(T, HID, device=dev, dtype=torch.bfloat16)
    nw = torch.randn(HID, device=dev, dtype=torch.bfloat16)

    def pre():
        return M.hc_pre(x, fn, hc_scale, hc_base, HC, 1e-5, 1e-6, 20, 2.0)

    li, h_res, h_post, fused = pre()
    def post():
        return M.hc_post(blk, x, h_post.float(), h_res.float(), HC)

    for f, name in ((pre, "hc_pre"), (post, "hc_post")):
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
        per = (time.time() - t0) / iters * 1000
        print(f"T={T} {name}: {per:.3f} ms/call  (x90/step = {per*90:.1f} ms)")

bench(1)
bench(128)
print("dispatch counts:", counts)
print("DONE")
