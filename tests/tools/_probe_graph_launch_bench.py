# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""hipGraphLaunch 发射开销微基准（platform 诊断，无模型依赖）。

背景（2026-10-16 nmz26 GLM-5.3-Flash TP4 decode 0.43s/步定性）：py-spy 实锤
主线程卡 torch/cuda/graphs.py replay（hipGraphLaunch）、4 scheduler 各烧满一
核而 GPU 利用率仅 ~28%（实算 ~120ms）→ 服务全量图 DTK 发射 ~300ms/步。
本探针用 addmm 造节点数递增的裸图，量 launch+sync / 纯 launch / eager 对照。
当时实测：50/500/2000 节点图 = 0.78/6.2/22.1ms，eager 2000 节点 = 47.7ms
（即裸图发射本身不慢，服务全量图的发射代价需 DTK 侧跟进；换 DTK 版本后
用本探针复测即可回归）。

跑法：容器内 `python tests/tools/_probe_graph_launch_bench.py`（任意单卡）。
"""
import time

import torch

dev = "cuda"
N = 2000
xs = [torch.ones(1, 256, device=dev, dtype=torch.float32) for _ in range(8)]
ws = [torch.randn(256, 256, device=dev, dtype=torch.float32) for _ in range(8)]

def body(i):
    x = xs[i % 8]
    xs[i % 8] = torch.addmm(x, x, ws[i % 8])

for graph_n in (50, 500, 2000):
    g = torch.cuda.CUDAGraph()
    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(graph_n):
            body(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    with torch.cuda.graph(g):
        for i in range(graph_n):
            body(i)
    torch.cuda.synchronize()

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        g.replay()
        torch.cuda.synchronize()
    per_sync = (time.time() - t0) / 30
    t0 = time.time()
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    per_launch = (time.time() - t0) / 30
    print(f"graph n={graph_n}: launch+sync={per_sync*1000:.2f}ms  launch={per_launch*1000:.2f}ms")

# eager same 2000 ops
for _ in range(3):
    for i in range(N):
        body(i)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    for i in range(N):
        body(i)
    torch.cuda.synchronize()
per_eager = (time.time() - t0) / 10
print(f"eager n={N}: {per_eager*1000:.2f}ms")
print("DONE")
