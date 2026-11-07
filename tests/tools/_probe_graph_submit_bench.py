# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""graph 提交路径微基准（DTK runtime 诊断，单卡，~2 分钟）。

回答四个问题（2026-10-16 validation-host-26 gfx938 dtk26041 实测结论写在每节）：
1) 提交成本是否随节点数线性？——线性，~6.8-8.9 µs/节点（纯 addmm 图）。
2) 多线程分包提交能否并行？——不能：2 线程 2×1000 节点 = 串行 0.98×
   （runtime 提交路径进程内全局串行，piecewise-graph 多线程方案死刑）。
3) replay API 是"提交贵"还是"等完成贵"？——提交贵：全同步后首次调用
   仍要 6.8ms/1000 节点（GPU 实算仅 ~2ms）。
4) （跨进程独立性需两个进程同时跑，本探针不含；实测两进程各 113/s =
   单进程原速，跨进程独立。）
另注：torch.cuda.graph(g, stream=...) 带 stream 捕获路径在本栈 segfault，
本探针只用默认流捕获。

跑法：容器内 HIP_VISIBLE_DEVICES=<空卡> PYTHONPATH 同服务环境
`python3 tests/tools/_probe_graph_submit_bench.py`。
"""
import time
import threading

import torch

DEV = "cuda"
N_LADDER = (1000, 2000, 4000)


def _mk_ops():
    xs = [torch.ones(1, 256, device=DEV, dtype=torch.float32) for _ in range(8)]
    ws = [torch.randn(256, 256, device=DEV, dtype=torch.float32) for _ in range(8)]

    def body(i):
        x = xs[i % 8]
        xs[i % 8] = torch.addmm(x, x, ws[i % 8])

    return body


def capture(n):
    body = _mk_ops()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n):
            body(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            body(i)
    torch.cuda.synchronize()
    return g


def bench(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


print("== 1) node ladder (launch + exec + sync) ==")
for n in N_LADDER:
    g = capture(n)
    t = bench(lambda: g.replay())
    print(f"single N={n}: {t:.2f} ms ({t / n * 1000:.1f} us/node)", flush=True)
    del g

print("== 2) two-thread submission ==")
gA = capture(1000)
gB = capture(1000)
t_seq = bench(lambda: (gA.replay(), gB.replay()))


def par2():
    t1 = threading.Thread(target=lambda: gA.replay())
    t2 = threading.Thread(target=lambda: gB.replay())
    t1.start(); t2.start(); t1.join(); t2.join()


t_par = bench(par2)
print(f"serial 2x1000: {t_seq:.2f} ms | two threads: {t_par:.2f} ms "
      f"-> parallel speedup {t_seq / t_par:.2f}x", flush=True)

print("== 3) blocking discriminator (pure API cost) ==")
g = capture(1000)
for _ in range(3):
    g.replay()
torch.cuda.synchronize()
for _ in range(3):
    torch.cuda.synchronize()
    t0 = time.time(); g.replay(); d1 = time.time() - t0
    t0 = time.time(); g.replay(); d2 = time.time() - t0
    torch.cuda.synchronize()
    print(f"call1(pure submit)={d1 * 1000:.2f} ms  call2(behind running)={d2 * 1000:.2f} ms",
          flush=True)
print("DONE", flush=True)
