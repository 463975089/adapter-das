#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""bench_serving.py — sglang serving 基准（零第三方依赖，python3 直接跑）。

用法：
  python3 bench_serving.py                                   # 默认 127.0.0.1:30187
  python3 bench_serving.py --url http://10.16.1.26:30187     # nmz26 宿主直跑
  python3 bench_serving.py -c 16 -n 128 -r 3                 # 16 并发 × 128 tok × 3 轮
  python3 bench_serving.py --anchors-only                    # 只跑锚点门

内容：健康检查 → 锚点（Berlin/Tokyo 文本门）→ 单流吞吐 → 并发吞吐。
计时口径：/generate 非流式、temperature=0、ignore_eos=True，
tok/s = 生成 token 数 / 请求墙钟；每步 ms = 墙钟 / token 数。
（prefill 计入首请求延迟；要纯 decode 口径用长一点的 -n 稀释。）
"""
import argparse
import json
import threading
import time
import urllib.request


def gen(url, text, n, timeout):
    body = {"text": text, "sampling_params": {
        "max_new_tokens": n, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(
        url + "/generate", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    txt = json.loads(resp.read())["text"]
    return txt, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:30187")
    ap.add_argument("-n", "--tokens", type=int, default=96)
    ap.add_argument("-c", "--concurrency", type=int, default=8)
    ap.add_argument("-r", "--rounds", type=int, default=1)
    ap.add_argument("--prompt", default="Write a story about a dragon:")
    ap.add_argument("--anchors-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    code = urllib.request.urlopen(url + "/health", timeout=10).status
    print(f"health: {code}  ({url})")

    for p, want in [("The capital of Germany is", " Berlin"),
                    ("The capital of Japan is", " Tokyo")]:
        try:
            t, _ = gen(url, p, 12, a.timeout)
            print(("OK   " if t.startswith(want) else "FAIL ")
                  + p + " -> " + repr(t[:44]))
        except Exception as e:
            print(f"ERR  {p}: {e}")
    if a.anchors_only:
        return

    for i in range(2):  # warmup
        gen(url, "Hello", 8, a.timeout)

    print(f"\n[single-stream] {a.tokens} tok/req")
    for r in range(a.rounds):
        _, dt = gen(url, a.prompt, a.tokens, a.timeout)
        print(f"  round{r + 1}: {a.tokens / dt:6.2f} tok/s   "
              f"step {dt / a.tokens * 1000:5.0f} ms   ({dt:.1f}s)")

    print(f"\n[concurrent c={a.concurrency}] {a.tokens} tok/req")
    for r in range(a.rounds):
        lat = [None] * a.concurrency

        def work(i):
            _, lat[i] = gen(url, a.prompt, a.tokens, a.timeout)

        th = [threading.Thread(target=work, args=(i,))
              for i in range(a.concurrency)]
        t0 = time.time()
        [t.start() for t in th]
        [t.join() for t in th]
        dt = time.time() - t0
        tot = a.concurrency * a.tokens
        print(f"  round{r + 1}: {tot / dt:6.2f} tok/s agg  "
              f"step {dt / a.tokens * 1000:5.0f} ms  "
              f"({dt:.1f}s, slowest req {max(x for x in lat if x):.1f}s)")


if __name__ == "__main__":
    main()
