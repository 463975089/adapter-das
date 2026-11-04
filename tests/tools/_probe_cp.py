# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json, sys, urllib.request

PORT = sys.argv[1]
TAG = sys.argv[2]

def ask(prompt, max_tokens=512):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": "d", "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]

# long-context prompt (CP only matters when prefill spans chunks; 30B ctx 262k, use ~4k)
story_seed = ("请仔细阅读以下设定并回答问题。设定：艾琳是一位行星地质学家，她在开普勒-442b 的冰下海洋"
              "科研站工作了七年。站里的量子钻探阵列代号'深蓝'，负责采集冰层深处的微生物样本。艾琳的"
              "搭档机器人叫皮皮，负责维护氧气循环。站上的补给船每十八个月一班。")
filler = " 她每天的日志都记录着冰层厚度、辐射读数与皮皮的电池健康度，日志编号从 L0001 一路累加。"
long_ctx = story_seed + filler * 60

tests = [
    ("arith", "123 * 456 等于多少？只回答数字。", "56088", 256),
    ("capital", "法国的首都是哪座城市？只回答城市名。", "巴黎", 256),
    ("long", long_ctx + "\n\n问题：艾琳的搭档机器人叫什么名字？补给船多久一班？", "皮皮", 320),
]
fails = []
for name, q, expect, mt in tests:
    try:
        a = ask(q, mt)
        ok = expect in a
        print(f"{TAG} {name}: {'OK' if ok else 'MISS'} -> {a[:80]!r}", flush=True)
        if not ok:
            fails.append(name)
    except Exception as e:
        print(f"{TAG} {name}: EXC {e}", flush=True)
        fails.append(name)
# determinism (same long ctx twice)
a1 = ask(long_ctx + "\n\n问题：用一句话概括艾琳的日常。", 200)
a2 = ask(long_ctx + "\n\n问题：用一句话概括艾琳的日常。", 200)
det = a1 == a2
print(f"{TAG} determinism: {'OK' if det else 'DIFF'}", flush=True)
if not det:
    fails.append("det")
print(f"{TAG} RESULT: " + ("PASS" if not fails else f"FAIL {fails}"), flush=True)
