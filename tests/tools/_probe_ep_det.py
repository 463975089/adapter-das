# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Determinism deep-dive for 30193 EP-auto: where exactly do reruns diverge,
and is the long-prompt answer stable across reruns?"""

import json
import sys
import urllib.request

URL = "http://127.0.0.1:30193/generate"


def gen(text, mt=40):
    body = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["text"]


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


A2 = "Which number is bigger, 9.9 or 9.11? Answer:"
print("== A2 x3 ==")
outs = [gen(A2, 200) for _ in range(3)]
for i, o in enumerate(outs):
    print(f"run{i} len={len(o)} tail={o[-80:].encode('unicode_escape').decode()[:100]}")
for i in (1, 2):
    cp = common_prefix_len(outs[0], outs[i])
    print(f"run0~run{i}: common prefix = {cp} chars "
          f"({'IDENTICAL' if outs[0] == outs[i] else 'diverge'})")

print("== long-prompt x2, forcing question ==")
story = (
    "Alice opened the small blue box and found a brass key inside. "
    "The key was warm, as if someone had just held it. She turned to her "
    "brother Ben and asked what it might open. Ben shrugged, but his eyes "
    "went to the old clock tower at the end of the garden. "
) * 30
Q = story + "\n\nQuestion: What did Alice find in the box? The answer is exactly one word, and the word is: "
louts = [gen(Q, 24) for _ in range(2)]
for i, o in enumerate(louts):
    print(f"long{i}: {o[:120].encode('unicode_escape').decode()[:150]}")
cp = common_prefix_len(louts[0], louts[1])
print(f"long0~long1 common prefix = {cp} chars "
      f"({'IDENTICAL' if louts[0] == louts[1] else 'diverge'})")
ok_key = any("key" in o.lower() for o in louts)
print(f"any-run 'key': {ok_key}")
print("RESULT: INFO_ONLY")
