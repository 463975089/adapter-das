# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DP-attention probe battery for the 30198 server (run INSIDE container).

Judges: 4 semantic anchors, determinism rerun, 6-way concurrent
distinct-prompt isolation (round-robin across DP ranks), long prompt.
"""

import concurrent.futures as cf
import json
import sys
import urllib.request

URL = "http://127.0.0.1:30198/generate"


def gen(text, mt=40):
    body = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["text"]


ANCHORS = [
    ("A1", "The capital of France is", 32),
    ("A2", "1+1=2, 2+2=4, 4+4=", 24),
    ("A3", "def fibonacci(n):\n    ", 40),
    ("A4", "Write one sentence about the sun.", 28),
]

fails = []

print("== single-request anchors ==")
outs = {}
for tag, text, mt in ANCHORS:
    t = gen(text, mt)
    outs[tag] = t
    print(f"{tag}: {t[:160].encode('unicode_escape').decode()[:200]}")

print("== determinism (A1 rerun) ==")
d = gen(ANCHORS[0][1], ANCHORS[0][2])
same = d == outs["A1"]
print(f"DET same={same}")
if not same:
    fails.append("determinism")

print("== 6-way concurrent distinct prompts (DP isolation) ==")
CONC = [
    ("C1", "The largest ocean on Earth is", 40),
    ("C2", "def quicksort(lst):\n    ", 60),
    ("C3", "2, 3, 5, 7, 11, 13,", 20),
    ("C4", "The chemical formula of water is", 24),
    ("C5", "The Great Wall is located in", 32),
    ("C6", "print('hello', ", 20),
]
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(gen, t, mt): tag for tag, t, mt in CONC}
    conc = {futs[f]: f.result() for f in cf.as_completed(futs)}
for tag, _, _ in CONC:
    print(f"{tag}: {conc[tag][:140].encode('unicode_escape').decode()[:180]}")

expect_hint = {
    "C1": "Pacific", "C2": "sort", "C3": "17", "C4": "H",
    "C5": "China", "C6": "hello",
}
for tag, hint in expect_hint.items():
    if hint.lower() not in conc[tag].lower():
        fails.append(f"{tag} hint '{hint}' missing")

print("== long prompt (~1100 tok) ==")
story = (
    "Alice opened the small blue box and found a brass key inside. "
    "The key was warm, as if someone had just held it. She turned to her "
    "brother Ben and asked what it might open. Ben shrugged, but his eyes "
    "went to the old clock tower at the end of the garden. "
) * 30
t = gen(story + "\n\nQuestion: What did Alice find in the box? Answer with one word: ", 16)
print(f"LONG: {t[:80].encode('unicode_escape').decode()[:100]}")
if "key" not in t.lower():
    fails.append("long-prompt 'key' missing")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
