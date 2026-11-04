# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""EP low_latency / PD / single-stream probe battery for bw7 rigs.

Anchors stop at 5 for the LL 30B rig (:30197); pass a different base URL for
the PD 0.6B rig (then set --pd so max_tokens/anchors fit the small model).

Validated 2026-10-13: LL 5/5.

Usage (inside container):
  python3 _probe_ep_ll_bw7.py                       # LL 30B anchors on :30197
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30197"

ANCHORS = [
    ("cmp", "9.9 and 9.11, which is bigger? Answer with just the number.", "9.9"),
    ("sum", "What is 1+2+...+100? Answer with just the number.", "5050"),
    ("cap", "What is the capital of France? One word.", "Paris"),
    ("h2o", "What is the chemical formula of water? Just the formula.", "H2O"),
    ("sqrt", "What is sqrt(100)? Just the number.", "10"),
]

ESSAY = ("Write a short essay about the history of computing, "
         "from Babbage to the modern AI era.")


def gen(prompt, max_new, model="q30ll"):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_new, "temperature": 0}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], time.time() - t0


def main():
    ok = 0
    for tag, q, want in ANCHORS:
        out, _ = gen(q, 512)
        hit = want in out.replace(" ", "")
        ok += hit
        print(f"{tag}: {'PASS' if hit else 'FAIL'} "
              f"({out.strip().replace(chr(10), ' ')[:48]!r})")
    print(f"ANCHORS: {ok}/5")

    for i in (1, 2):
        out, dt = gen(ESSAY, 256)
        print(f"THROUGHPUT run{i}: {dt:.1f}s (read server log for tok/s)")
    sys.exit(0 if ok == 5 else 1)


if __name__ == "__main__":
    main()
