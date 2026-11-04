# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MTP rig probe (bw7, run INSIDE container): 8 anchors + 2x256-tok throughput.

Works against base / mtp / mtpe roles alike. Accept-length must be read from
the SERVER log (`grep -oE "accept len: [0-9.]+"`; the probe only measures
wall-clock).

Validated 2026-10-13: base 19.6-20.6 tok/s ; mtp k=2 1.7 tok/s (host-cost
verdict experiments, see models.md). Anchors 8/8 on all three roles.

Usage: python3 _probe_mtp_bw7.py [base_url]
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30199"

ANCHORS = [
    ("The capital of France is", "Paris"),
    ("The capital of China is", "Beijing"),
    ("1 + 1 =", "2"),
    ("24 * 4 =", "96"),
    ("def add(a, b):\n    return", "a + b"),
    ("The sun rises in the east and sets in the", "west"),
    ("A, B, C, D, E,", "F"),
    ("The largest planet in the solar system is", "Jupiter"),
]

ESSAY = ("Write a short essay about the history of computing, from Babbage "
         "to the modern AI era. Cover at least six distinct milestones.")


def gen(text, max_new):
    body = json.dumps({"text": text,
                       "sampling_params": {"temperature": 0,
                                           "max_new_tokens": max_new}}).encode()
    req = urllib.request.Request(URL + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    ok = 0
    for prompt, want in ANCHORS:
        out, _ = gen(prompt, 96)
        good = want.lower() in str(out.get("text", "")).lower()
        ok += good
        print(("PASS" if good else "FAIL"), repr(prompt[:36]), "->",
              repr(str(out.get("text", ""))[:70]), flush=True)
    print(f"ANCHORS: {ok}/8")

    for i in (1, 2):
        out, dt = gen(ESSAY, 256)
        n = out.get("meta_info", {}).get("completion_tokens", 0)
        print(f"THROUGHPUT run{i}: {n} tok in {dt:.1f}s = {n/dt:.1f} tok/s",
              flush=True)
    sys.exit(0 if ok == 8 else 1)


if __name__ == "__main__":
    main()
