# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PD disaggregation probe: semantic anchors + determinism + long-prompt transfer.

Usage: python _probe_pd.py [base_url]   (default http://127.0.0.1:30090)
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30090"

ANCHORS = [
    ("The capital of France is", "Paris"),
    ("The capital of China is", "Beijing"),
    ("1, 2, 3, 4, 5,", "6"),
    ("123 + 456 =", "579"),
    ("A square has four", "sides"),
    ("Water boils at 100 degrees", "Celsius"),
]


def gen(text, max_new=24):
    body = json.dumps(
        {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out.get("text", ""), time.time() - t0


def main():
    passed = 0
    for prompt, expect in ANCHORS:
        text, dt = gen(prompt)
        ok = expect.lower() in text.lower()
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r} -> {text[:60]!r} (expect {expect}) {dt:.2f}s")

    o1, _ = gen(ANCHORS[0][0])
    o2, _ = gen(ANCHORS[0][0])
    same = o1 == o2
    print(f"[{'PASS' if same else 'FAIL'}] determinism rerun identical: {same}")

    long_prompt = "The quick brown fox jumps over the lazy dog. " * 120
    text, dt = gen(long_prompt, max_new=32)
    print(f"[INFO] long-prompt ~{len(long_prompt.split())} words, {dt:.1f}s -> {text[:60]!r}")

    print(f"TOTAL {passed}/{len(ANCHORS)}")


if __name__ == "__main__":
    main()
