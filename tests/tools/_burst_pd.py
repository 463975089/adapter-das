# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent PD burst probe: N threads, distinct prompts, print outputs.

Usage: python _burst_pd.py [base_url] [n]
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30090"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

ANCHOR = "The capital of France is"
EXPECT = " Paris"


def gen(text, tag, max_new=32):
    body = json.dumps(
        {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        return tag, out.get("text", ""), round(time.time() - t0, 2), None
    except Exception as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        return tag, "", round(time.time() - t0, 2), f"{e} {body}"


def main():
    jobs = [(ANCHOR, "anchor")] + [
        (f"Tell me a short story about a dragon number {i}:", f"story{i}") for i in range(1, N)
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(lambda a: gen(*a), jobs))
    for tag, text, dt, err in results:
        if err:
            print(f"[{tag}] ERROR {dt}s: {err}")
        else:
            print(f"[{tag}] {dt}s: {text[:80]!r}")
    anchor = next(r for r in results if r[0] == "anchor")
    ok = EXPECT in anchor[1]
    print(f"[{'PASS' if ok else 'FAIL'}] concurrent anchor -> {EXPECT}")


if __name__ == "__main__":
    main()
