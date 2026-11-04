# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Task73: Qwen3.5-4B MTP probe - semantic anchors + accept-length + throughput.

Run inside the container against the MTP server (default :30082, pass a
different base URL for the no-MTP baseline :30083). Accept length is read
from the server log ("accept len:" lines printed per decode batch).
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30082"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/logs/q4b_mtp_probe.json"


def gen(prompt, max_new=96):
    body = json.dumps(
        {"text": prompt,
         "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["text"], time.time() - t0


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


def main():
    results = []
    for prompt, want in ANCHORS:
        txt, dt = gen(prompt)
        ok = want.lower() in txt.lower()
        results.append({"prompt": prompt, "want": want, "got": txt[:150], "ok": ok})
        print(("PASS" if ok else "FAIL"), repr(prompt[:40]), "->",
              repr(txt[:80]), flush=True)

    txt, dt = gen("Write a detailed essay about the history of computing:", 256)
    thr = 256 / dt
    print("THROUGHPUT %.1f tok/s (%.1fs/256tok)" % (thr, dt), flush=True)
    json.dump({"results": results, "pass": sum(r["ok"] for r in results),
               "thr_tok_s": thr, "thr_text": txt[:300]},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print("SAVED", OUT, flush=True)


if __name__ == "__main__":
    main()
