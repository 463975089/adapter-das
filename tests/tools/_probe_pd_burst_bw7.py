# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PD zero-cross-talk battery (bw7 is_cuda form, run INSIDE the container).

Fire 8 concurrent requests through the mini-LB (:30090), each asking a
different arithmetic question; every answer must contain its own expected
number. A swapped/mixed batch fails exactly one or two assertions and is
easy to tell apart from a total outage.

Validated 2026-10-13: 8/8. Note the thinking-model gotcha — Qwen3 needs
~max_tokens 160 to reach the answer; 24 only produces the `<think>` head
and false-positives as MISMATCH.

Usage: python3 _probe_pd_burst_bw7.py [base_url]
"""
import concurrent.futures as cf
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30090/v1/chat/completions"
MODEL = "q06p"


def main():
    qa = [(i, f"What is {i}+{i*9}? Answer with ONLY the number, nothing else.")
          for i in range(1, 9)]

    def ask(t):
        i, q = t
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "user", "content": q}],
                           "max_tokens": 160, "temperature": 0}).encode()
        req = urllib.request.Request(URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return i, json.load(r)["choices"][0]["message"]["content"]

    with cf.ThreadPoolExecutor(8) as ex:
        results = dict(ex.map(ask, qa))

    ok = 0
    for i, _ in qa:
        want = str(i + i * 9)
        hit = want in results[i].strip()
        ok += hit
        print(f"req{i}: want={want} got={results[i].strip()[:40]!r} "
              f"{'OK' if hit else 'MISMATCH'}")
    print(f"ZERO-CROSS-TALK: {ok}/8")
    sys.exit(0 if ok == 8 else 1)


if __name__ == "__main__":
    main()
