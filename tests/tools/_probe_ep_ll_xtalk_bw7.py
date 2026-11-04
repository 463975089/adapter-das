# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""EP low_latency 6-way concurrent zero-cross-talk battery (run INSIDE container).

Validated 2026-10-13: 6/6 against the :30197 LL rig.

Usage: python3 _probe_ep_ll_xtalk_bw7.py [base_url] [model]
"""
import concurrent.futures as cf
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30197/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "q30ll"


def main():
    qa = [(i, f"What is {i}+{i*9}? Answer with ONLY the number.")
          for i in range(1, 7)]

    def ask(t):
        i, q = t
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "user", "content": q}],
                           "max_tokens": 700, "temperature": 0}).encode()
        req = urllib.request.Request(URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return i, json.load(r)["choices"][0]["message"]["content"]

    with cf.ThreadPoolExecutor(6) as ex:
        res = dict(ex.map(ask, qa))

    ok = 0
    for i, _ in qa:
        hit = str(i + i * 9) in res[i].replace(" ", "")
        ok += hit
        print(i, "OK" if hit else "MISS " + repr(res[i][:50]))
    print(f"X-TALK: {ok}/6")
    sys.exit(0 if ok == 6 else 1)


if __name__ == "__main__":
    main()
