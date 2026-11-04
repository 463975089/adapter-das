# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json, time, urllib.request

URL = "http://127.0.0.1:30084"

def gen(text, max_new=24):
    body = json.dumps({"text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}).encode()
    req = urllib.request.Request(URL + "/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["text"], time.time() - t0

story = ("Here is a unique story about a lighthouse keeper named Bram living on the rocky "
         "island of Vorn. Every morning he climbed the spiral staircase, polished the great "
         "brass lantern, and recorded the wind and waves in his leather logbook. ") * 6
q1 = story + "Question: what is the keeper name? Answer:"
q2 = story + "Question: what is the keeper name? Answer:"
q3 = story + "Question: what does he polish every morning? Answer:"

a1, t1 = gen(q1)
a2, t2 = gen(q2)
a3, t3 = gen(q3)
same12 = a1 == a2
print("R1 cold     %.2fs" % t1, flush=True)
print("R2 re-send  %.2fs same_as_r1=%s" % (t2, same12), flush=True)
print("R3 pfx+newQ %.2fs" % t3, flush=True)
json.dump({"t": [t1, t2, t3], "same12": same12},
          open("/root/logs/q4b_radix_probe.json", "w"), ensure_ascii=False, indent=1)
print("SAVED", flush=True)
