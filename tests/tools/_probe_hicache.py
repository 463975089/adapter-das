# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Task73: hicache eviction/reload probe.

Fill the small GPU pool (max-total-tokens 2048) with four ~600-token stories
with disjoint prefixes, so LRU evicts story 1/2 by the time story 4 lands.
Re-sending stories 1/2 then must hit the host-memory tier (schedule_policy
counts device+host hits as #cached-token) and reproduce the greedy output.
Pass the base URL as argv[1] (default :30085), output JSON argv[2].
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30085"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/logs/q4b_hicache_probe.json"


def gen(text, max_new=12):
    body = json.dumps(
        {"text": text,
         "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["text"], time.time() - t0


PARA = {
    1: ("alpha", "lighthouse keeper Bram on the island of Vorn who polished the brass lantern"),
    2: ("beta", "botanist Selene in the valley of Kar who catalogued rare orchids"),
    3: ("gamma", "astronomer Corin at the observatory of Thal who tracked comets"),
    4: ("delta", "fisherman Marek on the lake of Oss who mended blue nets"),
}


def story(i):
    name, body = PARA[i]
    return ("This is the %s chronicle, record %d. It tells of the %s. "
            "The details are deliberately unique so no prefix is shared. " % (name, i, body)) * 22


def main():
    out = {}
    for i in (1, 2, 3, 4):
        t, dt = gen(story(i) + " END. SUMMARY:")
        out[i] = t
        print("FILL s%d %.2fs -> %r" % (i, dt, t[:40]), flush=True)

    for i in (1, 2):
        t2, dt2 = gen(story(i) + " END. SUMMARY:")
        same = (t2 == out[i])
        print("RETRY s%d %.2fs -> %r same=%s" % (i, dt2, t2[:40], same), flush=True)
        out["retry_same_%d" % i] = same

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("SAVED", OUT, flush=True)


if __name__ == "__main__":
    main()
