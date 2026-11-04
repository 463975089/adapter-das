# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K-100 concurrent eval against a live sglang server (HCU adapter).

Reads the parquet test split, takes the first N=100 items, fires 16-way
concurrent /generate requests, extracts the final `#### <answer>` number,
and scores against gold. Non-thinking friendly: clamp with max_new_tokens
and cut anything after a closing </think> tag.

Usage: python gsm8k_eval.py [--url http://127.0.0.1:30110] [-n 100]
       [-c 16] [--max-new 512]
"""

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request

import pandas as pd

ANS_RE = re.compile(r"####\s*([\-0-9\.,]+)")


def last_number(text):
    matches = ANS_RE.findall(text)
    if matches:
        num = matches[-1].replace(",", "")
    else:
        nums = re.findall(r"-?\d+\.?\d*", text)
        if not nums:
            return None
        num = nums[-1]
    return num.rstrip(".")


def ask(url, question, max_new):
    prompt = (
        f"{question}\nPlease reason step by step, and put your final "
        f"answer after '####'."
    )
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new, "temperature": 0.0,
        },
    }).encode()
    req = urllib.request.Request(
        url + "/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["text"]
    if "</think>" in out:  # thinking model: score only post-think text
        out = out.split("</think>")[-1]
    return out


def gold_answer(gold_text):
    m = ANS_RE.search(gold_text)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30110")
    ap.add_argument("--data",
                    default="/models/datasets/gsm8k/main/test-00000-of-00001.parquet")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-c", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    items = df.head(args.n)[["question", "answer"]].to_dict("records")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.c) as ex:
        futs = [
            ex.submit(ask, args.url, it["question"], args.max_new)
            for it in items
        ]
        outs = [f.result() for f in futs]
    dt = time.time() - t0

    ok = 0
    fails = []
    for it, out in zip(items, outs):
        gold = gold_answer(it["answer"])
        got = last_number(out)
        if got is not None and gold is not None and abs(float(got) - float(gold)) < 1e-4:
            ok += 1
        else:
            fails.append((gold, got, out[-120:]))
    print(f"GSM8K-{len(items)}: correct={ok} acc={ok/len(items):.1%} in {dt:.0f}s")
    for gold, got, tail in fails[:6]:
        print(f"  MISS gold={gold} got={got} tail={tail!r}")


if __name__ == "__main__":
    main()
