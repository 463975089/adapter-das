# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Cross-check C5/C6 against the plain-TP2 30197 server (model-quality
discriminator for the DP-attention probe). Run INSIDE container."""

import json
import sys
import urllib.request

URL = "http://127.0.0.1:30197/generate"


def gen(text, mt):
    body = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["text"]


for tag, text, mt in [
    ("C5", "The Great Wall is located in", 32),
    ("C6", "print('hello', ", 20),
]:
    t = gen(text, mt)
    print(f"{tag}: {t[:160].encode('unicode_escape').decode()[:200]}")
sys.exit(0)
