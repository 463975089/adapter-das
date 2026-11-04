# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Full-model test: 8 greedy tokens with top-9 logprobs per position."""
import json
import sys
import urllib.request

req = {
    "text": "The capital of France is",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 8,
    },
}
r = urllib.request.Request(
    "http://127.0.0.1:30064/generate", json.dumps(req).encode(),
    {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read())
print("text:", repr(d["text"]))
print("output_ids:", d.get("output_ids"))
print("meta prompt_tokens:", d["meta_info"].get("prompt_tokens"))
