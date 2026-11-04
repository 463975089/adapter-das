# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Long-prompt decode test: >index_topk(2048) tokens flips the DSA decode
graph to the sparse variant. If long-prompt decode is coherent while the
5-token prompt is not, the dense k-only decode graph is the culprit."""
import json
import sys
import urllib.request

words = ("The quick brown fox jumps over the lazy dog while the sun "
         "sets behind the mountains and rivers flow quietly through "
         "the valley where ancient trees stand tall ").split()
# ~3000 tokens
long_text = " ".join(words * 100)[:56000]

req = {
    "text": long_text,
    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
}
r = urllib.request.Request(
    "http://127.0.0.1:30064/generate", json.dumps(req).encode(),
    {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=600).read())
print("prompt_tokens:", d["meta_info"].get("prompt_tokens"))
print("text:", repr(d["text"]))
print("output_ids:", d.get("output_ids"))
