# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Send /generate, print input/output ids and text."""
import json
import sys
import urllib.request

req = {
    "text": sys.argv[1] if len(sys.argv) > 1 else "Paris is the capital of",
    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    "return_hidden_states": True,
    "return_input_ids": True,
}
r = urllib.request.Request(
    "http://127.0.0.1:30064/generate", json.dumps(req).encode(),
    {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=180).read())
mi = d["meta_info"]
print("text:", repr(d["text"]))
print("input_ids:", mi.get("input_ids"))
print("output_ids:", d.get("output_ids"))
print("prompt_tokens:", mi.get("prompt_tokens"))
