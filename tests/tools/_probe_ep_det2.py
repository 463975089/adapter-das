"""Short-anchor exact determinism (LL-battery bar) on 30193."""

import json
import urllib.request

import sys
PORT = sys.argv[1] if len(sys.argv) > 1 else "30193"
URL = f"http://127.0.0.1:{PORT}/generate"


def gen(text, mt=40):
    body = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["text"]


CASES = [
    ("A1", "The capital of France is", 12),
    ("A3", "1+2+3+...+100 = ", 12),
    ("C3", "2, 3, 5, 7, 11, 13,", 12),
    ("LONG", None, 24),  # filled below
]
story = (
    "Alice opened the small blue box and found a brass key inside. "
    "The key was warm, as if someone had just held it. She turned to her "
    "brother Ben and asked what it might open. Ben shrugged, but his eyes "
    "went to the old clock tower at the end of the garden. "
) * 30
CASES[3] = ("LONG", story +
            "\n\nQuestion: What did Alice find in the box? "
            "The answer is exactly one word, and the word is: ", 24)

fails = []
for tag, text, mt in CASES:
    outs = [gen(text, mt) for _ in range(3)]
    ident = outs[0] == outs[1] == outs[2]
    print(f"{tag}: exact-identical x3 = {ident} "
          f"lens={[len(o) for o in outs]}")
    if not ident:
        print(f"  heads: {[o[:40].encode('unicode_escape').decode()[:48] for o in outs]}")
    if tag != "LONG" and not ident:
        fails.append(tag)
    if tag == "LONG":
        key = all("key" in o.lower() for o in outs)
        print(f"  all-answer-key = {key}")
        if not key:
            fails.append("LONG-key")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
