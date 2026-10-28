"""EP auto-mode probe battery for the 30193 server (run INSIDE container).

auto = prefill DeepEP normal (triton contiguous grouped GEMM) + decode
low_latency (DTK masked). Every fresh request cold-prefills through the
normal path (--disable-radix-cache); the long-prompt case drives a big
contiguous GEMM batch.
"""

import concurrent.futures as cf
import json
import sys
import urllib.request

URL = "http://127.0.0.1:30193/generate"


def gen(text, mt=40):
    body = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["text"]


ANCHORS = [
    ("A1", "The capital of France is", 12),
    ("A2", "Which number is bigger, 9.9 or 9.11? Answer:", 200),
    ("A3", "1+2+3+...+100 = ", 12),
    ("A4", "The chemical formula of water is", 12),
    ("A5", "The square root of 100 is", 12),
]

fails = []

print("== single-request anchors (prefill normal + decode LL) ==")
outs = {}
for tag, text, mt in ANCHORS:
    t = gen(text, mt)
    outs[tag] = t
    print(f"{tag}: {t[:180].encode('unicode_escape').decode()[:220]}")

hints = {
    "A1": "paris",
    "A3": "5050",
    "A4": "h2o",
    "A5": "10",
}
for tag, hint in hints.items():
    if hint not in outs[tag].lower():
        fails.append(f"{tag} hint '{hint}' missing")

print("== determinism (A2 x3, majority-identical bar) ==")
rer = [gen(ANCHORS[1][1], ANCHORS[1][2]) for _ in range(2)]
pool = [outs["A2"]] + rer
same = pool[0] == pool[1] or pool[0] == pool[2] or pool[1] == pool[2]
print(f"DET majority-identical={same} "
      f"(lens={[len(x) for x in pool]})")
if not same:
    for i, o in enumerate(pool):
        print(f"det{i}: {o[:120].encode('unicode_escape').decode()[:150]}")
    fails.append("determinism")

print("== 6-way concurrent distinct prompts ==")
CONC = [
    ("C1", "The largest ocean on Earth is", 12),
    ("C2", "def quicksort(lst):\n    ", 80),
    ("C3", "2, 3, 5, 7, 11, 13,", 12),
    ("C4", "The first president of the United States was", 12),
    ("C5", "The Great Wall is located in", 12),
    ("C6", "print('hello', ", 20),
]
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(gen, t, mt): tag for tag, t, mt in CONC}
    conc = {futs[f]: f.result() for f in cf.as_completed(futs)}
for tag, _, _ in CONC:
    print(f"{tag}: {conc[tag][:140].encode('unicode_escape').decode()[:180]}")

expect_hint = {
    "C1": "pacific", "C2": "sort", "C3": "17", "C4": "washington",
    "C5": "china", "C6": "hello",
}
for tag, hint in expect_hint.items():
    # C6 judges prompt+continuation (the echoed literal is in the prompt)
    c6_prompt = dict((t, p) for t, p, _ in CONC)["C6"]
    hay = (c6_prompt + conc[tag]) if tag == "C6" else conc[tag]
    if hint.lower() not in hay.lower():
        fails.append(f"{tag} hint '{hint}' missing")

print("== long prompt (~1100 tok, big contiguous prefill batch) ==")
story = (
    "Alice opened the small blue box and found a brass key inside. "
    "The key was warm, as if someone had just held it. She turned to her "
    "brother Ben and asked what it might open. Ben shrugged, but his eyes "
    "went to the old clock tower at the end of the garden. "
) * 30
t = gen(
    story
    + "\n\nQuestion: What did Alice find in the box? Answer with one word: ",
    16,
)
print(f"LONG: {t[:80].encode('unicode_escape').decode()[:100]}")
if "key" not in t.lower():
    fails.append("long-prompt 'key' missing")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
