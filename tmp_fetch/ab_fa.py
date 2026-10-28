import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30197"
TAG = sys.argv[2] if len(sys.argv) > 2 else "ab"

FILL = ("The asteroid survey vessel Kestrel logged daily magnetic "
        "anomalies over the southern dust plains, each entry signed by its "
        "watch officer and countersigned by the station archivist before "
        "being sealed into the yearly vault register.")


def est_tok(s):
    return int(len(s) / 3.8)  # English prose ~3.8 chars/token


def gen(text, n, t=240):
    body = json.dumps({"text": text,
                       "sampling_params": {"temperature": 0,
                                           "max_new_tokens": n}}).encode()
    r = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", body,
        {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(r, timeout=t).read())
    dt = time.time() - t0
    return out["text"], dt, out["meta_info"]["completion_tokens"]


# 1 anchors x3
for name, p, key in [("germany", "The capital of Germany is", "Berlin"),
                     ("china", "The capital of China is", "Beijing"),
                     ("japan", "The capital of Japan is", "Tokyo")]:
    try:
        txt, dt, _ = gen(p, 24)
        ok = key in txt
        print(f"[{TAG}] anchor {name}: {'OK' if ok else 'MISS'} | {txt[:60]!r}")
    except Exception as e:
        print(f"[{TAG}] anchor {name}: ERR {type(e).__name__} {e}")

# 2 decode short x3
gen("warmup hello", 16)
for i in range(3):
    try:
        txt, dt, n = gen("The capital of France is", 256)
        print(f"[{TAG}] SHORT-256 run{i}: {dt:.2f}s = {n/dt:.2f} tok/s")
    except Exception as e:
        print(f"[{TAG}] SHORT-256 run{i}: ERR {type(e).__name__} {e}")
        break

# 3 prefill 4k / 8k (prefill-heavy, 16 new tokens)
for label, reps in [("prefill2k", 32), ("prefill4k", 68),
                        ("prefill8k", 130)]:
    try:
        ctx = (FILL + " ") * reps
        txt, dt, n = gen(ctx, 16, t=300)
        print(f"[{TAG}] {label}: wall {dt:.2f}s (ctx est {est_tok(ctx)} tok, "
              f"{n} out; wall-16steps ~= TTFT)")
    except Exception as e:
        print(f"[{TAG}] {label}: ERR {type(e).__name__} {e}")

# 4 long-ctx dual fact ~6k
factline = ("The golden comet of Andalor was first observed in the year "
            "2312 by astronomer Mira Kovellen. The deepest mining colony "
            "on Thalos IX is called Darkmouth, and it produces 300 tons "
            "of ionite per year.")
facts = (factline + " ") * 48
try:
    txt, dt, n = gen(facts + " Question: Who first observed the golden "
                     "comet of Andalor and what is the deepest mining "
                     "colony on Thalos IX called? Answer briefly:",
                     220, t=300)
    ok = ("Kovellen" in txt) and ("Darkmouth" in txt)
    print(f"[{TAG}] LONGCTX-6k: {dt:.1f}s facts_ok={ok} "
          f"(est {est_tok(facts)} tok)")
    print(f"[{TAG}] LONGCTX tail: {txt[-180:]!r}")
except Exception as e:
    print(f"[{TAG}] LONGCTX-6k: ERR {type(e).__name__} {e}")

# 5 R-13 #2 wall probe ~10.5k ctx (LAST: on the emu arm this OOMs and can
# take the server down)
big = (FILL + " ") * 180
try:
    txt, dt, n = gen(big + " Question: What vessel logged magnetic "
                     "anomalies? One sentence:", 48, t=300)
    ok = "Kestrel" in txt
    print(f"[{TAG}] WALL-11k: OK {dt:.1f}s facts_ok={ok} "
          f"(est {est_tok(big)} tok)")
    print(f"[{TAG}] WALL-10k tail: {txt[:120]!r}")
except Exception as e:
    print(f"[{TAG}] WALL-10k: ERR {type(e).__name__} {e}")
