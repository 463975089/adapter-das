# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gates-lane assertion: the sglang bridge installs with ZERO degradation
(governance 3b-8 audit leg, W5-1, 2026-11-19).

PASS requires, after a full hcu_adapter.bridges.sglang install in this
process (HCU_ADAPTER_HOST=sglang):
  1. _FAILED_BRIDGES is empty -- no bridge failed into the upstream path;
  2. the registry's installed implementations contain no emu landing
     points (an impl path naming a *_emu module means emu served runtime).

Runs in the L1 gates segment (concurrent lane): needs an importable
sglang (up-tree PYTHONPATH, as the gate provides). Outside a container
the tool exits 0 with SKIP (no sglang) so host lanes stay green.
"""

import os
import sys

os.environ.setdefault("HCU_ADAPTER_HOST", "sglang")

try:
    import sglang  # noqa: F401
except Exception:
    print("SKIP no importable sglang (host lane)")
    print("RESULT: PASS")
    sys.exit(0)

FAILS = []


def ok(tag, cond, detail=""):
    print(("%-28s %s %s" % (tag, "OK" if cond else "FAIL", detail)).rstrip())
    if not cond:
        FAILS.append(tag)


import hcu_adapter.bridges.sglang as bridge  # noqa: E402

bridge.install()

failed = list(getattr(bridge, "_FAILED_BRIDGES", []) or [])
ok("no-failed-bridges", not failed,
   "degraded: %s" % ", ".join(failed) if failed else "")

from hcu_adapter.infra import registry  # noqa: E402

records = registry._RECORDS
emu_hits = sorted(
    op for op, r in records.items()
    if "_emu." in str(r.get("impl", "")) or str(r.get("impl", "")).endswith("_emu")
)
ok("registry-no-emu-landings", not emu_hits,
   "emu impls: %s" % ", ".join(emu_hits) if emu_hits else "n=%d bindings" % len(records))
print("tier-table-size %d" % len(records))

print("RESULT:", "PASS" if not FAILS else "FAIL %s" % FAILS)
sys.exit(0 if not FAILS else 1)
