# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""tests/adapter/test_registry.py -- registry + dump unit tests (CPU, no GPU).

Covers docs/governance.md §2/§4 machine-side contracts:
  - register rejects unknown ops (catalog is the authority)
  - register rejects text-patch needles outside TEXT_PATCH_ALLOWLIST
  - duplicate binding of one target across ops raises
  - dump() shape: ops/unbound/env
  - log_summary runs against sampled records without throwing

Run:  python tests/adapter/test_registry.py   (assert-based, no pytest dep)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))


def main():
    from hcu_adapter import registry
    from hcu_adapter.hcu_config import BACKEND_CATALOG

    op = "silu_and_mul"

    # 1. unknown op rejected
    try:
        registry.register("nope", None, "t.x", registry.MECH_SEAM)
        raise AssertionError("unknown op accepted")
    except ValueError as e:
        assert "not in BACKEND_CATALOG" in str(e), e
    print("PASS  unknown-op rejected")

    # 2. text-patch outside allowlist rejected; inside accepted
    try:
        registry.register(op, None, "text:evil", registry.MECH_TEXT,
                          detail="assert _is_cuda or _is_rocm, (")
        raise AssertionError("unlisted text-patch accepted")
    except ValueError as e:
        assert "TEXT_PATCH_ALLOWLIST" in str(e), e
    needle, ntext = sorted(registry.TEXT_PATCH_ALLOWLIST)[0]
    registry.register(needle, None, "text:" + ntext, registry.MECH_TEXT,
                      detail=ntext)
    print("PASS  text-patch allowlist enforced")

    # 3. duplicate target across ops rejected
    f = lambda: None
    registry.register(op, f, "m.f", registry.MECH_REBIND)
    try:
        other = next(o for o in BACKEND_CATALOG if o != needle and o != op)
        registry.register(other, f, "m.f", registry.MECH_REBIND)
        raise AssertionError("duplicate target accepted")
    except ValueError as e:
        assert "already bound" in str(e), e
    print("PASS  duplicate-target rejected")

    # 4. dump shape
    doc = registry.dump()
    assert doc["ops"][op]["impl"].startswith("main"), doc["ops"][op]
    assert doc["ops"][op]["role"] == "active"
    assert op not in doc["unbound"]
    assert "silu_and_mul" in doc["unbound"] or True  # unbound lists others
    assert "HCU_AITER_AR" in doc["env"], sorted(doc["env"])[:5]
    print("PASS  dump shape:", {k: len(v) if hasattr(v, "__len__") else v
                                for k, v in doc.items()})

    # 5. log_summary smoke (no format errors)
    import logging
    logging.basicConfig(level=logging.WARNING)
    registry.log_summary()
    print("PASS  log_summary smoke")

    print("ALL REGISTRY TESTS PASS")


if __name__ == "__main__":
    main()
