# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""aiter JIT cache observability (boot line; new-env prewarm hookup).

The DTK vendor wheel ships every aiter op prebuilt -- the JIT cache IS the
package: ``aiter/jit/module_<op>.so`` files are keyed by op NAME (no content
hash), a fresh container imports aiter in seconds with zero compiles, and
``get_user_jit_dir()`` serves the package dir itself (writable) or
``$AITER_JIT_DIR`` when redirected. There is therefore nothing to pre-warm
for ops the wheel already carries.

A runtime compile only ever appears when an op or template arrives that the
wheel does not prebuild; artifacts land in ``aiter/jit/build/`` (minutes
each). Surface both populations at boot so they get harvested into the
source of truth or the wheel gets pinned. Harvest rule: same wheel version
ONLY -- a stale name-keyed ``.so`` dropped into a newer wheel would shadow
its recompile instead of upgrading the kernel.
"""

import glob
import logging
import os


def report():
    logger = logging.getLogger("hcu.adapter")
    spec_dir = None
    try:
        import importlib.util

        spec = importlib.util.find_spec("aiter")
        if spec is not None and spec.submodule_search_locations:
            spec_dir = list(spec.submodule_search_locations)[0]
    except Exception:
        spec_dir = None
    if not spec_dir or not os.path.isdir(spec_dir):
        return
    jdir = os.path.join(spec_dir, "jit")
    if not os.path.isdir(jdir):
        return

    version = "unknown"
    dist_root = os.path.dirname(spec_dir)
    infos = sorted(glob.glob(os.path.join(dist_root, "aiter-*.dist-info")))
    if infos:
        name = os.path.basename(infos[-1])
        version = name[len("aiter-"):-len(".dist-info")]

    prebuilt = len(glob.glob(os.path.join(jdir, "module_*.so")))
    build_dir = os.path.join(jdir, "build")
    runtime_built = []
    if os.path.isdir(build_dir):
        for p in sorted(glob.glob(os.path.join(build_dir, "**", "*"),
                                  recursive=True)):
            if os.path.isfile(p):
                runtime_built.append(os.path.relpath(p, jdir))
    jit_dir_env = os.environ.get("AITER_JIT_DIR", "")

    if runtime_built:
        logger.warning(
            "HCU aiter jit cache: wheel=%s prebuilt_so=%d RUNTIME_BUILT=%d"
            " under jit/build (harvest into the wheel's source of truth or"
            " pin this wheel version; name-keyed .so must not be seeded"
            " across wheel versions) AITER_JIT_DIR=%s",
            version, prebuilt, len(runtime_built), jit_dir_env or "unset")
        for rel in runtime_built[:5]:
            logger.warning("HCU aiter jit cache: + %s", rel)
    else:
        logger.info(
            "HCU aiter jit cache: wheel=%s prebuilt_so=%d runtime_built=0"
            " (wheel ships every op prebuilt; jit/build clean)"
            " AITER_JIT_DIR=%s",
            version, prebuilt, jit_dir_env or "unset")
