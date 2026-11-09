# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single source of the sgl-kernel dist version (governance §7.4).

The shadow kernel wheel reads its version from here; the hcu-adapter dist
carries an independent version line of its own (packaging/hcu-adapter/
VERSION). The interlock pin in packaging/hcu-adapter/pyproject.toml must
equal THIS version (ci/l0_gate checks it; packaging/build.sh re-verifies).

main carries an unpinned dev version on purpose: the baseline is decided
when a release branch is cut, tracking the *kernel* version of the paired
upstream sglang tree (aot pyproject + dependency pin + engine.py probe,
NOT the sglang release number). Example: release/sglang_v0.5.22 pairs with
v0.5.22 whose aot is 0.4.9 => 0.4.9+hcu.1 on that branch.
"""

__version__ = "0.0.0+hcu.dev0"
