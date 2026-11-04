# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Index backend facade（门面）; preserves the existing backend selection."""


def expand_swa_index(pivots, out, *, backend):
    if backend == "torch":
        from .index_ops_emu import _torch_expand_swa_index

        return _torch_expand_swa_index(pivots, out)
    from .index_ops_triton import expand_swa_index as implementation

    implementation(pivots, out)
