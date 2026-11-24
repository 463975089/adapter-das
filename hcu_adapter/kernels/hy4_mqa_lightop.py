# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP8 prefill scorer ABI translation; no framework or tensor conversions."""


def bind_prefill(mqa):
    def prefill(Q, KV, kv_scales, weights, cu_starts, cu_ends):
        return mqa(Q, KV, weights, cu_starts, cu_ends, kv_scales,
                   clean_logit=True)
    return prefill
