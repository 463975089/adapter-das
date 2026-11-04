# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import traceback

try:
    import flash_attn
    print("flash_attn:", getattr(flash_attn, "__version__", "?"), flash_attn.__file__)
except Exception as e:
    print("flash_attn ERR:", repr(e)[:200])

try:
    import aiter
    print("aiter:", getattr(aiter, "__file__", "?"))
    print("aiter has flash_attn_varlen_func:", hasattr(aiter, "flash_attn_varlen_func"))
    import aiter.ops.triton
    cands = [n for n in dir(aiter.ops.triton) if "flash" in n.lower() or "attn" in n.lower()]
    print("aiter.ops.triton attn names:", cands[:20])
except Exception as e:
    print("aiter ERR:", repr(e)[:200])
    traceback.print_exc()
