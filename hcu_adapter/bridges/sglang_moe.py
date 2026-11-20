# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DTK aiter fused-MoE bridge (2026-10-11)

Extracted verbatim from bridges/sglang.py install() (10. DTK aiter fused-MoE bridge (2026-10-11): wraps the stock triton)
"""

# under layout governance 2026-11-18; zero logic change. Section notes:
# 10. DTK aiter fused-MoE bridge (2026-10-11): wraps the stock triton
# runner's fused_experts entry; w8a8 bf16 calls first try DTK
# aiter_moe (blockwise -> ASM tuned csv, channelwise -> moe_c),
# unlisted shapes fall back to the stock triton kernel (log-once).
# Knob: HCU_FUSED_MOE_IMPL = auto | aiter | triton (auto default).
# Gate: validation-host-18 BW200/gfx938 rel 0.02-0.04 vs bf16 ref (see
# hcu_adapter/aiter_moe.py docstring).
# 10.5. W4A16 MoE bridge (2026-10-19, R-05): same fused_experts hook
# surface as the aiter arm above, for the int4-w4a16 face of the
# stock moe_wna16 runner -- the DTK deepgemm masked w4a16 C++
# grouped GEMMs behind a dispatch/silu/combine pipeline.
# Explicit opt-in only: HCU_W4A16_MOE_IMPL=deepgemm (auto keeps
# the stock triton arm; capture/inplace calls stay on it too).
# 10.6. Channel-wise int4 MoE kwarg rewrite (2026-10-25, box16
# dsv41f-INT4 bring-up): same fused_experts hook surface as the
# two arms above. strategy="channel" (group=-1) arrives with
# block_shape=[0,-1]; the tree's triton gptq dequant branch only
# fires for block_shape[1] > 0, so u8 packed weights hit the
# plain kernel's tl.dot(bf16, u8) and triton aborts at compile
# time (first hit = decode graph capture). channel == one group
# per whole K row: rewrite to block_shape=[0, max(K_up, K_down)]
# (the [E, N, 1] scales the scheme produced already match), so
# the stock gptq branch serves it -- capture included. Only the
# use_int4_w4a16 + group<=0 face is touched.
# Knob: HCU_WNA16_CHANNEL_MOE = 1 | 0 (default 1).
# 10.7. moe_c W4A16 vendor tier (2026-10-26, box16 dsv41f-INT4): the
# outermost fused_experts arm. Drives DTK aiter's
# moe_c_moe_gemm_marlin_w4a16 vendor GEMMs (align -> GEMM1 ->
# exact dsv4 clamp-silu -> GEMM2 -> sum) for the int4 channel
# face: probe parity med 0.0055, M16 full-layer 1161us vs the
# stock triton gptq kernel's 8100us (7x; the triton floor is
# latency-bound, no sub-ms config exists). Wheel-law guards: g32
# grid (channel scales expanded by value), gemm1 (K/32)%16==0,
# N%256, I>=256 with gemm2 K zero-padded (nibble 8) to a 512
# multiple when I isn't one (2304 -> 2560, exact math). Falls
# through to the 10.6/10.5/10/stock chain on any miss.
# Knob: HCU_W4A16_MOE_IMPL = moe_c (explicit; auto keeps triton).
# Preserve CT activation quantization by default. Legacy compatibility
# routes are explicit opt-ins; native does not hide missing kernels.
from __future__ import annotations

import logging


def install(_he=None) -> None:
    try:
        from hcu_adapter.features.aiter import aiter_moe as _amo

        _amo.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: aiter fused-MoE bridge not installed", exc_info=True
        )

    try:
        from hcu_adapter.features.ct_w4a8 import w4a16_moe as _w16m

        _w16m.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: w4a16 MoE bridge not installed", exc_info=True
        )

    try:
        from hcu_adapter.features.ct_w4a8 import wna16_channel_moe as _wnch

        _wnch.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: wna16 channel-int4 rewrite not installed",
            exc_info=True,
        )

    try:
        from hcu_adapter.features.ct_w4a8 import moe_c_w4a16 as _mcw

        _mcw.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: moe_c w4a16 bridge not installed", exc_info=True
        )

    try:
        from hcu_adapter.features.ct_w4a8 import ct_w4a8_moe as _cw4a8

        _cw4a8.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: CT W4A8 MoE route bridge not installed",
            exc_info=True,
        )
