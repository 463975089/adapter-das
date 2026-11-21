# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""family-override modules _register_for interception (incl. deepseek_v2 DSA branch)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations
from hcu_adapter.infra import hcu_env



def install(_he=None) -> None:
    # Family-override modules import `_register_for` by reference at their own
    # import time (later than ours), so patching the attribute here intercepts
    # every registration, including deepseek_v2's DSA branch.
    try:
        import functools
        import logging

        from sglang.srt.arg_groups import model_override_base as _mob

        def _hcu_dsa_index_kpool(hf_config):
            for cfg in (hf_config, getattr(hf_config, "text_config", None)):
                kpool = getattr(cfg, "index_kpool", None)
                if kpool:
                    return int(kpool)
            return None

        def _hcu_wrap_override(fn):
            @functools.wraps(fn)
            def wrapped(server_args, hf_config):
                overrides = fn(server_args, hf_config)
                if overrides.get("page_size") != 1:
                    return overrides
                wanted = hcu_env.resolve("HCU_DSA_PAGE_SIZE")[0] or getattr(
                    server_args, "page_size", None
                )
                try:
                    wanted = int(wanted)
                except (TypeError, ValueError):
                    return overrides
                if wanted <= 1:
                    return overrides
                kpool = _hcu_dsa_index_kpool(hf_config)
                if kpool and wanted % kpool:
                    logging.getLogger("hcu.adapter").warning(
                        "HCU adapter: page_size %d not a multiple of index_kpool"
                        " %d, keeping upstream default 1",
                        wanted,
                        kpool,
                    )
                    return overrides
                overrides["page_size"] = wanted
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: keeping page_size=%d for DSA (upstream legacy"
                    " HIP default is 1, index_kpool=%s)",
                    wanted,
                    kpool,
                )
                return overrides

            return wrapped

        _hcu_orig_register_for = _mob._register_for

        def _hcu_register_for(*args, **kwargs):
            decorator = _hcu_orig_register_for(*args, **kwargs)

            def register(fn):
                return decorator(_hcu_wrap_override(fn))

            return register

        _mob._register_for = _hcu_register_for
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSA page_size bridge not installed", exc_info=True
        )
