"""HCU compatibility bridges for the upstream sglang stack.

Loaded automatically (as `sitecustomize`) whenever PYTHONPATH includes this
directory, including spawn'd sglang scheduler children. sglang and vllm files
stay untouched; what we bridge here:

1. vllm._custom_ops.rms_norm / fused_add_rms_norm -> sgl_kernel triton kernels.
   The HIP vllm build's `torch.ops._C` extension does not carry these ops, so
   sglang's layernorm forward_hip would crash at call time (the import-time
   `_has_vllm_rms_norm` check passes). We replace the python wrappers with
   bridges to our kernel — same signatures, fp32-accumulated math.
2. `aiter` stub registration happens via `import sgl_kernel` (see
   sgl_kernel._aiter_hcu_stub) for every AMD-only import on the HIP path.
3. DSA model overrides (`sglang.srt.arg_groups.model_overrides.deepseek_v2`,
   which also owns Glm5Next) force `page_size=1` on legacy HIP (no aiter
   preshuffle paged-MQA). Models whose indexer pools keys (index_kpool > 1,
   e.g. GLM-5.3-Flash kpool=4) then die on `page_size % index_kpool != 0`.
   The memory pool only needs a multiple of index_kpool, so we wrap the
   override registry to keep a CLI-provided page_size when upstream forces 1.
"""

import os

import torch

if getattr(torch.version, "hip", None) is not None:
    try:
        import vllm._custom_ops as _vco

        from sgl_kernel import rmsnorm as _sk_rmsnorm

        if not hasattr(torch.ops._C, "rms_norm"):

            def rms_norm(out, input, weight, epsilon):
                _sk_rmsnorm(input, weight, epsilon, out=out)

            def fused_add_rms_norm(
                out, input, residual_out, residual, weight, epsilon
            ):
                # sglang calls the 6-arg HIP form: out = rmsnorm(residual+input),
                # residual_out = residual+input. vllm's 4-arg in-place form is
                # bridged into it without extra copies.
                residual_in = residual if residual_out is residual else residual_out
                residual_in.copy_(residual)
                residual_in.add_(input)
                _sk_rmsnorm(residual_in, weight, epsilon, out=out)

            _vco.rms_norm = rms_norm
            _vco.fused_add_rms_norm = fused_add_rms_norm
    except Exception:  # pragma: no cover - vllm absent: sglang falls back natively
        pass

    try:
        import sgl_kernel as _sk

        if not hasattr(_sk, "apply_token_bitmask_inplace_cuda"):
            # sglang's xgrammar shim takes the HIP branch and imports this from
            # sgl_kernel; bridge it to the in-tree triton op (CUDA branch).
            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton as _bitmask_triton,
            )

            _sk.apply_token_bitmask_inplace_cuda = _bitmask_triton
    except Exception:
        pass

    # Family-override modules import `_register_for` by reference at their own
    # import time (later than ours), so patching the attribute here intercepts
    # every registration, including deepseek_v2's DSA branch.
    try:
        import functools
        import logging
        import os

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
                wanted = os.environ.get("HCU_DSA_PAGE_SIZE") or getattr(
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

    # Force the DSA "aiter preshuffle paged-MQA" gate on. The DTK wheel ships
    # working triton preshuffle MQA-logits kernels (pa_mqa_logits etc.), but
    # upstream's gate keys on SGLANG_USE_AITER — setting that env would also
    # flip unrelated fp8 weight-shuffle/gemm routes. Rebinding the predicate
    # in its home module before any consumer's byname import copies it makes
    # only the DSA page-size/table/preshuffle-tile decisions follow the
    # preshuffle path (page_size=64*index_kpool, 16-aligned cache tiles).
    try:
        import sglang.srt.layers.attention.dsa.utils as _dsa_utils

        if os.environ.get("HCU_DSA_PRESHUFFLE") == "0":
            # Legacy page_size=1 / KVBlockSize=1 DSA path (paired with the
            # HCU_MQA_EMU torch emulation on gfx936-class DCUs).
            _dsa_utils.aiter_can_use_preshuffle_paged_mqa = lambda: False
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced OFF (legacy"
                " page_size=1 path)"
            )
        else:
            _dsa_utils.aiter_can_use_preshuffle_paged_mqa = lambda: True
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced on"
                " (DTK wheel kernels)"
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: preshuffle gate patch not installed", exc_info=True
        )

    # slimquant_w4a8 checkpoints (GLM-5-W4A8_V2_6 family): expert int4 unpack
    # + register the quant method, opt-in via HCU_SLIMQUANT=1.
    if os.environ.get("HCU_SLIMQUANT") == "1":
        try:
            from sgl_kernel._adapters import slimquant as _slimquant

            _slimquant.activate()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: slimquant bridge not installed", exc_info=True
            )

    # DSA indexer top-k: the sgl-kernel backend ops have no HIP build; bridge
    # to torch equivalents (see sgl_kernel._adapters.dsa_topk).
    try:
        import sgl_kernel as _sk
        import sgl_kernel._adapters.dsa_topk as _dsa_topk

        _dsa_topk.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: dsa topk bridge not installed", exc_info=True
        )

    # gfx936-class DCUs have no fp8 matrix unit and the DTK aiter MQA-logits
    # kernels hard-gate on gfx942/950: emulate paged/ragged fp8 MQA logits in
    # torch (dequant -> bf16 GEMM, fp32 scales). Legacy page_size=1 only.
    if os.environ.get("HCU_MQA_EMU") == "1":
        try:
            import sgl_kernel._adapters.mqa_logits_emu as _mqa_emu

            _mqa_emu.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: MQA-logits emu not installed", exc_info=True
            )
