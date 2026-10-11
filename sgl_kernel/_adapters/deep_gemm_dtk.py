"""Alias the DTK ``deepgemm`` wheel as upstream ``deep_gemm`` (EP bridge).

The DTK stack ships Hygon's deepgemm fork (2.1.0+dtk2604) under the
one-word package name ``deepgemm`` with upstream-compatible entry points
(``m_grouped_bf16_gemm_nt_masked`` / ``m_grouped_fp8_gemm_nt_masked[_ll]``
/ ``..._contiguous`` etc.). sglang only ever imports ``deep_gemm``, so on
this stack the DeepGEMM-backed MoE runners (the only deepep_ll-capable
runners besides aiter/npu ones) are unreachable.

``install()`` (called from sitecustomize under ``HCU_DEEPGEMM_BRIDGE=1``)
builds a ``deep_gemm`` module in ``sys.modules`` backed by the DTK package
plus the ``deep_gemm.utils.layout`` helper module the wrapper imports at
module load. The utils.layout scale transforms have no verified DTK
equivalent — they raise loudly instead of silently computing wrong
numbers, which keeps fp8 DeepEP off the table until validated (bf16
grouped GEMM needs no scales).
"""

from __future__ import annotations

import sys
import types


def _make_layout_module() -> types.ModuleType:
    mod = types.ModuleType("deep_gemm.utils.layout")

    def _unimplemented(name):
        def _raise(*args, **kwargs):
            raise NotImplementedError(
                f"deep_gemm.utils.layout.{name}: no verified DTK deepgemm "
                "equivalent (fp8 DeepEP scale layouts unvalidated on HCU); "
                "bf16 grouped GEMM paths must not reach here"
            )

        return _raise

    mod.get_mn_major_tma_aligned_tensor = _unimplemented(
        "get_mn_major_tma_aligned_tensor"
    )
    mod.get_col_major_tma_aligned_tensor = _unimplemented(
        "get_col_major_tma_aligned_tensor"
    )
    mod.transform_sf_into_required_layout = _unimplemented(
        "transform_sf_into_required_layout"
    )
    return mod


def install():
    """Bind ``deep_gemm`` (+ ``deep_gemm.utils.layout``) to the DTK wheel.

    No-op when a real ``deep_gemm`` is importable or an alias already
    exists, so a future upstream-wheel install wins automatically.
    """
    if "deep_gemm" in sys.modules:
        return sys.modules["deep_gemm"]
    try:
        import importlib.util

        if importlib.util.find_spec("deep_gemm") is not None:
            return None  # real deep_gemm present; do not shadow it
    except Exception:
        pass
    import deepgemm

    shim = types.ModuleType("deep_gemm")
    shim.__dict__.update(
        {k: v for k, v in vars(deepgemm).items() if not k.startswith("__")}
    )
    shim.__package__ = "deep_gemm"
    shim.__file__ = getattr(deepgemm, "__file__", None)
    shim.dtk_deepgemm = deepgemm  # escape hatch to the raw fork

    _dtk_masked = getattr(deepgemm, "m_grouped_bf16_gemm_nt_masked", None)
    _dtk_pack_bf16 = getattr(deepgemm, "pack_bf16_masked_weight", None)
    if _dtk_masked is not None:

        _bf16_packed = {}  # (data_ptr, shape, dtype) -> packed weight

        def _bf16_masked(a, b, d, masked_m=None, expected_m=None,
                         expected_m_per_group=None, config=None, **unknown):
            # sglang names the 5th arg `expected_m` (entrypoint positional,
            # compile_utils warmup passes it as a kwarg); the DTK fork calls
            # it `expected_m_per_group` and puts `config` last. The default
            # masked mode (2000, BM64/BN256/BK128 HIPC) reads the weight in a
            # packed layout — raw (G,N,K) weights give garbage numbers, so
            # weights are packed once and cached by storage identity.
            if unknown:
                raise TypeError(
                    f"m_grouped_bf16_gemm_nt_masked: unexpected kwargs "
                    f"{sorted(unknown)}"
                )
            if expected_m is None:
                expected_m = expected_m_per_group
            if masked_m is None or expected_m is None:
                raise TypeError(
                    "m_grouped_bf16_gemm_nt_masked: masked_m and "
                    "expected_m are required"
                )
            if _dtk_pack_bf16 is not None and b.dim() == 3:
                key = (b.data_ptr(), tuple(b.shape), b.dtype)
                packed = _bf16_packed.get(key)
                if packed is None:
                    packed = _dtk_pack_bf16(b.contiguous())
                    _bf16_packed[key] = packed
                b = packed
            if config is not None:
                return _dtk_masked(a, b, d, masked_m, expected_m, config)
            return _dtk_masked(a, b, d, masked_m, expected_m)

        shim.m_grouped_bf16_gemm_nt_masked = _bf16_masked

    utils = types.ModuleType("deep_gemm.utils")
    layout = _make_layout_module()
    utils.layout = layout
    shim.utils = utils

    sys.modules["deep_gemm"] = shim
    sys.modules["deep_gemm.utils"] = utils
    sys.modules["deep_gemm.utils.layout"] = layout
    return shim
