"""HCU adapter: fuse the CT int8 per-channel scales into wqkv_a (DSV4).

v0522 ``deepseek_v4.py`` load_weights fuses checkpoint ``wq_a``/``wkv``
pairs into the model's fused ``wqkv_a`` param (SGLANG_OPT_FUSE_WQA_WKV on
for the DSV4.1 family). The fuse branch catches ``.weight`` /
``.weight_scale_inv`` (fp8 naming) / ``.qweight`` (GGUF) — but NOT
``.weight_scale``, the compressed-tensors **INT8 per-channel scale**
name. DAS w4a8 checkpoints (DeepSeek-V4.1-Flash-Channel-INT4-w4a8) carry
exactly that name: the fused ``wqkv_a.weight`` gets its data but the q/kv
scales fall through to "not found in params_dict" and are dropped, so
``wqkv_a.weight_scale`` (created by CompressedTensorsW8A8Int8
.create_weights, verified v0522 compressed_tensors_w8a8_int8.py:78) never
receives data — every attention projection dequantizes with an
uninitialized scale. Boot-fine, numerics-garbage; nmz16 2026-10-26
(warnings during shard load; scheme/target forensics same day: the
WeigthsMapper rewrites the group targets to ``self_attn.(?:wqkv_a|wq_b)``
so the unfused names have no scheme either — disabling the fuse does NOT
dodge this, it makes wq_a/wkv fully unquantized instead).

Platform-independent upstream gap (CUDA serves this checkpoint family
with the same dropped scales); this needle holds the line until an
upstream PR lands. Gate: ``HCU_DSV4_WQKV_SCALE=0`` skips.
"""

import logging
import os
import pathlib
import tempfile

LOG = logging.getLogger("hcu.adapter")

_NEEDLE = (
    '                                    or name.endswith(".wq_a.weight_scale_inv")\n'
    '                                    or name.endswith(".wkv.weight")\n'
    '                                    or name.endswith(".wkv.weight_scale_inv")'
)

_REPL = (
    '                                    or name.endswith(".wq_a.weight_scale_inv")\n'
    '                                    or name.endswith(".wq_a.weight_scale")  # HCU dsv4_wqkv_scale: CT int8 per-channel scale\n'
    '                                    or name.endswith(".wkv.weight")\n'
    '                                    or name.endswith(".wkv.weight_scale_inv")\n'
    '                                    or name.endswith(".wkv.weight_scale")  # HCU dsv4_wqkv_scale: CT int8 per-channel scale'
)

_DETAIL = "dsv4 wqkv_a fuse: accept CT int8 per-channel .weight_scale shards"


def _atomic_write(path: pathlib.Path, text: str) -> None:
    d = tempfile.mkdtemp(dir=str(path.parent), prefix=".dsv4_ws_")
    try:
        tmp = pathlib.Path(d) / path.name
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        try:
            os.rmdir(d)
        except OSError:
            pass


def _apply(text: str) -> str:
    """Pure transform (test surface): idempotent needle patch.

    Already-patched or upstream-refactored-around-marker -> input
    unchanged; genuine needle drift -> RuntimeError (governance §3).
    """
    if _REPL in text:
        return text
    if _NEEDLE not in text:
        if "HCU dsv4_wqkv_scale" in text:
            return text  # our marker present: upstream PR landed / refactored
        raise RuntimeError(
            "dsv4_wqkv_scale_patch: needle missing in deepseek_v4.py --"
            " upstream changed; re-derive (governance §3)"
        )
    return text.replace(_NEEDLE, _REPL, 1)


def install():
    from sgl_kernel._adapters import hcu_env

    if not hcu_env.enabled("HCU_DSV4_WQKV_SCALE"):
        return

    import sglang.kernels.jit as _jit_root
    import sgl_kernel._adapters.registry as _reg

    pkg_root = pathlib.Path(_jit_root.__file__).parents[2]
    p = pkg_root / "srt" / "models" / "deepseek_v4.py"
    if not p.exists():
        LOG.info("dsv4_wqkv_scale_patch: deepseek_v4.py absent; skip")
        return
    text = p.read_text(encoding="utf-8")
    new = _apply(text)
    if new != text:
        _atomic_write(p, new)
        LOG.warning(
            "HCU adapter: dsv4 wqkv_a scale-fuse patch installed (CT int8 "
            ".weight_scale shards now fused; HCU_DSV4_WQKV_SCALE=0 to skip)"
        )
    _reg.register(
        "platform_detect",
        "dsv4-wqkv-scale-fuse",
        target="text:sglang/srt/models/deepseek_v4.py",
        mechanism=_reg.MECH_TEXT,
        detail=_DETAIL,
    )
