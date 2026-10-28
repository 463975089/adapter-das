"""HCU adapter: gate DSV2/V4 capture-mode dual-stream MoE off on real HIP.

``DeepseekV2MoE.forward`` selects ``forward_normal_dual_stream`` ONLY under
``get_is_capture_mode()`` (the shared expert overlaps the routed experts on
``alt_stream``). On DTK/ROCm (gfx938) the cross-stream captured graph replays
with a broken shared-expert dependency: every replayed decode token becomes
deterministic garbage ('#####') while eager (single-stream
``forward_normal``) stays perfect. Eager is unaffected by this gate -- the
dual-stream branch is capture-only by construction.

The gate must test the true device fact (``torch.version.hip``): the
platform_seam (governance §0.5) flips sglang's ``is_hip()``/``_is_hip`` to
False in the server process, so an ``_is_hip``-based gate never fires.
Located: DSV4-Flash nmz16 2026-10-21 (fingerprint bisect: layer-0 post-attn
boundary all bit-exact; MoE in-place write-back showed the corruption;
moepath markers proved the captured path; full account in docs/models.md).
"""

import logging
import os
import pathlib
import tempfile

LOG = logging.getLogger("hcu.adapter")

_NEEDLE = """                and not is_in_breakable_cuda_graph()
            ):
                return self.forward_normal_dual_stream("""

_REPL = """                and not is_in_breakable_cuda_graph()
                and torch.version.hip is None  # HCU moe_dualstream_gate: cross-stream captured MoE replays broken on DTK/ROCm
            ):
                return self.forward_normal_dual_stream("""


def _atomic_write(path: pathlib.Path, text: str) -> None:
    d = tempfile.mkdtemp(dir=str(path.parent), prefix=".moe_dsg_")
    try:
        tmp = pathlib.Path(d) / path.name
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        try:
            os.rmdir(d)
        except OSError:
            pass


def install():
    import sglang.kernels.jit as _jit_root
    import hcu_adapter.registry as _reg

    pkg_root = pathlib.Path(_jit_root.__file__).parents[2]
    p = pkg_root / "srt" / "models" / "deepseek_v2.py"
    if not p.exists():
        LOG.info("moe_dualstream_gate: deepseek_v2.py absent; skip")
        return
    text = p.read_text(encoding="utf-8")
    if _REPL in text:
        return  # already patched (idempotent)
    if _NEEDLE not in text:
        if "torch.version.hip is None  # HCU moe_dualstream_gate" in text:
            return  # upstream refactored around our own gate; leave it
        raise RuntimeError(
            "moe_dualstream_gate: needle missing in deepseek_v2.py --"
            " upstream changed; re-derive (governance §3)"
        )
    _atomic_write(p, text.replace(_NEEDLE, _REPL, 1))
    _reg.register(
        "platform_detect",
        "dtk-hipcc-portability",
        target="text:sglang/srt/models/deepseek_v2.py",
        mechanism=_reg.MECH_TEXT,
        detail="capture-mode dual-stream MoE gated off on real HIP",
    )
    LOG.warning(
        "HCU adapter: DSV2 capture-mode dual-stream MoE gated off on HIP"
        " (cross-stream graph replay corruption, gfx938)"
    )
