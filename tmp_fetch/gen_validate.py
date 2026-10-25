"""Regenerate tmp_fetch/validate_box16.py: compile the adapter's cuda_fp8.h/
cuda_bf16.h forwarder bodies (with __nv_* aliases) against real DTK hipcc on
box16, plus watchdog-factory signature probe. Run locally, push via base64."""
import pathlib
import sys

sys.path.insert(0, ".")
from sgl_kernel._adapters import jit_compat  # noqa: E402

mark = jit_compat._MARK
bodies = {
    n: mark + "\n" + "\n".join(l) + "\n"
    for n, l in jit_compat._FORWARDS.items()
    if n in ("cuda_fp8.h", "cuda_bf16.h")
}
N = chr(10)
tu = N.join([
    "#include <hip/hip_runtime.h>",  # real JIT csrc always has runtime first
    "#include <cuda_fp8.h>",
    "#include <cuda_bf16.h>",
    "__global__ void k(float2 f, float x){",
    "  __nv_fp8x2_e4m3 v(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3));",
    "  __nv_bfloat16 b = __nv_bfloat16(x);",
    "  __nv_fp8x2_storage_t s = v.__x;",
    "  (void)s; (void)b;",
    "}",
])
script = N.join([
    "import pathlib, subprocess, inspect",
    "bodies = " + repr(bodies),
    'inc = pathlib.Path("/tmp/fwtest"); inc.mkdir(exist_ok=True)',
    "for n, b in bodies.items():",
    "    (inc / n).write_text(b)",
    'tu = inc / "tu.cc"',
    "tu.write_text(" + repr(tu) + ")",
    'print("LAYOUT /opt/dtk/include/hip/hip_bf16.h:", pathlib.Path("/opt/dtk/include/hip/hip_bf16.h").exists(),',
    '      "| /opt/dtk/hip/include/hip/hip_bf16.h:", pathlib.Path("/opt/dtk/hip/include/hip/hip_bf16.h").exists())',
    'r = subprocess.run(["/opt/dtk/bin/hipcc", "-c", str(tu), "-I", str(inc), "-I", "/opt/dtk/include", "-o", "/tmp/tu.o"], capture_output=True, text=True)',
    'print("VAL1_RC=", r.returncode)',
    "if r.returncode: print(r.stderr[-500:])",
    'else: print("VAL1=nv-token device TU compiled clean")',
    "try:",
    "    from sglang.srt.managers.scheduler_components.invariant_checker import create_scheduler_watchdog as f",
    '    print("VAL2_SIG=", inspect.signature(f))',
    "except Exception as e:",
    '    print("VAL2=FAIL", repr(e))',
])
pathlib.Path("tmp_fetch/validate_box16.py").write_text(script, encoding="utf-8")
print("regen ok, bytes:", len(script))
