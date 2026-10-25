import pathlib, subprocess, inspect
bodies = {'cuda_bf16.h': '// HCU adapter cuda->hip forwarder\n#pragma once\n#include <hip/hip_bf16.h>\nusing __nv_bfloat16 = __hip_bfloat16;\n', 'cuda_fp8.h': '// HCU adapter cuda->hip forwarder\n#pragma once\n#include <hip/hip_fp8.h>\nusing __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;\nusing __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;\n#define __NV_E4M3 __HIP_E4M3\n#define __NV_SATFINITE __HIP_SATFINITE\n#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2\n#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2\n'}
inc = pathlib.Path("/tmp/fwtest"); inc.mkdir(exist_ok=True)
for n, b in bodies.items():
    (inc / n).write_text(b)
tu = inc / "tu.cc"
tu.write_text('#include <hip/hip_runtime.h>\n#include <cuda_fp8.h>\n#include <cuda_bf16.h>\n__global__ void k(float2 f, float x){\n  __nv_fp8x2_e4m3 v(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3));\n  __nv_bfloat16 b = __nv_bfloat16(x);\n  __nv_fp8x2_storage_t s = v.__x;\n  (void)s; (void)b;\n}')
print("LAYOUT /opt/dtk/include/hip/hip_bf16.h:", pathlib.Path("/opt/dtk/include/hip/hip_bf16.h").exists(),
      "| /opt/dtk/hip/include/hip/hip_bf16.h:", pathlib.Path("/opt/dtk/hip/include/hip/hip_bf16.h").exists())
r = subprocess.run(["/opt/dtk/bin/hipcc", "-c", str(tu), "-I", str(inc), "-I", "/opt/dtk/include", "-o", "/tmp/tu.o"], capture_output=True, text=True)
print("VAL1_RC=", r.returncode)
if r.returncode: print(r.stderr[-500:])
else: print("VAL1=nv-token device TU compiled clean")
try:
    from sglang.srt.managers.scheduler_components.invariant_checker import create_scheduler_watchdog as f
    print("VAL2_SIG=", inspect.signature(f))
except Exception as e:
    print("VAL2=FAIL", repr(e))