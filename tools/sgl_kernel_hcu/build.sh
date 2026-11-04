#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Build sglang-kernel (sgl_kernel) for HCU/DTK from the official v0.5.22 aot
# tree — no source-tree modification: everything happens in a workspace copy.
#
# usage (inside the dtk2610 py3.12 container):
#   bash build.sh                 # detect GPU target (default gfx938)
#   HCU_GPU_TARGET=gfx938 bash build.sh
#
# artifacts: $WS/dist/sglang_kernel-*.whl   (WS default /home/build_sgl_kernel)
set -euo pipefail

TOOLS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SRC_TREE="${SRC_TREE:-/home/sglang-v0522/python/sglang/kernels/aot}"
WS="${WS:-/home/build_sgl_kernel}"

echo "== workspace: $WS (from $SRC_TREE) =="
rm -rf "$WS"
mkdir -p "$WS"
cp -r "$SRC_TREE" "$WS/aot"

echo "== fetch flashinfer headers (commit pinned by upstream CMakeLists) =="
FI_COMMIT=bc29697ba20b7e6bdb728ded98f04788e16ee021
FI_SHA256=931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d
mkdir -p "$WS/deps"
curl -fsSL -o "$WS/deps/fi.tar.gz" \
  "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/$FI_COMMIT"
echo "$FI_SHA256  $WS/deps/fi.tar.gz" | sha256sum -c -
tar xzf "$WS/deps/fi.tar.gz" -C "$WS/deps"
mv "$WS/deps/flashinfer-$FI_COMMIT" "$WS/deps/flashinfer"
mkdir -p "$WS/aot/3rdparty"
ln -s "$WS/deps/flashinfer" "$WS/aot/3rdparty/flashinfer"
fi_utils="$(find "$WS/deps/flashinfer" -name pytorch_extension_utils.h | head -1)"
[ -n "$fi_utils" ] || { echo "flashinfer fetch incomplete: pytorch_extension_utils.h not found"; exit 1; }

echo "== patch flashinfer PTX math (GCN port) =="
# The pinned commit ships math.cuh (exactly 10 asm volatile statements);
# math_hip.cuh only exists in newer flashinfer — pointing there was the
# round-7 FileNotFoundError.
python3 "$TOOLS/patch_math_hip.py" "$WS/deps/flashinfer/include/flashinfer/math.cuh" || exit 1

echo "== patch loader (top-level .so glob) =="
python3 "$TOOLS/patch_load_utils.py" "$WS/aot/python/sgl_kernel/load_utils.py"

echo "== minimal cuda compat shims (flashinfer _hip header closure only) =="
# DTK's cuda-12 compat tree cannot go on the hipcc include path (clashes
# with hip's thrust/vector_types); we map only the spellings actually
# used, onto hip's native API. Shims are tracked in tools/sgl_kernel_hcu/shims.
SHIM="$WS/aot/3rdparty/cuda_compat"
mkdir -p "$SHIM"
cp "$TOOLS"/shims/*.h "$SHIM/"
ls "$SHIM"

echo "== patch dtk2610 wave64 masks (u32 -> u64 __shfl_*_sync masks) =="
# dtk2610's hip/amd_detail/amd_warp_sync_functions.h statically requires
# 64-bit masks on wave64; the sources default to 0xffffffffu (CUDA norm;
# dtk2604 accepted it). Patch the .cu PRE-hipify — torch's ROCm flow
# generates the .hip files from these at build time. Workspace copy only.
sed -i 's/unsigned mask = 0xffffffffu/unsigned long long mask = 0xffffffffffffffffull/g' \
  "$WS/aot/csrc/moe/moe_align_kernel.cu"
grep -n "mask = 0x" "$WS/aot/csrc/moe/moe_align_kernel.cu" | head -3 || true

echo "== build =="
cp "$TOOLS/setup_hcu.py" "$WS/aot/setup_hcu.py"
cd "$WS/aot"
python3 setup_hcu.py bdist_wheel --dist-dir "$WS/dist"

echo "== artifacts =="
ls -la "$WS/dist"
python3 - <<'EOF'
import glob, zipfile
w = sorted(glob.glob("/home/build_sgl_kernel/dist/*.whl"))[-1]
names = zipfile.ZipFile(w).namelist()
so = [n for n in names if n.endswith(".so")]
py = [n for n in names if n.endswith("load_utils.py")]
print("wheel:", w)
print("payload .so:", so)
print("has load_utils:", bool(py), "| entries:", len(names))
EOF
echo "== done =="
