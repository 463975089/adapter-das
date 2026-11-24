# SPDX-License-Identifier: Apache-2.0
"""Execute actual mapped-host validation with explicit HIP/Tensor query doubles.

This checks span walking and overflow/identity rules, not HIP mapping or GPU
copy correctness. Real registration/gather tests run separately.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("refused"); } while (0)
namespace c10 { constexpr int kStrided = 0; }
namespace at {
struct Device { bool cpu; bool is_cpu() const { return cpu; } };
struct Pointer { uintptr_t value; void* get() const { return reinterpret_cast<void*>(value); } };
struct Storage {
  uintptr_t base; size_t bytes;
  size_t nbytes() const { return bytes; }
  Pointer data_ptr() const { return {base}; }
};
struct Tensor {
  Storage backing; int64_t offset = 0; int64_t count = 8192;
  bool dense = true; bool cpu = true;
  Device device() const { return {cpu}; }
  int layout() const { return c10::kStrided; }
  int dim() const { return 2; }
  bool is_contiguous() const { return dense; }
  size_t element_size() const { return 1; }
  int64_t numel() const { return count; }
  int64_t storage_offset() const { return offset; }
  Storage storage() const { return backing; }
  const void* const_data_ptr() const {
    return reinterpret_cast<const void*>(backing.base + offset);
  }
};
}
using hipError_t = int;
constexpr int hipSuccess = 0, hipMemoryTypeHost = 1;
constexpr int HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 1;
constexpr int HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 2;
constexpr int HIP_POINTER_ATTRIBUTE_MAPPED = 3;
constexpr int HIP_POINTER_ATTRIBUTE_BUFFER_ID = 4;
struct hipPointerAttribute_t { int memoryType = 0; bool isManaged = false; };
struct Region {
  uintptr_t host; size_t bytes; uintptr_t alias; uint64_t id;
  unsigned mapped = 1; uint64_t device_id = 0;
};
std::vector<Region> regions;
Region* host_region(uintptr_t p) {
  for (auto& r : regions) if (p >= r.host && p - r.host < r.bytes) return &r;
  return nullptr;
}
Region* device_region(uintptr_t p) {
  for (auto& r : regions) if (p >= r.alias && p - r.alias < r.bytes) return &r;
  return nullptr;
}
hipError_t hipGetLastError() { return 0; }
const char* hipGetErrorString(hipError_t) { return "query-double failure"; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* out, void* p) {
  if (!host_region(reinterpret_cast<uintptr_t>(p))) return 1;
  out->memoryType = hipMemoryTypeHost; out->isManaged = false; return 0;
}
hipError_t hipPointerGetAttribute(void* out, int attribute, void* pointer) {
  uintptr_t p = reinterpret_cast<uintptr_t>(pointer);
  Region* host = host_region(p); Region* device = device_region(p);
  Region* r = host ? host : device;
  if (!r) return 1;
  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR:
      *static_cast<void**>(out) = reinterpret_cast<void*>(r->host); break;
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE: *static_cast<size_t*>(out) = r->bytes; break;
    case HIP_POINTER_ATTRIBUTE_MAPPED: *static_cast<unsigned*>(out) = r->mapped; break;
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID:
      *static_cast<uint64_t*>(out) = !host && r->device_id ? r->device_id : r->id; break;
    default: return 1;
  }
  return 0;
}
hipError_t hipHostGetDevicePointer(void** out, void* pointer, unsigned) {
  uintptr_t p = reinterpret_cast<uintptr_t>(pointer); Region* r = host_region(p);
  if (!r) return 1;
  *out = reinterpret_cast<void*>(r->alias + (p - r->host)); return 0;
}
'''

MAIN = r'''
int main() {
  const uintptr_t h = 0x10000, d = 0x80000;
  at::Tensor tensor{{h, 16384}};
  auto mapped = [&]() { return mapped_host_spans(tensor).front().device; };
  auto refused = [&]() {
    bool rejected = false;
    try { (void)mapped(); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid registration accepted");
  };
  regions = {{h, 16384, d, 1}};
  if (mapped() != d) return 1;
  // Different buffer IDs are legal across adjacent registered spans.
  regions = {{h, 4096, d, 1}, {h + 4096, 4096, d + 4096, 2}};
  if (mapped() != d) return 2;
  // Interior view crosses three registrations; alias stays linear.
  regions = {{h, 4096, d, 1}, {h + 4096, 4096, d + 4096, 2},
             {h + 8192, 8192, d + 8192, 3}};
  tensor.offset = 16; tensor.count = 12288;
  const auto linear = mapped_host_spans(tensor);
  if (linear.size() != 1 || linear[0].begin != 0 ||
      linear[0].end != 12288 || linear[0].device != d + 16) return 3;
  regions.erase(regions.begin() + 1); refused(); // registered endpoints, interior gap
  regions = {{h, 4096, d, 1}, {h + 4096, 12288, d + 8192, 2}};
  const auto split = mapped_host_spans(tensor);
  if (split.size() != 2 || split[0].begin != 0 || split[0].end != 4080 ||
      split[1].begin != 4080 || split[1].end != 12288 ||
      split[1].device != d + 8192) return 6;
  regions[1].alias = d + 4096; regions[1].device_id = 99; refused();
  regions[1].device_id = 0; regions[1].mapped = 0; refused();
  regions[1].mapped = 1;
  if (mapped() != d + 16) return 4;
  regions.pop_back(); refused(); // no cached acceptance after unregister
  tensor.offset = 0; tensor.count = 0;
  if (mapped() != d) return 5; // empty view still validates registered anchor
  regions.clear(); refused();
  regions = {{h, 16384, d, 1}};
  tensor.count = 8192; tensor.offset = 12288; refused(); // outside storage
  tensor.offset = 0; tensor.dense = false; refused();
  tensor.dense = true; tensor.cpu = false; refused();
  tensor.cpu = true; tensor.backing = {UINTPTR_MAX - 15, 16384}; refused();
  tensor.backing = {h, 16384};
  regions = {{h, 16384, UINTPTR_MAX - 15, 1}}; refused();
  std::cout << "mapped-span positives/negative guards passed\n";
}
'''


def test_actual_host_span_validation_accepts_adjacent_registrations(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable; not a HIP result")
    source = (ROOT / "csrc/hisparse/hisparse_cpp.cu").read_text()
    body = source[source.index("void check_host_query("):source.index("void same_device(")]
    path = tmp_path / "mapped_span.cc"
    path.write_text(PREFIX + body + MAIN)
    executable = tmp_path / "mapped_span"
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
         "-fno-omit-frame-pointer", str(path), "-o", str(executable)],
        capture_output=True, text=True, timeout=90)
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mapped-span positives/negative guards passed" in result.stdout
