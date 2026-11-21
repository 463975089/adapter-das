// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HiSparse decode hot-buffer kernels for sparse MLA (GLM-5 / DeepSeek-V3.2).
//
// The swap-in algorithm is a port of SGLang's hisparse
// load_cache_to_device_buffer kernel (sgl jit_kernel/csrc/hisparse.cuh),
// adapted to vLLM addressing:
//  - tokens are keyed by their global KV slot id (block_table-converted
//    indexer output) instead of in-request positions, so no per-request
//    host-location table is needed: host pool row i is global slot i.
//  - each state row owns `region_stride == hot_size` LRU-managed slots;
//    page tables translate those slots to physical hot-cache rows.
//
// The kernels only move bytes; they are dtype agnostic.

// Explicit HIP port: no stableTorchABI, PTX or hardware-subgroup operations.
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>
#include <climits>

#include <algorithm>
#include <cstdint>
#include <optional>


namespace {

// A software copy group, NOT warpSize. No ballots/shuffles/lockstep are used;
// two such groups may share one wave64, or each may occupy a wave32.
constexpr int kWarpSize = 32;
// Empty key in the shared open-addressing table of host rows.
constexpr int32_t kHashEmpty = -1;

// Always use the device alias returned by HIP, never assume host UVA identity.
// The caller must retain pinned storage until work on the current stream ends.
const char* mapped_host_pointer(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "host_cache must be CPU memory");
  if (tensor.numel() == 0) return nullptr;
  hipPointerAttribute_t attr{};
  auto status = hipPointerGetAttributes(&attr, tensor.const_data_ptr());
  TORCH_CHECK(status == hipSuccess && attr.memoryType == hipMemoryTypeHost,
              "host_cache must be HIP-registered pinned CPU memory");
  void* alias = nullptr;
  status = hipHostGetDevicePointer(&alias,
      const_cast<void*>(tensor.const_data_ptr()), 0);
  TORCH_CHECK(status == hipSuccess && alias != nullptr,
              "host_cache must have a mapped HIP device alias");
  return static_cast<const char*>(alias);
}

void same_device(const at::Tensor& t, const at::Tensor& hot) {
  TORCH_CHECK(t.is_cuda() && t.device() == hot.device(),
              "HiSparse tensors must share one HIP device");
}

void check_grid(int64_t rows, int64_t columns) {
  TORCH_CHECK(rows >= 0 && rows <= INT32_MAX && columns >= 0 &&
              columns <= INT32_MAX, "HiSparse launch dimensions must fit int32");
}

void check_cache_layout(const at::Tensor& hot) {
  TORCH_CHECK(hot.dim() == 2 || hot.dim() == 3, "hot cache must be 2D or 3D");
  const int64_t bytes = hot.size(-1) * hot.element_size();
  TORCH_CHECK(bytes > 0 && bytes % 16 == 0 && hot.stride(-1) == 1,
              "HiSparse requires dense, positive, 16-byte rows");
  TORCH_CHECK(hot.stride(-2) * hot.element_size() == bytes,
              "hot cache rows must be dense");
  if (hot.dim() == 3) {
    TORCH_CHECK(hot.size(1) > 0 && hot.size(1) <= INT32_MAX &&
                hot.size(0) * hot.size(1) < INT32_MAX &&
                hot.stride(0) >= hot.size(1) * hot.size(2),
                "hot cache pages must not overlap and physical rows must fit int32");
  } else {
    TORCH_CHECK(hot.size(0) < INT32_MAX, "staging rows must fit int32");
  }
}

__device__ __forceinline__ int32_t hash_slot(int32_t key, int size) {
  // Knuth multiplicative hash scaled onto [0, size) by a multiply-high, so
  // the table can take any size without a division.
  return static_cast<int32_t>(
      __umulhi(static_cast<uint32_t>(key) * 2654435761u, size));
}

__device__ __forceinline__ int next_slot(int h, int size) {
  return h + 1 == size ? 0 : h + 1;
}

// Copy one row of `row_bytes` bytes with a single warp. MLA rows take the
// 16-byte vectorized path; packed indexer rows use the scalar tail-safe path.
__device__ __forceinline__ void copy_row_warp(int lane_id, const char* src,
                                              char* dst, int64_t row_bytes) {
  const auto alignment = reinterpret_cast<uintptr_t>(src) |
                         reinterpret_cast<uintptr_t>(dst) |
                         static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      dst4[j] = src4[j];
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    const unsigned int* src_words = reinterpret_cast<const unsigned int*>(src);
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      dst_words[j] = src_words[j];
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    dst[j] = src[j];
  }
}

// Zero one row with a software group (ordinary HIP stores), same
// addressing contract as copy_row_warp.
__device__ __forceinline__ void zero_row_warp(int lane_id, char* dst,
                                              int64_t row_bytes) {
  const auto alignment =
      reinterpret_cast<uintptr_t>(dst) | static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    uint64_t* dst8 = reinterpret_cast<uint64_t*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      uint64_t* d = dst8 + j * 2;
      d[0] = 0;
      d[1] = 0;
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      dst_words[j] = 0u;
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    dst[j] = 0;
  }
}

__device__ __forceinline__ char* cache_row_ptr(char* cache, int64_t row,
                                               int32_t block_size,
                                               int64_t block_stride,
                                               int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

__device__ __forceinline__ void zero_cache_row_warp(int lane_id, char* cache,
                                                    int64_t row,
                                                    int32_t block_size,
                                                    int64_t block_stride,
                                                    int64_t row_bytes) {
  zero_row_warp(lane_id,
                cache_row_ptr(cache, row, block_size, block_stride, row_bytes),
                row_bytes);
}

__device__ __forceinline__ int64_t
get_physical_hot_row(const int32_t* hot_block_table, int32_t row,
                     int64_t table_stride, int32_t block_size, int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int64_t>(block) * block_size + slot % block_size;
}

__device__ __forceinline__ void store_hot_index(
    int32_t* hot_indices, int32_t* attention_indices, int32_t index,
    int32_t physical_row, int32_t hot_block_size,
    int64_t attention_block_stride) {
  hot_indices[index] = physical_row;
  if (attention_indices != nullptr) {
    attention_indices[index] =
        physical_row < 0
            ? -1
            : (physical_row / hot_block_size) * attention_block_stride +
                  physical_row % hot_block_size;
  }
}

// A union entry's value holds its first (row, top-k) position until phase 2
// or 3 resolves it to a hot slot tagged with how it was resolved.
constexpr int32_t kEntryHit = 1 << 30;
constexpr int32_t kEntryMiss = 1 << 29;
constexpr int32_t kEntryInvalid = 1 << 28;
constexpr int32_t kEntrySlotMask = 0xFFFF;
constexpr int kResidencyCounters = 5;
// Per-request table entries per referenced top-k position (load <= 1/4
// until the hash_size cap).
constexpr int kHashLoadInverse = 4;

// Translate a request-relative top-k position to its resident GPU row (-1 when
// the page is not resident) and returns its host row (-1 when not host-backed).
__device__ __forceinline__ int32_t translate_topk_entry(
    const int32_t token_index, const int32_t request_row,
    const int32_t state_row, const int32_t* __restrict__ source_block_table,
    const int32_t* __restrict__ resident_block_table, const int64_t host_rows,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block, int32_t& resident_row) {
  int32_t g = token_index;
  resident_row = -1;
  if (source_block_table != nullptr) {
    const int32_t source_block =
        token_index >= 0 ? token_index / source_block_size : -1;
    if (request_row >= 0 && request_row < source_num_reqs &&
        source_block >= 0 && source_block < source_num_blocks) {
      const int32_t physical_block =
          source_block_table[static_cast<int64_t>(request_row) *
                                 source_bt_stride +
                             source_block];
      const int64_t translated = physical_block > 0 ?
          static_cast<int64_t>(physical_block) * source_block_size +
              token_index % source_block_size : -1;
      g = translated >= 0 && translated < host_rows && translated <= INT32_MAX ?
          static_cast<int32_t>(translated) : -1;
    } else {
      g = -1;
    }
    if (resident_block_table != nullptr) {
      const int32_t resident_block =
          token_index >= 0 ? token_index / resident_block_size : -1;
      if (state_row >= 0 && state_row < resident_num_reqs &&
          resident_block >= 0 && resident_block < resident_num_blocks) {
        const int32_t physical_block =
            resident_block_table[static_cast<int64_t>(state_row) *
                                     resident_bt_stride +
                                 resident_block];
        if (physical_block != resident_null_block && physical_block >= 0) {
          const int64_t translated = static_cast<int64_t>(physical_block) *
              resident_block_size + token_index % resident_block_size;
          if (translated > INT32_MAX) __builtin_trap();
          resident_row = static_cast<int32_t>(translated);
        }
      }
    }
  }
  return g >= host_rows ? -1 : g;
}

// Write the hot index of one reference to a resolved union entry and tally it.
__device__ __forceinline__ void write_entry_index(
    const int32_t entry, const int row, const int i, const int32_t top_k,
    int32_t* __restrict__ hot_indices, int32_t* __restrict__ attention_indices,
    const int64_t attention_row_stride, int32_t* __restrict__ miss_mask,
    const int32_t* __restrict__ hot_block_table, const int request_row,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int64_t attention_block_stride, const bool first_reference, int& hits,
    int& misses) {
  int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
  int32_t* row_attention =
      attention_indices != nullptr
          ? attention_indices + static_cast<int64_t>(row) * attention_row_stride
          : nullptr;
  if (entry & kEntryInvalid) {
    store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                    attention_block_stride);
    return;
  }
  store_hot_index(row_out, row_attention, i,
                  static_cast<int32_t>(get_physical_hot_row(
                      hot_block_table, request_row, hot_table_stride,
                      hot_block_size, entry & kEntrySlotMask)),
                  hot_block_size, attention_block_stride);
  // Only an entry's first reference loads it; later references reuse the
  // slot like a hit.
  if ((entry & kEntryMiss) && first_reference) {
    if (miss_mask != nullptr) {
      miss_mask[static_cast<int64_t>(row) * top_k + i] = 1;
    }
    ++misses;
  } else {
    ++hits;
  }
}

// #69057 semantic fix, independent of wave width. A serial stable in-place
// filter is safe because its write cursor never passes its read cursor. Keep
// hot_indices/miss_mask unmodified: they still describe the original copy plan.
__device__ void pack_attention_rows(int first_row, int end_row, int top_k,
    int32_t* attention_indices, int64_t attention_row_stride,
    int32_t* valid_counts, int64_t valid_count_stride) {
  if (valid_counts == nullptr || attention_indices == nullptr) return;
  for (int row = first_row; row < end_row; ++row) {
    int32_t* values = attention_indices + static_cast<int64_t>(row) * attention_row_stride;
    int count = 0;
    for (int i = 0; i < top_k; ++i) {
      const int32_t value = values[i];
      if (value >= 0) values[count++] = value;
    }
    valid_counts[static_cast<int64_t>(row) * valid_count_stride] = count;
    for (int i = count; i < top_k; ++i) values[i] = -1;
  }
}

// One block per request, launched per batch row. A request's rows (several
// with speculative decoding) are contiguous and share its hot-buffer state, so
// the block of its first row resolves the union of their host misses once and
// maps every row onto it; the other blocks exit. The union is bounded by the
// caller's max_union_rows (every decode query's top-k), which
// device_buffer_size covers; a larger union traps. Misses take eviction slots
// in (row, top-k) order of first reference, so the plan is deterministic.
//
// Shared memory layout (int32 region followed by int16 region):
//   s_hash_keys[hash_size]   open addressing: union of the rows' host ids
//   s_hash_vals[hash_size]   first position, then slot | kEntry* tag
//   s_counters[5]            union/valid counts (other entries reserved)
//   s_lru_out[hot_size]      int16, compacted slots: [hits fwd | evict bwd]
// Duplicate references reuse one union entry and emit exactly one copy.
__global__ __launch_bounds__(256) void hisparse_resolve_residency_kernel(
    const int32_t* __restrict__ hot_block_table,  // [max_rows, hot_blocks]
    const int32_t* __restrict__ global_indices,   // global or request-relative
    const int32_t* __restrict__ request_ids,      // [num_rows] or nullptr
    const int32_t* __restrict__ source_block_table,  // [num_reqs, max_blocks]
    const int32_t* __restrict__ resident_block_table,
    int32_t* __restrict__ resolved_global_indices,    // [num_rows, top_k]
    int32_t* __restrict__ valid_counts,               // [num_rows] or nullptr
    int32_t* __restrict__ swap_host_physical_rows,    // [num_rows, top_k]
    int32_t* __restrict__ swap_device_physical_rows,  // [num_rows, top_k]
    int32_t* __restrict__ swap_counts,                // [num_rows]
    int32_t* __restrict__ hot_indices,                // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,          // [num_rows, top_k]
    int32_t* __restrict__ miss_mask,  // [num_rows, top_k] or nullptr
    int32_t* __restrict__ device_global_indices,  // [max_rows, region_stride]
    int16_t* __restrict__ lru_slots,              // [max_rows, hot_size]
    unsigned long long* __restrict__ stats,       // [2] hits,misses or nullptr
    const int32_t* __restrict__ request_state_indices,  // [num_requests] or
                                                        // nullptr
    const int32_t request_state_count, const int64_t state_rows,
    const int64_t hot_rows, const int32_t hot_table_rows, const int64_t host_rows,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int32_t top_k, const int32_t hot_size, const int32_t hash_size,
    const int64_t region_stride, const int64_t attention_block_stride,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block, const int64_t input_row_stride,
    const int64_t attention_row_stride, const int64_t valid_count_stride) {


  // The first row of each contiguous run of a request's rows leads it. FULL
  // CUDA-graph padding rows map to request 0 after the real rows, forming a
  // second run, but carry no valid top-k: that run returns before phase 2
  // touches the request's shared state.
  const int first_row = blockIdx.x;
  const int request_row =
      request_ids != nullptr ? request_ids[first_row] : first_row;
  if (request_ids != nullptr && first_row > 0 &&
      request_ids[first_row - 1] == request_row) {
    return;
  }
  int end_row = first_row + 1;
  if (request_ids != nullptr) {
    while (end_row < static_cast<int>(gridDim.x) &&
           request_ids[end_row] == request_row) {
      ++end_row;
    }
  }
  const int state_row =
      request_state_indices != nullptr && request_row >= 0 &&
              request_row < request_state_count
          ? request_state_indices[request_row]
          : (request_state_indices == nullptr ? request_row : -1);
  // V2 publishes -1 for CUDA-graph padding rows.
  if (state_row < 0) {
    for (int row = first_row; row < end_row; ++row) {
      for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        const int64_t index = static_cast<int64_t>(row) * top_k + i;
        hot_indices[index] = -1;
        if (attention_indices != nullptr) {
          attention_indices[static_cast<int64_t>(row) * attention_row_stride +
                            i] = -1;
        }
        if (resolved_global_indices != nullptr) {
          resolved_global_indices[index] = -1;
        }
        if (miss_mask != nullptr) miss_mask[index] = 0;
      }
      if (valid_counts != nullptr && threadIdx.x == 0) {
        valid_counts[static_cast<int64_t>(row) * valid_count_stride] = 0;
      }
      if (swap_counts != nullptr && threadIdx.x == 0) {
        swap_counts[row] = 0;
      }
    }
    return;
  }
  if (state_row >= state_rows || request_row < 0 || request_row >= hot_table_rows) {
    __builtin_trap();
  }
  const int tid = threadIdx.x;


  int32_t* row_dgi =
      device_global_indices + static_cast<int64_t>(state_row) * region_stride;
  int16_t* row_lru = lru_slots + static_cast<int64_t>(state_row) * hot_size;

  // A union within max_union_rows (hash_size - 1) and within the rows' top-k
  // leaves an empty slot, so every probe sequence ends. Use a slice sized to
  // this request's rows so single-row requests initialize a small table.
  const int64_t positions = static_cast<int64_t>(end_row - first_row) * top_k;
  if (positions >= kEntryInvalid) __builtin_trap();
  const int table_size = static_cast<int>(
      min(static_cast<int64_t>(hash_size), kHashLoadInverse * positions));

  extern __shared__ char smem_raw[];
  int32_t* s_hash_keys = reinterpret_cast<int32_t*>(smem_raw);
  int32_t* s_hash_vals = s_hash_keys + hash_size;
  int32_t* s_counters = s_hash_vals + hash_size;
  int16_t* s_lru_out = reinterpret_cast<int16_t*>(s_counters + kResidencyCounters);

  for (int i = tid; i < table_size; i += blockDim.x) {
    s_hash_keys[i] = kHashEmpty;
    s_hash_vals[i] = INT32_MAX;
  }
  if (tid < kResidencyCounters) {
    s_counters[tid] = 0;
  }
  __syncthreads();

  // Phase 1: translate every row's request-relative positions, resolve
  // resident rows directly, and collect the union of host rows.
  for (int row = first_row; row < end_row; ++row) {
    const int32_t* row_topk =
        global_indices + static_cast<int64_t>(row) * input_row_stride;
    int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
    int32_t* row_attention =
        attention_indices != nullptr
            ? attention_indices +
                  static_cast<int64_t>(row) * attention_row_stride
            : nullptr;
    int32_t* row_miss = miss_mask != nullptr
                            ? miss_mask + static_cast<int64_t>(row) * top_k
                            : nullptr;
    for (int i = tid; i < top_k; i += blockDim.x) {
      int32_t resident_row;
      const int32_t g = translate_topk_entry(
          row_topk[i], request_row, state_row, source_block_table,
          resident_block_table, host_rows, source_bt_stride, source_num_reqs,
          source_num_blocks, source_block_size, resident_bt_stride,
          resident_num_reqs, resident_num_blocks, resident_block_size,
          resident_null_block, resident_row);
      if (resolved_global_indices != nullptr) {
        resolved_global_indices[static_cast<int64_t>(row) * top_k + i] = g;
      }
      if (resident_row >= 0 || g >= 0) atomicAdd(&s_counters[2], 1);
      if (row_miss != nullptr) row_miss[i] = 0;
      if (resident_row >= hot_rows) __builtin_trap();
      if (resident_row >= 0) {
        store_hot_index(row_out, row_attention, i, resident_row, hot_block_size,
                        attention_block_stride);
      } else if (g < 0) {
        store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                        attention_block_stride);
      } else {
        int h = hash_slot(g, table_size);
        bool inserted = false;
        for (int probe = 0; probe < table_size; ++probe) {
          const int32_t old = atomicCAS(&s_hash_keys[h], kHashEmpty, g);
          if (old == kHashEmpty) {
            atomicAdd(&s_counters[1], 1);
            inserted = true;
            break;
          }
          if (old == g) {
            inserted = true;
            break;
          }
          h = next_slot(h, table_size);
        }
        if (inserted) {
          atomicMin(&s_hash_vals[h], (row - first_row) * top_k + i);
        } else {
          // A full table: push the union count past the cap so the check
          // below traps.
          atomicAdd(&s_counters[1], hash_size);
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      if (valid_counts != nullptr) {
        valid_counts[static_cast<int64_t>(row) * valid_count_stride] =
            s_counters[2];
      }
      s_counters[2] = 0;
    }
    __syncthreads();
  }
  // The table and eviction slots are sized for max_union_rows (hash_size - 1);
  // a larger union would never reach an empty probe slot or run past s_lru_out.
  if (s_counters[1] >= hash_size) {
    if (tid == 0) {
      printf(
          "HiSparse residency: request %d resolves %d host rows, above the "
          "configured bound of %d\n",
          request_row, s_counters[1], hash_size - 1);
    }
    __builtin_trap();
  }
  // Fully resident rows need only request-relative page translation. Avoid
  // scanning or rewriting the hot LRU when no selected row can consult it.
  if (s_counters[1] == 0) {
    for (int row = first_row + tid; row < end_row; row += blockDim.x) {
      if (swap_counts != nullptr) swap_counts[row] = 0;
    }
    if (tid == 0) pack_attention_rows(first_row, end_row, top_k,
        attention_indices, attention_row_stride, valid_counts, valid_count_stride);
    return;
  }

  // Correctness-first metadata path. One thread orders the union and LRU;
  // the hash construction above remains parallel. This trades metadata
  // throughput for identical ordering on wave32/wave64, without masked scans.
  // Core033 target-only ABI with #69057 stable valid-index packing semantics.
  if (tid != 0) return;
  int total_hits = 0;
  int total_evictable = 0;
  for (int pos = 0; pos < hot_size; ++pos) {
    const int16_t slot = row_lru[pos];
    const int32_t cached = slot >= 0 && slot < region_stride ? row_dgi[slot] : -1;
    bool hit = false;
    if (cached >= 0) {
      int h = hash_slot(cached, table_size);
      while (s_hash_keys[h] != kHashEmpty) {
        if (s_hash_keys[h] == cached) {
          const int64_t physical = get_physical_hot_row(hot_block_table,
              request_row, hot_table_stride, hot_block_size, slot);
          if (physical < 0 || physical >= hot_rows) __builtin_trap();
          s_hash_vals[h] = kEntryHit | slot;
          hit = true;
          break;
        }
        h = next_slot(h, table_size);
      }
    }
    if (hit) s_lru_out[total_hits++] = slot;
    else s_lru_out[hot_size - 1 - total_evictable++] = slot;
  }
  int total_misses = 0;
  int entry_hits = 0;
  int entry_misses = 0;
  for (int row = first_row; row < end_row; ++row) {
    for (int i = 0; i < top_k; ++i) {
      int32_t resident;
      const int32_t g = translate_topk_entry(
          global_indices[static_cast<int64_t>(row) * input_row_stride + i],
          request_row, state_row, source_block_table, resident_block_table,
          host_rows, source_bt_stride, source_num_reqs, source_num_blocks,
          source_block_size, resident_bt_stride, resident_num_reqs,
          resident_num_blocks, resident_block_size, resident_null_block, resident);
      if (resident >= 0 || g < 0) continue;
      int h = hash_slot(g, table_size);
      while (s_hash_keys[h] != g) h = next_slot(h, table_size);
      int32_t entry = s_hash_vals[h];
      const bool first_miss = !(entry & (kEntryHit | kEntryMiss | kEntryInvalid));
      if (first_miss) {
        if (total_misses >= total_evictable) __builtin_trap();
        const int16_t slot = s_lru_out[hot_size - 1 - total_misses];
        int32_t physical = -1;
        if (slot >= 0 && slot < region_stride) {
          const int64_t p = get_physical_hot_row(hot_block_table, request_row,
              hot_table_stride, hot_block_size, slot);
          if (p >= 0 && p < hot_rows) physical = static_cast<int32_t>(p);
        }
        entry = physical < 0 ? kEntryInvalid : (kEntryMiss | slot);
        s_hash_vals[h] = entry;
        const int64_t compact = static_cast<int64_t>(first_row) * top_k + total_misses;
        if (swap_host_physical_rows != nullptr) {
          swap_host_physical_rows[compact] = g;
          swap_device_physical_rows[compact] = physical;
        }
        if (physical >= 0) row_dgi[slot] = g;
        ++total_misses;
      }
      write_entry_index(entry, row, i, top_k, hot_indices, attention_indices,
          attention_row_stride, miss_mask, hot_block_table, request_row,
          hot_table_stride, hot_block_size, attention_block_stride,
          first_miss, entry_hits, entry_misses);
    }
  }
  if (swap_counts != nullptr) {
    for (int row = first_row; row < end_row; ++row) {
      swap_counts[row] = max(0, min(top_k, total_misses - (row - first_row) * top_k));
    }
  }
  if (stats != nullptr) {
    atomicAdd(&stats[0], static_cast<unsigned long long>(entry_hits));
    atomicAdd(&stats[1], static_cast<unsigned long long>(entry_misses));
  }
  const int remaining = total_evictable - total_misses;
  for (int i = 0; i < hot_size; ++i) {
    if (i < remaining) row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    else if (i < remaining + total_misses)
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining)];
    else row_lru[i] = s_lru_out[i - remaining - total_misses];
  }
  pack_attention_rows(first_row, end_row, top_k, attention_indices,
      attention_row_stride, valid_counts, valid_count_stride);
}

// Shared-layer plan replay. Given a plan (hot_indices + miss_mask) already
// computed by the group's index-producing layer, gather THIS
// layer's own missed KV rows into the planned hot slots. No LRU resolution:
// index-sharing shared layers see identical global_indices/hot_indices, so the
// slot assignment is identical -- only the per-layer bytes differ. Fixed shape
// (num_rows x top_k), so it is CUDA-graph-capture safe.
__global__ void hisparse_gather_plan_kernel(
    const char* __restrict__ host_cache,         // [host_rows, row_bytes]
    char* __restrict__ hot_cache,                // [hot_rows, row_bytes]
    const int32_t* __restrict__ global_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ hot_indices,  // [num_rows, top_k] abs hot rows
    const int32_t* __restrict__ miss_mask,    // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ request_state_indices,  // [num_rows] or nullptr
    const int64_t host_rows, const int64_t hot_rows, const int64_t row_bytes,
    const int64_t hot_block_stride, const int32_t hot_block_size,
    const int32_t top_k, const int64_t attention_block_stride) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  // Columns are interleaved across gridDim.y blocks per row so few-row
  // launches (local-prefill staging's single-row layout, small decode
  // batches) still fill the device with outstanding host reads instead of
  // starving on one block per row.
  const int row = blockIdx.x;
  const bool is_padding =
      request_state_indices != nullptr && request_state_indices[row] < 0;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < top_k; col += col_stride) {
    const int32_t dst = hot_indices[base + col];
    if (lane_id == 0 && attention_indices != nullptr) {
      attention_indices[base + col] =
          is_padding
              ? -1
              : (dst < 0 ? -1
                         : (dst / hot_block_size) * attention_block_stride +
                               dst % hot_block_size);
    }
    if (is_padding) {
      continue;
    }
    if (miss_mask[base + col] == 0) {
      continue;
    }
    const int32_t g = global_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_row_warp(lane_id, host_cache + static_cast<int64_t>(g) * row_bytes,
                    cache_row_ptr(hot_cache, dst, hot_block_size,
                                  hot_block_stride, row_bytes),
                    row_bytes);
    } else {
      // No source row for g: zero the planned slot rather than serving
      // whatever bytes it held (see the swap-in kernel's phase 5).
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes);
    }
  }
}

// Copy the compact swap rows produced by the residency resolver.
__global__ void hisparse_gather_compact_kernel(
    const char* __restrict__ host_cache, char* __restrict__ hot_cache,
    const int32_t* __restrict__ miss_global_indices,
    const int32_t* __restrict__ miss_hot_indices,
    const int32_t* __restrict__ miss_counts, const int64_t host_rows,
    const int64_t hot_rows, const int64_t row_bytes,
    const int64_t hot_block_stride, const int32_t hot_block_size,
    const int32_t top_k) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int row = blockIdx.x;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int miss_count = min(max(miss_counts[row], 0), top_k);
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < miss_count; col += col_stride) {
    const int32_t g = miss_global_indices[base + col];
    const int32_t dst = miss_hot_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_row_warp(lane_id, host_cache + static_cast<int64_t>(g) * row_bytes,
                    cache_row_ptr(hot_cache, dst, hot_block_size,
                                  hot_block_stride, row_bytes),
                    row_bytes);
    } else {
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes);
    }
  }
}

__global__ void hisparse_invalidate_written_slots_kernel(
    int32_t* __restrict__ device_global_indices,
    const int32_t* __restrict__ request_state_indices,
    const int32_t* __restrict__ req_id_per_token,
    const int64_t* __restrict__ written_slots, const int64_t num_tokens,
    const int64_t num_request_ids, const int64_t num_state_rows,
    const int64_t region_stride) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int32_t req_idx = req_id_per_token[token_idx];
  if (req_idx < 0 || req_idx >= num_request_ids) {
    return;
  }
  const int32_t state_idx = request_state_indices[req_idx];
  const int64_t written_slot = written_slots[token_idx];
  if (state_idx < 0 || state_idx >= num_state_rows || written_slot < 0) {
    return;
  }
  int32_t* row = device_global_indices + state_idx * region_stride;
  for (int64_t offset = threadIdx.x; offset < region_stride;
       offset += blockDim.x) {
    if (row[offset] == written_slot) {
      row[offset] = -1;
    }
  }
}

int64_t check_2d_rows(const at::Tensor& t, const char* name,
                      int64_t row_bytes) {
  TORCH_CHECK(t.dim() == 2, name, " must be 2D");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.size(1) * t.element_size() == row_bytes, name,
                  " row width mismatch");
  return t.size(0);
}

}  // namespace

void hisparse_invalidate_written_slots(
    at::Tensor const& device_global_indices,
    at::Tensor const& request_state_indices,
    at::Tensor const& req_id_per_token,
    at::Tensor const& written_slots) {
  TORCH_CHECK(device_global_indices.is_cuda() &&
                      request_state_indices.is_cuda() &&
                      req_id_per_token.is_cuda() && written_slots.is_cuda(),
                  "HiSparse invalidation tensors must be on CUDA");
  TORCH_CHECK(
      device_global_indices.scalar_type() ==
              at::ScalarType::Int &&
          device_global_indices.dim() == 2 &&
          device_global_indices.is_contiguous(),
      "device_global_indices must be a contiguous 2D int32 CUDA tensor");
  TORCH_CHECK(
      request_state_indices.scalar_type() ==
              at::ScalarType::Int &&
          request_state_indices.dim() == 1 &&
          request_state_indices.is_contiguous(),
      "request_state_indices must be a contiguous 1D int32 CUDA tensor");
  TORCH_CHECK(
      req_id_per_token.scalar_type() == at::ScalarType::Int &&
          req_id_per_token.dim() == 1 && req_id_per_token.is_contiguous(),
      "req_id_per_token must be a contiguous 1D int32 CUDA tensor");
  TORCH_CHECK(
      written_slots.scalar_type() == at::ScalarType::Long &&
          written_slots.dim() == 1 && written_slots.is_contiguous(),
      "written_slots must be a contiguous 1D int64 CUDA tensor");
  TORCH_CHECK(req_id_per_token.numel() == written_slots.numel(),
                  "req_id_per_token and written_slots must have equal length");
  const int device_index = device_global_indices.get_device();
  TORCH_CHECK(
      request_state_indices.get_device() == device_index &&
          req_id_per_token.get_device() == device_index &&
          written_slots.get_device() == device_index,
      "HiSparse invalidation tensors must be on the same CUDA device");

  const int64_t num_tokens = written_slots.numel();
  const int64_t num_state_rows = device_global_indices.size(0);
  const int64_t region_stride = device_global_indices.size(1);
  if (num_tokens == 0 || num_state_rows == 0 || region_stride == 0) {
    return;
  }
  TORCH_CHECK(num_tokens <= INT32_MAX,
                  "HiSparse invalidation grid exceeds CUDA limits");

  constexpr int kBlockSize = 256;
  const c10::DeviceGuard device_guard(device_global_indices.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hisparse_invalidate_written_slots_kernel<<<static_cast<int>(num_tokens),
                                             kBlockSize, 0, stream>>>(
      device_global_indices.mutable_data_ptr<int32_t>(),
      request_state_indices.const_data_ptr<int32_t>(),
      req_id_per_token.const_data_ptr<int32_t>(),
      written_slots.const_data_ptr<int64_t>(), num_tokens,
      request_state_indices.numel(), num_state_rows, region_stride);
  const hipError_t launch_error = hipGetLastError();
  TORCH_CHECK(launch_error == hipSuccess,
                  "HiSparse invalidation kernel launch failed: ",
                  hipGetErrorString(launch_error));
}

void hisparse_resolve_residency(
    at::Tensor const& host_cache, at::Tensor const& hot_cache,
    at::Tensor const& hot_block_table,
    at::Tensor const& global_indices,
    at::Tensor const& hot_indices,
    at::Tensor const& device_global_indices,
    at::Tensor const& lru_slots,
    std::optional<at::Tensor> const& request_state_indices,
    int64_t region_stride, int64_t max_union_rows,
    std::optional<at::Tensor> const& miss_mask,
    std::optional<at::Tensor> const& stats,
    std::optional<at::Tensor> const& attention_indices,
    int64_t attention_block_stride,
    std::optional<at::Tensor> const& request_ids,
    std::optional<at::Tensor> const& source_block_table,
    int64_t source_block_size,
    std::optional<at::Tensor> const& resolved_global_indices,
    std::optional<at::Tensor> const& valid_counts,
    std::optional<at::Tensor> const& swap_host_physical_rows,
    std::optional<at::Tensor> const& swap_device_physical_rows,
    std::optional<at::Tensor> const& swap_counts,
    std::optional<at::Tensor> const& resident_block_table,
    int64_t resident_block_size, int64_t resident_null_block) {
  TORCH_CHECK(global_indices.dim() == 2 && hot_indices.dim() == 2,
              "HiSparse indices must be 2D");
  check_grid(global_indices.size(0), global_indices.size(1));
  check_cache_layout(hot_cache);
  same_device(hot_block_table, hot_cache);
  same_device(global_indices, hot_cache);
  same_device(hot_indices, hot_cache);
  same_device(device_global_indices, hot_cache);
  same_device(lru_slots, hot_cache);
  for (const auto* opt : {&request_state_indices, &miss_mask, &stats,
       &attention_indices, &request_ids, &source_block_table,
       &resolved_global_indices, &valid_counts, &swap_host_physical_rows,
       &swap_device_physical_rows, &swap_counts, &resident_block_table}) {
    if (opt->has_value()) same_device(opt->value(), hot_cache);
  }
  TORCH_CHECK(hot_block_table.dim() == 2 && hot_block_table.stride(1) == 1 &&
              hot_block_table.size(0) <= INT32_MAX,
              "hot block table must be row-major 2D");
  TORCH_CHECK(source_block_size >= 0 && source_block_size <= INT32_MAX &&
              resident_block_size >= 0 && resident_block_size <= INT32_MAX &&
              resident_null_block >= 0 && resident_null_block <= INT32_MAX,
              "block parameters must fit nonnegative int32");
  TORCH_CHECK(
      host_cache.device().is_cpu() && host_cache.is_pinned(),
      "host_cache must be pinned CPU memory");
  TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  TORCH_CHECK(
      hot_block_table.is_cuda() &&
          hot_block_table.scalar_type() == at::ScalarType::Int &&
          hot_block_table.dim() == 2,
      "hot_block_table must be a 2D int32 CUDA tensor");
  TORCH_CHECK(global_indices.is_cuda() && hot_indices.is_cuda() &&
                      device_global_indices.is_cuda() && lru_slots.is_cuda(),
                  "index tensors must be on CUDA");
  TORCH_CHECK(
      global_indices.scalar_type() == at::ScalarType::Int &&
          hot_indices.scalar_type() == at::ScalarType::Int,
      "global_indices/hot_indices must be int32");
  TORCH_CHECK(
      device_global_indices.scalar_type() == at::ScalarType::Int,
      "device_global_indices must be int32");
  TORCH_CHECK(
      lru_slots.scalar_type() == at::ScalarType::Short,
      "lru_slots must be int16");
  TORCH_CHECK(global_indices.dim() == 2 && global_indices.stride(1) == 1,
                  "global_indices must be row-major 2D");
  const int64_t num_rows = global_indices.size(0);
  TORCH_CHECK(num_rows >= 0 && num_rows <= INT32_MAX,
                  "num_rows must fit the CUDA grid");
  TORCH_CHECK(hot_indices.size(0) == num_rows &&
                      hot_indices.size(1) == global_indices.size(1) &&
                      hot_indices.is_contiguous(),
                  "hot_indices must have one contiguous row per launch row");
  TORCH_CHECK(
      device_global_indices.dim() == 2 && device_global_indices.is_contiguous(),
      "device_global_indices must be contiguous 2D");
  TORCH_CHECK(
      lru_slots.dim() == 2 &&
          lru_slots.size(0) == device_global_indices.size(0) &&
          lru_slots.is_contiguous(),
      "lru_slots must be contiguous with one row per request state");
  TORCH_CHECK(hot_cache.dim() == 3,
                  "hot_cache must be [num_blocks, block_size, row_width]");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  TORCH_CHECK(row_bytes % 16 == 0, "KV rows must be 16-byte aligned");
  TORCH_CHECK(hot_cache.stride(1) * hot_cache.element_size() == row_bytes,
                  "hot-cache rows must be contiguous");
  const int64_t hot_block_size = hot_cache.size(1);
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  TORCH_CHECK(host_rows <= INT32_MAX, "host row indices must fit int32");
  const auto launch_rows = static_cast<int32_t>(num_rows);
  const auto top_k = static_cast<int32_t>(global_indices.size(1));
  TORCH_CHECK(lru_slots.size(1) <= 32768, "hot size must fit int16 before narrowing");
  const auto hot_size = static_cast<int32_t>(lru_slots.size(1));
  TORCH_CHECK(hot_size >= top_k, "hot buffer size must be >= top_k");
  TORCH_CHECK(hot_size <= 32768, "hot buffer size must fit int16 slots");
  TORCH_CHECK(region_stride == hot_size,
                  "region_stride must match the LRU size");
  TORCH_CHECK(device_global_indices.size(1) == region_stride,
                  "device_global_indices must cover the full hot region");
  TORCH_CHECK(hot_block_table.size(1) >=
                      (region_stride + hot_block_size - 1) / hot_block_size,
                  "hot_block_table has too few columns");
  TORCH_CHECK(hot_rows < INT32_MAX, "hot indices must fit int32");

  const int32_t* request_ids_ptr = nullptr;
  const int32_t* source_block_table_ptr = nullptr;
  int64_t source_bt_stride = 0;
  int32_t source_num_reqs = 0;
  int32_t source_num_blocks = 0;
  TORCH_CHECK(
      request_ids.has_value() == source_block_table.has_value(),
      "request_ids and source_block_table must be provided together");
  if (source_block_table.has_value()) {
    auto const& req = request_ids.value();
    auto const& table = source_block_table.value();
    TORCH_CHECK(
        req.is_cuda() &&
            req.scalar_type() == at::ScalarType::Int &&
            req.numel() >= launch_rows && req.is_contiguous(),
        "request_ids must be contiguous int32 on CUDA with one entry per row");
    TORCH_CHECK(
        table.is_cuda() &&
            table.scalar_type() == at::ScalarType::Int &&
            table.dim() == 2 && table.stride(1) == 1,
        "source_block_table must be a row-major 2D int32 CUDA tensor");
    TORCH_CHECK(source_block_size > 0,
                    "source_block_size must be positive");
    request_ids_ptr = req.const_data_ptr<int32_t>();
    source_block_table_ptr = table.const_data_ptr<int32_t>();
    check_grid(table.size(0), table.size(1));
    source_bt_stride = table.stride(0);
    source_num_reqs = static_cast<int32_t>(table.size(0));
    source_num_blocks = static_cast<int32_t>(table.size(1));
  }
  const int32_t required_hot_rows =
      request_ids_ptr != nullptr ? source_num_reqs : launch_rows;
  TORCH_CHECK(hot_block_table.size(0) >= required_hot_rows,
                  "hot_block_table has too few rows");

  const int32_t* resident_block_table_ptr = nullptr;
  int64_t resident_bt_stride = 0;
  int32_t resident_num_reqs = 0;
  int32_t resident_num_blocks = 0;
  if (resident_block_table.has_value()) {
    auto const& table = resident_block_table.value();
    TORCH_CHECK(
        source_block_table.has_value() && request_ids.has_value(),
        "resident lookup requires request_ids and source_block_table");
    TORCH_CHECK(
        table.is_cuda() &&
            table.scalar_type() == at::ScalarType::Int &&
            table.dim() == 2 && table.stride(1) == 1,
        "resident_block_table must be a row-major 2D int32 CUDA tensor");
    TORCH_CHECK(resident_block_size == hot_block_size,
                    "resident and hot block sizes must match");
    TORCH_CHECK(resident_null_block >= 0,
                    "resident null block must be non-negative");
    resident_block_table_ptr = table.const_data_ptr<int32_t>();
    check_grid(table.size(0), table.size(1));
    resident_bt_stride = table.stride(0);
    resident_num_reqs = static_cast<int32_t>(table.size(0));
    resident_num_blocks = static_cast<int32_t>(table.size(1));
  }

  int32_t* resolved_global_indices_ptr = nullptr;
  if (resolved_global_indices.has_value()) {
    auto const& resolved = resolved_global_indices.value();
    TORCH_CHECK(
        resolved.is_cuda() &&
            resolved.scalar_type() == at::ScalarType::Int &&
            resolved.dim() == 2 && resolved.size(0) == launch_rows &&
            resolved.size(1) == global_indices.size(1) &&
            resolved.is_contiguous(),
        "resolved_global_indices must be contiguous int32 matching indices");
    resolved_global_indices_ptr = resolved.mutable_data_ptr<int32_t>();
  }

  int32_t* valid_counts_ptr = nullptr;
  if (valid_counts.has_value()) {
    auto const& counts = valid_counts.value();
    TORCH_CHECK(
        counts.is_cuda() &&
            counts.scalar_type() == at::ScalarType::Int &&
            counts.dim() == 1 && counts.size(0) == launch_rows &&
            (launch_rows <= 1 || counts.stride(0) > 0),
        "valid_counts must be int32 on CUDA with one entry per row");
    valid_counts_ptr = counts.mutable_data_ptr<int32_t>();
  }

  int32_t* swap_host_physical_rows_ptr = nullptr;
  int32_t* swap_device_physical_rows_ptr = nullptr;
  int32_t* swap_counts_ptr = nullptr;
  const bool has_compact_swaps = swap_host_physical_rows.has_value();
  TORCH_CHECK(has_compact_swaps == swap_device_physical_rows.has_value() &&
                      has_compact_swaps == swap_counts.has_value(),
                  "compact swap tensors must be provided together");
  if (has_compact_swaps) {
    auto const& globals = swap_host_physical_rows.value();
    auto const& hots = swap_device_physical_rows.value();
    auto const& counts = swap_counts.value();
    TORCH_CHECK(
        globals.is_cuda() && hots.is_cuda() && counts.is_cuda() &&
            globals.scalar_type() == at::ScalarType::Int &&
            hots.scalar_type() == at::ScalarType::Int &&
            counts.scalar_type() == at::ScalarType::Int &&
            globals.dim() == 2 && hots.dim() == 2 &&
            globals.size(0) == launch_rows &&
            globals.size(1) == global_indices.size(1) &&
            hots.size(0) == launch_rows &&
            hots.size(1) == global_indices.size(1) &&
            counts.numel() >= launch_rows && globals.is_contiguous() &&
            hots.is_contiguous() && counts.is_contiguous(),
        "compact swaps must be contiguous int32 matching indices");
    swap_host_physical_rows_ptr = globals.mutable_data_ptr<int32_t>();
    swap_device_physical_rows_ptr = hots.mutable_data_ptr<int32_t>();
    swap_counts_ptr = counts.mutable_data_ptr<int32_t>();
  }

  const int32_t* request_state_ptr = nullptr;
  int32_t request_state_count = 0;
  if (request_state_indices.has_value()) {
    auto const& state_indices = request_state_indices.value();
    TORCH_CHECK(
        state_indices.is_cuda() &&
            state_indices.scalar_type() == at::ScalarType::Int &&
            state_indices.dim() == 1 && state_indices.is_contiguous() &&
            state_indices.numel() <= INT32_MAX,
        "request_state_indices must be contiguous 1D int32 on CUDA");
    request_state_ptr = state_indices.const_data_ptr<int32_t>();
    request_state_count = static_cast<int32_t>(state_indices.numel());
  } else {
    TORCH_CHECK(device_global_indices.size(0) >= launch_rows,
                    "device_global_indices has too few rows");
  }

  // Optional output: 1 at columns requiring a host-to-device swap, 0 elsewhere.
  int32_t* miss_mask_ptr = nullptr;
  if (miss_mask.has_value()) {
    auto const& mm = miss_mask.value();
    TORCH_CHECK(
        mm.is_cuda() &&
            mm.scalar_type() == at::ScalarType::Int &&
            mm.dim() == 2 && mm.is_contiguous() && mm.size(0) == launch_rows &&
            mm.size(1) == global_indices.size(1),
        "miss_mask must be a contiguous int32 CUDA tensor matching "
        "global_indices");
    miss_mask_ptr = mm.mutable_data_ptr<int32_t>();
  }

  unsigned long long* stats_ptr = nullptr;
  if (stats.has_value()) {
    auto const& st = stats.value();
    TORCH_CHECK(
        st.is_cuda() && st.is_contiguous() && st.dim() == 1 &&
            st.scalar_type() == at::ScalarType::UInt64 &&
            st.numel() == 2,
        "stats must be a contiguous two-element uint64 CUDA tensor");
    stats_ptr = static_cast<unsigned long long*>(st.mutable_data_ptr());
  }

  int32_t* attention_indices_ptr = nullptr;
  if (attention_indices.has_value()) {
    auto const& indices = attention_indices.value();
    TORCH_CHECK(
        indices.is_cuda() &&
            indices.scalar_type() == at::ScalarType::Int &&
            indices.dim() == 2 && indices.size(0) == launch_rows &&
            indices.stride(1) == 1 && indices.size(1) == global_indices.size(1) &&
            (launch_rows <= 1 || indices.stride(0) >= indices.size(1)),
        "attention_indices must be row-major int32 matching global_indices");
    TORCH_CHECK(attention_block_stride >= hot_block_size,
                    "attention block stride must cover one hot block");
    attention_indices_ptr = indices.mutable_data_ptr<int32_t>();
  }

  if (launch_rows == 0 || top_k == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  TORCH_CHECK(max_union_rows > 0 && max_union_rows <= hot_size,
                  "max_union_rows must be in [1, hot_size], got ",
                  max_union_rows);
  // One spare entry above the union bound keeps every probe sequence finite.
  const int hash_size = static_cast<int>(max_union_rows) + 1;
  const size_t smem_bytes = sizeof(int32_t) * (2 * hash_size + kResidencyCounters) +
      sizeof(int16_t) * hot_size;

  const c10::DeviceGuard device_guard(
      hot_cache.device());
  int max_smem_bytes = 0;
  TORCH_CHECK(hipDeviceGetAttribute(
                      &max_smem_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock,
                      hot_cache.get_device()) == hipSuccess,
                  "failed to query shared memory per block");
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(max_smem_bytes),
                  "HiSparse device_buffer_size=", hot_size, " needs ",
                  smem_bytes,
                  " bytes of shared memory to resolve residency, "
                  "but the device allows ",
                  max_smem_bytes,
                  "; lower device_buffer_size or the number of speculative "
                  "tokens.");
  const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  const int64_t attention_row_stride =
      attention_indices.has_value() ? attention_indices.value().stride(0) : 0;
  const int64_t valid_count_stride =
      valid_counts.has_value() ? valid_counts.value().stride(0) : 0;
  auto kernel = hisparse_resolve_residency_kernel;
  if (smem_bytes > 48 * 1024) {
    const hipError_t attribute_error = hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(attribute_error == hipSuccess,
                    "failed to configure HiSparse swap-in shared memory: ",
                    hipGetErrorString(attribute_error));
  }
  kernel<<<launch_rows, kBlockSize, smem_bytes, stream>>>(
      hot_block_table.const_data_ptr<int32_t>(),
      global_indices.const_data_ptr<int32_t>(), request_ids_ptr,
      source_block_table_ptr, resident_block_table_ptr,
      resolved_global_indices_ptr, valid_counts_ptr,
      swap_host_physical_rows_ptr, swap_device_physical_rows_ptr,
      swap_counts_ptr, hot_indices.mutable_data_ptr<int32_t>(),
      attention_indices_ptr, miss_mask_ptr,
      device_global_indices.mutable_data_ptr<int32_t>(),
      lru_slots.mutable_data_ptr<int16_t>(), stats_ptr, request_state_ptr,
      request_state_count, device_global_indices.size(0), hot_rows,
      static_cast<int32_t>(hot_block_table.size(0)), host_rows, hot_block_table.stride(0), hot_block_size,
      top_k, hot_size, hash_size, region_stride, attention_block_stride,
      source_bt_stride, source_num_reqs, source_num_blocks,
      static_cast<int32_t>(source_block_size), resident_bt_stride,
      resident_num_reqs, resident_num_blocks,
      static_cast<int32_t>(resident_block_size),
      static_cast<int32_t>(resident_null_block), global_indices.stride(0),
      attention_row_stride, valid_count_stride);
  const hipError_t launch_error = hipGetLastError();
  TORCH_CHECK(launch_error == hipSuccess,
                  "HiSparse residency kernel launch failed: ",
                  hipGetErrorString(launch_error));
}

void hisparse_gather_plan(
    at::Tensor const& host_cache, at::Tensor const& hot_cache,
    at::Tensor const& global_indices,
    at::Tensor const& hot_indices,
    at::Tensor const& miss_mask,
    std::optional<at::Tensor> const& request_state_indices,
    std::optional<at::Tensor> const& attention_indices,
    int64_t attention_block_stride) {
  TORCH_CHECK(global_indices.dim() == 2 && hot_indices.dim() == 2 &&
              miss_mask.dim() == 2, "HiSparse plan tensors must be 2D");
  check_grid(global_indices.size(0), global_indices.size(1));
  check_cache_layout(hot_cache);
  same_device(global_indices, hot_cache);
  same_device(hot_indices, hot_cache);
  same_device(miss_mask, hot_cache);
  if (request_state_indices.has_value()) {
    same_device(request_state_indices.value(), hot_cache);
    TORCH_CHECK(request_state_indices->dim() == 1 &&
                request_state_indices->is_contiguous(), "plan state indices must be contiguous 1D");
  }
  if (attention_indices.has_value()) same_device(attention_indices.value(), hot_cache);
  TORCH_CHECK(
      host_cache.device().is_cpu() && host_cache.is_pinned(),
      "host_cache must be pinned CPU memory");
  TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  TORCH_CHECK(
      global_indices.is_cuda() && hot_indices.is_cuda() && miss_mask.is_cuda(),
      "plan tensors must be on CUDA");
  TORCH_CHECK(
      global_indices.scalar_type() == at::ScalarType::Int &&
          hot_indices.scalar_type() == at::ScalarType::Int &&
          miss_mask.scalar_type() == at::ScalarType::Int,
      "plan tensors must be int32");
  TORCH_CHECK(global_indices.dim() == 2 && global_indices.is_contiguous(),
                  "global_indices must be contiguous 2D");
  TORCH_CHECK(
      hot_indices.size(0) == global_indices.size(0) &&
          hot_indices.size(1) == global_indices.size(1) &&
          miss_mask.size(0) == global_indices.size(0) &&
          miss_mask.size(1) == global_indices.size(1) &&
          hot_indices.is_contiguous() && miss_mask.is_contiguous(),
      "hot_indices/miss_mask must match contiguous 2D global_indices");

  TORCH_CHECK(hot_cache.dim() == 2 || hot_cache.dim() == 3,
                  "hot_cache must be a 2D staging buffer or paged 3D cache");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  TORCH_CHECK(row_bytes % 16 == 0, "KV rows must be 16-byte aligned");
  TORCH_CHECK(
      hot_cache.stride(hot_cache.dim() - 2) * hot_cache.element_size() ==
          row_bytes,
      "hot-cache rows must be contiguous");
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  TORCH_CHECK(host_rows <= INT32_MAX, "host row indices must fit int32");

  const auto num_rows = static_cast<int32_t>(global_indices.size(0));
  const auto top_k = static_cast<int32_t>(global_indices.size(1));

  const int32_t* request_state_ptr = nullptr;
  if (request_state_indices.has_value()) {
    auto const& state_indices = request_state_indices.value();
    TORCH_CHECK(
        state_indices.is_cuda() &&
            state_indices.scalar_type() == at::ScalarType::Int &&
            state_indices.numel() >= num_rows,
        "request_state_indices must be int32 on CUDA with one entry per row");
    request_state_ptr = state_indices.const_data_ptr<int32_t>();
  }

  int32_t* attention_indices_ptr = nullptr;
  if (attention_indices.has_value()) {
    auto const& indices = attention_indices.value();
    TORCH_CHECK(
        indices.is_cuda() &&
            indices.scalar_type() == at::ScalarType::Int &&
            indices.dim() == 2 && indices.is_contiguous() &&
            indices.size(0) == global_indices.size(0) &&
            indices.size(1) == global_indices.size(1),
        "attention_indices must be a contiguous int32 CUDA tensor matching "
        "global_indices");
    attention_indices_ptr = indices.mutable_data_ptr<int32_t>();
  }

  if (num_rows == 0 || top_k == 0) {
    return;
  }

  // Match the swap-in kernel's block size: the gather serves 3 of every 4
  // layers' misses (index-sharing replay), so per-row copy parallelism is
  // the throughput limiter on cold rows.
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  // Interleave columns over enough blocks per row to cover the device even
  // for few-row launches (local-prefill staging's single-row layout, small
  // decode batches), keeping >= 1 column per warp.
  constexpr int kTargetBlocks = 256;
  const int max_chunks = std::max(1, (top_k + kNumWarps - 1) / kNumWarps);
  const int num_chunks =
      std::min(max_chunks, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  const int32_t hot_block_size =
      hot_cache.dim() == 3 ? static_cast<int32_t>(hot_cache.size(1)) : 1;
  if (attention_indices.has_value()) {
    TORCH_CHECK(attention_block_stride >= hot_block_size,
                    "attention block stride must cover one hot block");
  }
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  const c10::DeviceGuard device_guard(
      hot_cache.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hisparse_gather_plan_kernel<<<grid, kBlockSize, 0, stream>>>(
      mapped_host_pointer(host_cache),
      static_cast<char*>(hot_cache.mutable_data_ptr()),
      global_indices.const_data_ptr<int32_t>(),
      hot_indices.const_data_ptr<int32_t>(),
      miss_mask.const_data_ptr<int32_t>(), attention_indices_ptr,
      request_state_ptr, host_rows, hot_rows, row_bytes, hot_block_stride,
      hot_block_size, top_k, attention_block_stride);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "HiSparse gather_plan launch failed");
}

void hisparse_gather_compact(at::Tensor const& host_cache,
                             at::Tensor const& hot_cache,
                             at::Tensor const& miss_global_indices,
                             at::Tensor const& miss_hot_indices,
                             at::Tensor const& miss_counts) {
  TORCH_CHECK(miss_global_indices.dim() == 2 && miss_hot_indices.dim() == 2,
              "compact indices must be 2D");
  check_grid(miss_global_indices.size(0), miss_global_indices.size(1));
  check_cache_layout(hot_cache);
  same_device(miss_global_indices, hot_cache);
  same_device(miss_hot_indices, hot_cache);
  same_device(miss_counts, hot_cache);
  TORCH_CHECK(
      host_cache.device().is_cpu() && host_cache.is_pinned(),
      "host_cache must be pinned CPU memory");
  TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  TORCH_CHECK(
      miss_global_indices.is_cuda() && miss_hot_indices.is_cuda() &&
          miss_counts.is_cuda() &&
          miss_global_indices.scalar_type() ==
              at::ScalarType::Int &&
          miss_hot_indices.scalar_type() ==
              at::ScalarType::Int &&
          miss_counts.scalar_type() == at::ScalarType::Int &&
          miss_global_indices.dim() == 2 && miss_hot_indices.dim() == 2 &&
          miss_global_indices.size(0) == miss_hot_indices.size(0) &&
          miss_global_indices.size(1) == miss_hot_indices.size(1) &&
          miss_counts.numel() >= miss_global_indices.size(0) &&
          miss_global_indices.is_contiguous() &&
          miss_hot_indices.is_contiguous() && miss_counts.is_contiguous(),
      "compact miss plan must be matching contiguous int32 CUDA tensors");
  TORCH_CHECK(hot_cache.dim() == 3,
                  "hot_cache must be [num_blocks, block_size, row_width]");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  TORCH_CHECK(row_bytes % 16 == 0, "KV rows must be 16-byte aligned");
  TORCH_CHECK(hot_cache.stride(1) * hot_cache.element_size() == row_bytes,
                  "hot-cache rows must be contiguous");
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  TORCH_CHECK(host_rows <= INT32_MAX, "host row indices must fit int32");
  const auto num_rows = static_cast<int32_t>(miss_global_indices.size(0));
  const auto top_k = static_cast<int32_t>(miss_global_indices.size(1));
  if (num_rows == 0 || top_k == 0) {
    return;
  }

  constexpr int kBlockSize = 512;
  constexpr int kTargetBlocks = 64;
  const int num_chunks = std::min(8, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  const int32_t hot_block_size = static_cast<int32_t>(hot_cache.size(1));
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  const c10::DeviceGuard device_guard(
      hot_cache.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hisparse_gather_compact_kernel<<<grid, kBlockSize, 0, stream>>>(
      mapped_host_pointer(host_cache),
      static_cast<char*>(hot_cache.mutable_data_ptr()),
      miss_global_indices.const_data_ptr<int32_t>(),
      miss_hot_indices.const_data_ptr<int32_t>(),
      miss_counts.const_data_ptr<int32_t>(), host_rows, hot_rows, row_bytes,
      hot_block_stride, hot_block_size, top_k);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "HiSparse gather_compact launch failed");
}

// Loaded explicitly as a dispatcher library, never imported as a Python module.
TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def(
      "hisparse_resolve_residency(Tensor host_cache,"
      "                 Tensor! hot_cache,"
      "                 Tensor hot_block_table,"
      "                 Tensor global_indices,"
      "                 Tensor! hot_indices,"
      "                 Tensor! device_global_indices,"
      "                 Tensor! lru_slots,"
      "                 Tensor? request_state_indices,"
      "                 int region_stride,"
      "                 int max_union_rows,"
      "                 Tensor(a!)? miss_mask=None,"
      "                 Tensor(b!)? stats=None,"
      "                 Tensor(c!)? attention_indices=None,"
      "                 int attention_block_stride=0,"
      "                 Tensor? request_ids=None,"
      "                 Tensor? source_block_table=None,"
      "                 int source_block_size=0,"
      "                 Tensor(d!)? resolved_global_indices=None,"
      "                 Tensor(e!)? valid_counts=None,"
      "                 Tensor(f!)? swap_host_physical_rows=None,"
      "                 Tensor(g!)? swap_device_physical_rows=None,"
      "                 Tensor(h!)? swap_counts=None,"
      "                 Tensor? resident_block_table=None,"
      "                 int resident_block_size=0,"
      "                 int resident_null_block=0) -> ()");

  m.def(
      "hisparse_invalidate_written_slots(Tensor! device_global_indices,"
      "                                   Tensor request_state_indices,"
      "                                   Tensor req_id_per_token,"
      "                                   Tensor written_slots) -> ()");

  m.def(
      "hisparse_gather_plan(Tensor host_cache,"
      "                     Tensor! hot_cache,"
      "                     Tensor global_indices,"
      "                     Tensor hot_indices,"
      "                     Tensor miss_mask,"
      "                     Tensor? request_state_indices,"
      "                     Tensor(a!)? attention_indices=None,"
      "                     int attention_block_stride=0) -> ()");

  m.def(
      "hisparse_gather_compact(Tensor host_cache,"
      "                        Tensor! hot_cache,"
      "                        Tensor miss_global_indices,"
      "                        Tensor miss_hot_indices,"
      "                        Tensor miss_counts) -> ()");

}
TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl("hisparse_resolve_residency", &hisparse_resolve_residency);
  m.impl("hisparse_invalidate_written_slots", &hisparse_invalidate_written_slots);
  m.impl("hisparse_gather_plan", &hisparse_gather_plan);
  m.impl("hisparse_gather_compact", &hisparse_gather_compact);
}
