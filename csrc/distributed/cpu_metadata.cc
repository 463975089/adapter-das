// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Small process-shared CPU int32 SUM. No Torch, HIP, or Python ABI dependency.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <sched.h>
#include <time.h>

namespace {
constexpr uint32_t kMaxWorld = 32;
constexpr uint32_t kMaxElems = 256;
constexpr uint64_t kMagic = 0x4843554d45544131ULL;
struct alignas(64) Slot {
  std::atomic<uint64_t> ready{0}, done{0};
  alignas(64) uint32_t values[kMaxElems]{};
};
struct alignas(64) Shared {
  uint64_t magic = 0;
  uint32_t world = 0, elems = 0;
  std::atomic<uint32_t> aborted{0};
  alignas(64) std::atomic<uint64_t> entered[kMaxWorld]{};
  Slot banks[2][kMaxWorld];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);

uint64_t monotonic_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}

int wait_for(Shared* s, int bank, uint64_t epoch, bool done,
             uint64_t start, uint64_t timeout_ns) {
  uint64_t spins = 0;
  for (;;) {
    if (s->aborted.load(std::memory_order_acquire)) return -3;
    bool all = true;
    for (uint32_t r = 0; r < s->world; ++r) {
      auto& seq = done ? s->banks[bank][r].done : s->banks[bank][r].ready;
      if (seq.load(std::memory_order_acquire) < epoch) {
        all = false;
        break;
      }
    }
    if (all) return 0;
    if ((++spins & 63) == 0) {
      if (timeout_ns && monotonic_ns() - start >= timeout_ns) {
        s->aborted.store(1, std::memory_order_release);
        return -2;
      }
      sched_yield();
    }
  }
}

Shared* checked(void* mem, uint64_t bytes) {
  if (!mem || reinterpret_cast<uintptr_t>(mem) % alignof(Shared)
      || bytes < sizeof(Shared)) return nullptr;
  return static_cast<Shared*>(mem);
}
}  // namespace

extern "C" uint64_t hcu_meta_size_v1() { return sizeof(Shared); }

extern "C" int hcu_meta_init_v1(void* mem, uint64_t bytes, int world, int elems) {
  if (!checked(mem, bytes) || world < 1 || world > int(kMaxWorld)
      || elems < 1 || elems > int(kMaxElems)) return -1;
  auto* s = new (mem) Shared{};
  s->world = uint32_t(world);
  s->elems = uint32_t(elems);
  // The bootstrap collective publishes initialization before peers map it.
  s->magic = kMagic;
  return 0;
}

extern "C" int hcu_meta_abort_v1(void* mem, uint64_t bytes) {
  auto* s = checked(mem, bytes);
  if (!s || s->magic != kMagic) return -1;
  s->aborted.store(1, std::memory_order_release);
  return 0;
}

extern "C" int hcu_meta_sum_v1(void* mem, uint64_t bytes, int rank,
                               uint64_t epoch, int32_t* data, int elems,
                               uint64_t timeout_ns) {
  auto* s = checked(mem, bytes);
  if (!s || s->magic != kMagic || !s->world || s->world > kMaxWorld) return -1;
  if (s->aborted.load(std::memory_order_acquire)) return -3;
  if (rank < 0 || uint32_t(rank) >= s->world || elems != int(s->elems)
      || !data || reinterpret_cast<uintptr_t>(data) % alignof(int32_t)
      || !epoch || epoch == std::numeric_limits<uint64_t>::max()) {
    s->aborted.store(1, std::memory_order_release);
    return -1;
  }
  uint64_t previous = epoch - 1;
  if (!s->entered[rank].compare_exchange_strong(previous, epoch)) {
    s->aborted.store(1, std::memory_order_release);
    return -1;
  }
  const uint64_t start = timeout_ns ? monotonic_ns() : 0;
  const int bank = int((epoch - 1) & 1);
  int code = 0;
  // Do not overwrite a bank until every reader finished its previous epoch.
  if (epoch > 2 && (code = wait_for(s, bank, epoch - 2, true, start, timeout_ns)))
    return code;
  auto& slot = s->banks[bank][rank];
  std::memcpy(slot.values, data, size_t(elems) * sizeof(int32_t));
  slot.ready.store(epoch, std::memory_order_release);
  if ((code = wait_for(s, bank, epoch, false, start, timeout_ns))) return code;
  for (int i = 0; i < elems; ++i) {
    // Unsigned addition defines exactly the native int32 wraparound behavior.
    uint32_t sum = 0;
    for (uint32_t r = 0; r < s->world; ++r) sum += s->banks[bank][r].values[i];
    std::memcpy(data + i, &sum, sizeof(sum));
  }
  slot.done.store(epoch, std::memory_order_release);
  return 0;
}
