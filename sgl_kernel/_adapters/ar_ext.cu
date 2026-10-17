// HCU custom all-reduce device extension (HIP C++).
//
// Measured HCU/gfx938 + DTK visibility rules (2026-10-11, GPU6/7 probes):
//   * cross-GPU writes (plain/atomic/volatile, any cache modifier) from a
//     triton OR C++ kernel never reach the peer — not even after kernel end
//   * cross-GPU spins are impossible on device-memory flags: reader caches
//     the line on first touch, invalidation only happens at kernel boundaries
//   * cross-GPU READS of device IPC memory are coherent (fresh at each
//     kernel's first touch of a line)
//   * host-mapped shared memory (/dev/shm + hipHostRegister) is the ONE
//     working signaling fabric: GPU accesses traverse PCIe uncached, writes
//     land in host DRAM, every poll fetches fresh — coherent mid-kernel
//     across all GPUs on the node (probe: 3/3 rounds observed at ~1.4us/poll)
//
// Design: BOTH the flags AND the payload live in host shm. Measured on
// HCU/gfx938: a writer's VRAM stores become remotely readable only some
// time AFTER kernel-end (fence/atomic flushes do NOT order them against a
// host-flag publish — probed: concurrent remote reads deterministically
// see the pre-store value). Host-mapped memory has none of these problems:
// GPU accesses are uncached PCIe transactions; posted writes from the same
// source are hardware-ordered, so data stores issued before the PUB store
// land in DRAM first, and every reader poll fetches fresh.
//
// Per round: stage my input into my host row -> fence (host stores complete)
// -> publish round r -> spin peers' publish -> pull-read peers' host rows ->
// reduce -> fence -> publish done -> commit seq. Round/slot derive from an
// on-device u64 sequence counter, so any number of CUDA-graph replays stays
// correct with zero host involvement; slot reuse is guarded by done flags.
//
// KNOWN RACE (verdict 2026-10-15, R-01 s7nccl A/B, nmz26 tp4): the
// rendezvous counts block arrivals with DEVICE-scope atomicAdd on a1/a2, but
// device-scope atomics do not push the OTHER blocks' host-shm payload stores
// out of the GPU (plain stores can linger past __threadfence_system on this
// stack — see the belt&braces note below). The last block's PUB/DONE write
// can therefore reach host DRAM before some payload writes of earlier
// blocks, and peers intermittently reduce stale (previous-round) or torn
// rows: Berlin skip-think flips 48/80 rounds with this AR vs 1/40 on the
// HCU_TRITON_AR=0 route (aiter AR GraphSafe). Default disabled
// (HCU_TRITON_AR=0) until the rendezvous moves to host-shm system-scope
// arrival counters, so every block flushes its own stores before its
// arrival is observed.
//
// Device buffer layout (bytes, must match triton_ar.py) — local sync only:
//   0x000 u64 seq | 0x008 u32 arrived1 | 0x00c u32 arrived2
//   0x010 u64 diag (budget-exhaustion counter)
// Host shm layout: [PUB ws x S i64][DONE ws x S i64][pad to 0x1000]
//   then payload rows: rank r slot s at 0x1000 + (r*S + s)*max_size.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define DEVINLINE __device__ __forceinline__

DEVINLINE float bf16_to_f32(uint16_t u) {
    return __uint_as_float(((unsigned int)u) << 16);
}
DEVINLINE uint16_t f32_to_bf16(float v) {
    unsigned int x = __float_as_uint(v);
    unsigned int lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return (uint16_t)(x >> 16);
}
DEVINLINE __half uh(uint16_t u) {
    __half h;
    __builtin_memcpy(&h, &u, 2);
    return h;
}
DEVINLINE uint16_t hu(const __half& h) {
    uint16_t u;
    __builtin_memcpy(&u, &h, 2);
    return u;
}

// bounded spin on a HOST-mapped int64 flag (uncached PCIe load each poll).
DEVINLINE long long spin_ge_ll(volatile long long* p, long long want,
                                long long budget) {
    long long it = 0, v = *p;
    while (v < want && (budget <= 0 || it < budget)) {
        v = *p;
        ++it;
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(256)
hcu_ar_kernel(
    const void* inp, void* out,
    const char* my_data,          // my slot region (device)
    const char* p0, const char* p1, const char* p2, const char* p3,
    const char* p4, const char* p5, const char* p6,   // peers, rank order, self skipped
    volatile long long* flags,    // host shm: PUB[ws][S] then DONE[ws][S]
    unsigned long long* sync,     // [0] u64 seq, [8] u32 a1, [12] u32 a2, [16] u64 diag
    long long n, long long slot_bytes,
    int ws, int rank, int S, int elem_mode,  // 0 fp32, 1 bf16, 2 fp16
    long long budget
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nb = gridDim.x;
    volatile unsigned* a1 = (volatile unsigned*)sync + 2;
    volatile unsigned* a2 = (volatile unsigned*)sync + 3;

    __shared__ unsigned long long s_k;
    __shared__ unsigned s_old;
    if (tid == 0) s_k = *(volatile unsigned long long*)sync;  // rounds completed
    __syncthreads();
    const unsigned long long k = s_k + 1;                     // rounds are 1-based
    const int slot = (int)(k % (unsigned long long)S);

    const char* peer[7] = {p0, p1, p2, p3, p4, p5, p6};
#define PUB(j, sl)  (flags + (size_t)(j) * S + (sl))
#define DONE(j, sl) (flags + (size_t)ws * S + (size_t)(j) * S + (sl))

    // ---- slot-free wait: peers finished reading my slot used at round k-S
    if (tid == 0 && k > (unsigned long long)S) {
        const long long need = (long long)(k - (unsigned long long)S);
        for (int j = 0; j < ws; ++j) {
            if (j == rank) continue;
            long long v = spin_ge_ll(DONE(j, slot), need, budget);
            if (v < need)
                atomicAdd((unsigned long long*)(sync + 2), 1ULL);
        }
    }
    __syncthreads();

    // ---- stage my slice into my host row (16B-vector stores: PCIe writes
    // coalesce into full-line transactions; ragged tail by the last block)
    char* slotp = (char*)((size_t)my_data + (size_t)slot * slot_bytes);
    {
        const long long total = n * (elem_mode == 0 ? 4 : 2);
        const long long nvec = total >> 4;
        const uint4* s = (const uint4*)inp;
        uint4* d = (uint4*)slotp;
        for (long long i = bid * 256 + tid; i < nvec; i += (long long)nb * 256)
            d[i] = s[i];
        if (bid == nb - 1) {
            const char* sb = (const char*)inp;
            char* db = (char*)slotp;
            for (long long b = nvec * 16 + tid; b < total; b += 256)
                db[b] = sb[b];
        }
    }
    __threadfence_system();
    __syncthreads();  // EVERY thread's stores are fenced before anyone counts

    // ---- rendezvous 1: last block publishes round r
    if (tid == 0) s_old = atomicAdd((unsigned*)a1, 1u);
    __syncthreads();
    if (s_old == (unsigned)nb - 1) {
        // belt & braces flush: plain stores can linger in L1 past
        // __threadfence_system on this stack (probed); a system-scope RMW
        // pushes the store stream to the coherence point
        __threadfence_system();
        atomicAdd_system((unsigned*)a2, 0u);
        __threadfence_system();
        *PUB(rank, slot) = (long long)k;
    }

    // ---- wait for every peer's publish (host spins, uncached)
    if (tid == 0) {
        for (int j = 0; j < ws; ++j) {
            if (j == rank) continue;
            long long v = spin_ge_ll(PUB(j, slot), (long long)k, budget);
            if (v < (long long)k)
                atomicAdd((unsigned long long*)(sync + 2), 1ULL);
        }
    }
    __syncthreads();

    // ---- reduce: my host row + peers' host rows (vectorized loads, fp32
    // accumulate, ragged tail by the last block)
    {
        const long long total = n * (elem_mode == 0 ? 4 : 2);
        const long long nvec = total >> 4;
        for (long long v = bid * 256 + tid; v < nvec; v += (long long)nb * 256) {
            uint4 mv = *((const uint4*)slotp + v);
            uint4 acc = mv;
            for (int j = 0; j < ws; ++j) {
                if (j == rank) continue;
                const char* pb = peer[j - (j > rank)] + (size_t)slot * slot_bytes;
                uint4 pv = *((const uint4*)pb + v);
                if (elem_mode == 0) {
                    float4 a = *reinterpret_cast<float4*>(&acc);
                    const float4 b = *reinterpret_cast<const float4*>(&pv);
                    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                    acc = *reinterpret_cast<uint4*>(&a);
                } else {
                    uint32_t* a32 = reinterpret_cast<uint32_t*>(&acc);
                    const uint32_t* b32 = reinterpret_cast<const uint32_t*>(&pv);
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        if (elem_mode == 1) {
                            float f = bf16_to_f32((uint16_t)(a32[w] & 0xffffu))
                                    + bf16_to_f32((uint16_t)(b32[w] & 0xffffu));
                            uint16_t lo = f32_to_bf16(f);
                            f = bf16_to_f32((uint16_t)(a32[w] >> 16))
                              + bf16_to_f32((uint16_t)(b32[w] >> 16));
                            uint16_t hi = f32_to_bf16(f);
                            a32[w] = (uint32_t)lo | ((uint32_t)hi << 16);
                        } else {
                            __half2 a2 = __halves2half2(
                                uh((uint16_t)(a32[w] & 0xffffu)),
                                uh((uint16_t)(a32[w] >> 16)));
                            __half2 b2 = __halves2half2(
                                uh((uint16_t)(b32[w] & 0xffffu)),
                                uh((uint16_t)(b32[w] >> 16)));
                            a2 = __hadd2(a2, b2);
                            __builtin_memcpy(&a32[w], &a2, 4);
                        }
                    }
                    acc = *reinterpret_cast<uint4*>(a32);
                }
            }
            *((uint4*)out + v) = acc;
        }
        if (bid == nb - 1) {
            const long long base_e = (nvec * 16) / (elem_mode == 0 ? 4 : 2);
            for (long long i = base_e + tid; i < n; i += 256) {
                if (elem_mode == 0) {
                    float a = ((const float*)slotp)[i];
                    for (int j = 0; j < ws; ++j) {
                        if (j == rank) continue;
                        const char* pb = peer[j - (j > rank)] + (size_t)slot * slot_bytes;
                        a += ((const float*)pb)[i];
                    }
                    ((float*)out)[i] = a;
                } else if (elem_mode == 1) {
                    float a = bf16_to_f32(((const uint16_t*)slotp)[i]);
                    for (int j = 0; j < ws; ++j) {
                        if (j == rank) continue;
                        const char* pb = peer[j - (j > rank)] + (size_t)slot * slot_bytes;
                        a += bf16_to_f32(((const uint16_t*)pb)[i]);
                    }
                    ((uint16_t*)out)[i] = f32_to_bf16(a);
                } else {
                    float a = __half2float(((const __half*)slotp)[i]);
                    for (int j = 0; j < ws; ++j) {
                        if (j == rank) continue;
                        const char* pb = peer[j - (j > rank)] + (size_t)slot * slot_bytes;
                        a += __half2float(((const __half*)pb)[i]);
                    }
                    ((__half*)out)[i] = __float2half(a);
                }
            }
        }
    }

    // ---- fence + rendezvous 2: last block publishes done, commits round,
    // resets the arrival counters (safe: everyone already passed both)
    __threadfence_system();
    __syncthreads();  // every thread's peer reads retired before DONE counts
    if (tid == 0) s_old = atomicAdd((unsigned*)a2, 1u);
    __syncthreads();
    if (s_old == (unsigned)nb - 1) {
        __threadfence_system();
        atomicAdd_system((unsigned*)a1, 0u);
        __threadfence_system();
        *DONE(rank, slot) = (long long)k;
        __threadfence_system();
        *(volatile unsigned long long*)sync = k;
        *a1 = 0u;
        *a2 = 0u;
    }
}

extern "C" hipError_t hcu_ar_launch(
    const void* inp, void* out,
    long long my_data,
    long long p0, long long p1, long long p2, long long p3,
    long long p4, long long p5, long long p6,
    long long flags, long long sync,
    long long n, long long slot_bytes,
    int elem_mode, int ws, int rank, int slots, long long budget,
    hipStream_t stream
) {
    const long long nbytes = n * (elem_mode == 0 ? 4 : 2);
    int blocks = (int)((nbytes + 32767) / 32768);
    if (blocks < 1) blocks = 1;
    if (blocks > 64) blocks = 64;
    void* args[] = {
        (void*)&inp, (void*)&out, (void*)&my_data,
        (void*)&p0, (void*)&p1, (void*)&p2, (void*)&p3,
        (void*)&p4, (void*)&p5, (void*)&p6,
        (void*)&flags, (void*)&sync,
        (void*)&n, (void*)&slot_bytes,
        (void*)&ws, (void*)&rank, (void*)&slots, (void*)&elem_mode,
        (void*)&budget,
    };
    return hipLaunchKernel(
        (const void*)hcu_ar_kernel, dim3(blocks), dim3(256), args, 0, stream
    );
}

// ---- host-pinned shared memory helpers -----------------------------------

extern "C" hipError_t hcu_register_host(void* ptr, size_t size, void** dev_ptr) {
    hipError_t e = hipHostRegister(ptr, size, hipHostRegisterMapped);
    if (e != hipSuccess) return e;
    return hipHostGetDevicePointer(dev_ptr, ptr, 0);
}

extern "C" hipError_t hcu_unregister_host(void* ptr) {
    return hipHostUnregister(ptr);
}

// ---- one-shot flag probes (visibility debugging) --------------------------

__global__ void hcu_flag_write_kernel(volatile int* f, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *f = v;
        __threadfence_system();
    }
}

extern "C" hipError_t hcu_flag_write(long long f_addr, int v, hipStream_t stream) {
    volatile int* f = (volatile int*)(uintptr_t)f_addr;
    void* args[] = {(void*)&f, (void*)&v};
    return hipLaunchKernel((const void*)hcu_flag_write_kernel, dim3(1), dim3(32), args, 0, stream);
}

__global__ void hcu_flag_spin_kernel(volatile int* f, int want, long long budget, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long it = 0;
        int v = *f;
        while (v < want && it < budget) {
            v = *f;
            ++it;
        }
        out[0] = v;
        out[1] = (int)(it >> 32);
        out[2] = (int)(it & 0xffffffff);
    }
}

extern "C" hipError_t hcu_flag_spin(long long f_addr, int want, long long budget, int* out, hipStream_t stream) {
    volatile int* f = (volatile int*)(uintptr_t)f_addr;
    void* args[] = {(void*)&f, (void*)&want, (void*)&budget, (void*)&out};
    return hipLaunchKernel((const void*)hcu_flag_spin_kernel, dim3(1), dim3(32), args, 0, stream);
}

// ---- data-transfer probe: does a fenced/atomic-flushed plain store become
// visible to a CONCURRENT remote plain reader? ------------------------------

extern "C" __global__ void hcu_stage_write_kernel(
    float* data, long long n, float val, volatile int* flag, int mode)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (long long i = 0; i < n; ++i) data[i] = val;   // plain stores
        if (mode == 0) {
            __threadfence_system();
        } else {
            // triton-proven flush form: release RMW on own memory, sys scope
            float four = 4.0f;
            atomicAdd_system(&four, 0.0f);
            __threadfence_system();
        }
        *flag = 1;
        __threadfence_system();
    }
}

extern "C" hipError_t hcu_stage_write(
    long long d_addr, long long n, float val, long long f_addr, int mode,
    hipStream_t stream)
{
    float* d = (float*)(uintptr_t)d_addr;
    volatile int* f = (volatile int*)(uintptr_t)f_addr;
    void* args[] = {(void*)&d, (void*)&n, (void*)&val, (void*)&f, (void*)&mode};
    return hipLaunchKernel((const void*)hcu_stage_write_kernel, dim3(1), dim3(32), args, 0, stream);
}

extern "C" __global__ void hcu_stage_read_kernel(
    const float* peer, long long n, volatile int* flag, long long budget,
    float* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long it = 0;
        while (*flag < 1 && it < budget) { ++it; }
        float mn = 1e30f, mx = -1e30f;
        for (long long i = 0; i < n; ++i) {
            float v = peer[i];        // plain remote read, first touch
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        out[0] = *flag;
        out[1] = mn;
        out[2] = mx;
        out[3] = peer[0];
    }
}

extern "C" hipError_t hcu_stage_read(
    long long p_addr, long long n, long long f_addr, long long budget,
    long long out_addr, hipStream_t stream)
{
    const float* p = (const float*)(uintptr_t)p_addr;
    volatile int* f = (volatile int*)(uintptr_t)f_addr;
    float* o = (float*)(uintptr_t)out_addr;
    void* args[] = {(void*)&p, (void*)&n, (void*)&f, (void*)&budget, (void*)&o};
    return hipLaunchKernel((const void*)hcu_stage_read_kernel, dim3(1), dim3(32), args, 0, stream);
}
