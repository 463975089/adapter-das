# vLLM 0.33 FP4 indexer compatibility candidate

This document records the accepted compatibility route for DeepSeek-V4.1 FP4
indexer pages on HCU. It is a candidate integration, not a production or
performance claim.

## Constrained PCP8 integration

The FP4 admission also accepts eager MRV2 TP1/PCP8/DP1/PP1/DCP1 with EP and
DeepEP high throughput. Nonadaptive DSpark uses an eager V4.1 TP1 draft;
the Layer Split serving profile uses five speculative tokens. The Adapter
compressor metadata builder retains a full native ring tail on every rank so
verification can resume after a rejected suffix. The paired Plugin admits
static EPLB through the calibrated expert map. Runtime EPLB, DBO and LoRA
are rejected by the PCP8 admission checks.
The Plugin-owned FP4 PCP attention uses the Adapter-owned PCP attention and
reuses native replicated-cache preparation,
local-query metadata and source-layer cache sharing, with the PCP Engram
implementation. The paired Plugin gives draft SWA caches replicated pools and
constructs draft MoE with the target's physical PCP/EP geometry. SWA decoder
replay checks that the draft window fits the target window. CPU and GPU
evidence for this DSpark candidate is tracked separately from the historical
results below. Core sources are not modified.

The teacher Adapter defaults to `lightop` and the packer to `triton`.
For comparisons explicitly set `HCU_VLLM_FP4_INDEXER_IMPL=stock` and
`HCU_VLLM_FP4_PACKER_IMPL=stock` for the FP8 arm; use `lightop`/`triton`
for the FP4 arm. These changes do not change the defaults.

The Plugin reader subchunks **query rows**, retaining the complete gathered K
window for every query. It reuses Core's `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
for the aligned FP32 score width, and releases each score matrix before the
next one. Decode capacity is frozen from scheduler limits at construction.
Profiling allocates and releases a temporary byte tensor for transient
prefill/decode scratch. The ratio-specific decode staging pools remain reserved
and are reused by real decode. Their bytes are accounted separately from
transient scratch, so the profile avoids retaining a duplicate score workspace.
Core records the transient allocated peak.

CPU tests cover admission, native constructor routing, source cache sharing,
eight-rank coordinate replay, CR1/CR2, empty/padded rows, query window and
candidate-row slicing, output equality and tensor lifetimes. Transport and GPU
kernel leaves in these CPU tests are doubles. The runtime budget estimates
explicit tensors. Vendor-private scratch and allocator behavior require device
measurements.

### Historical device validation on 2026-11-11

These results apply to the earlier pre-refactor revisions. The boundary-refactor
rebase requires fresh device and model validation.

The paired runs used eight HCU devices, the Channel-FP8 checkpoint, eager MRV2,
TP1/DP1/PCP8/EP8, DeepGEMM, DeepEP high throughput, logical block size 64 and
private CPU Engram offload with THP disabled. Both arms used GPU memory
utilization 0.9, max-num-batched-tokens 32768, max-model-len 65536 and
max-num-seqs 128. PC, SWA replay, speculation, EPLB, KV transfer and profiler
capture were disabled. Source package origins, identical commands and prompt
hashes were checked. The two environment differences selected the Indexer
and packer implementations.

| Check | Explicit FP8 stock arm | FP4 LightOp / Triton arm |
|---|---|---|
| Fixed 32768-token input, output 1, N1/C1 | First warmup failed with an 8.50 GiB allocation in `fp8_mqa_logits_torch` | Two warmups and three measured requests completed |
| Measured whole-request duration | Unavailable after OOM | 2.988899 / 2.987854 / 2.985082 seconds |
| Highest device-memory sample | 139.4059 GiB | 127.6222 GiB |
| Rank-0 available KV-cache budget | 18.52 GiB | 18.37 GiB |

Device memory was sampled every 0.25 seconds and includes weights, KV pools,
communication buffers and temporary tensors. These samples can miss an
instantaneous peak. The FP8 arm failed before completing its request, so this
comparison does not isolate a component's memory reduction or establish a
throughput speedup.

The FP8 failure occurred after server startup in the Torch score/ReLU/weight
expression. The separate FP4 profile-workspace lifetime change addresses a
persistent reservation in that FP4 path. The paired run validates the combined
FP4 implementation for this configuration.

HumanEval used its original prompts, concurrency 16, one generation per task,
temperature 0, thinking disabled and at most 2048 output tokens. All 164 task
IDs were present exactly once. The result was 161/164, pass@1 98.17%, with
three unit-test failures and no OOM or HTTP 500 errors. No fresh FP8 HumanEval
comparison was run. Both test services were stopped and the devices released.

CPU validation passed 166 tests and all eight existing FP8 PCP replay cases.
The Adapter L0 gate passed. Candidate status is retained for the documented
execution domain.

## Known deviation from the adapter layout rule

`adapter-das/skills/hcu-adapter-dev/SKILL.md:195-198` says not to copy a
cache tensor to work around layout. The decode path in this candidate is an
explicit, reviewed deviation from that rule. It exists because LightOp's
`paged_mqa_logits_fp4` ABI accepts only canonical 64-record pages. This is a
compatibility path, not the design answer.

The prefill path does not stage. `dsv41_gather_index_k_fp4(cache, slots,
page_size)` receives the allocator page size and gathers directly. Decode uses
staging only when the allocator page has a record count other than 64.

## ABI evidence

The inspected LightOp implementation contains `constexpr int kPageSlots = 64`
and checks the cache geometry against `[pages,64,1,68]` or `[pages,4352]`.
The source evidence is the inspected LightOp source
`csrc/paged_mqa_logits/paged_mqa_logits_fp4.cu:699-719` in the 0.6.0 source
bundle. The installed binary independently exposes the same rejection strings
in `workspace/issue/pr/20261108_v033_fp4_indexer/evidence/zerocopy_probe/native_abi_strings_symbols.log:10-12`.
The single-card probe rejected 32-record views and accepted the canonical
64-record control. It also rejected `table_block_size=32` and accepted the
64-record, `table_block_size=64` control.

## Staging shape and cost

For a physical page with `R != 64` records, decode gathers one payload
`int8[64]` and one `int32` scale word per logical slot. It assembles canonical
pages with 4096 payload bytes followed by 256 scale bytes, or 4352 bytes per
staged page. Capacity is static and uses
`pages_per_row = (max_context_len + 63) // 64`.

The staging pool allocation is:

```text
rows * pages_per_row * 4352 bytes
```

For example, with `max_context_len=4096`, `pages_per_row=64`, one decode row
costs `64 * 4352 = 278,528` bytes, about 272 KiB. Eight rows cost 2,228,224
bytes, about 2.125 MiB. The actual allocation scales with the layer's static
`max_context_len` and the flattened decode row count.

The operation performs one device-side byte gather per decode call. It is not
zero-copy and no performance improvement is claimed. Invalid columns use slot
zero only as an in-bounds read and remain masked by the native scorer's
`context_lens`.

## Removal plan

Remove staging only after one of these contracts is true and covered by a
native probe and regression test:

1. LightOp `paged_mqa_logits_fp4` accepts 32/128-record pages and a matching
   variable table-block contract.
2. The vLLM allocator is changed to provide uniform 64-record FP4 pages for
   every admitted ratio, with producer and metadata semantics preserved.

Until then, the compatibility route requires review acceptance. CPU/mock tests
prove shape and metadata behavior only. They do not prove native correctness,
end-to-end accuracy, or performance.
