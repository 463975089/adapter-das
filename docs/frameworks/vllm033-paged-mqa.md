# vLLM/plugin non-MX FP8 paged MQA (candidate)

## Why this is adapter work

The existing DeepInfer paged-MQA emulation accepts SM100 MX scales and
varlen scheduling. The existing vLLM C4 bridge adapts a different sparse
indexer class. Neither establishes support for plugin callers of vLLM's
generic non-MX paged-MQA ABI. HCU HYV4 Flash IndexPool is such a caller.

Model pooling, SWA, top-k policy and cache production stay in vllm-plugin-das.
This adapter change adds the missing generic numerical contract and an
explicit provider API, reusing the existing page reader and Torch math.
No official vLLM files, global DeepGEMM names or framework capability
predicates are patched. Existing SGLang preshuffled routes stay untouched.

## API and exact cache layout

`hcu_adapter.bridges.vllm_paged_mqa.get_paged_mqa_provider()` returns
`(get_paged_mqa_logits_metadata, fp8_fp4_paged_mqa_logits)`. The scorer
accepts vLLM's `max_model_len=` keyword. Selection is explicit at model
construction, including spawn workers. Importing the bridge alone does
not initialize Torch or a GPU. Registration is an idempotent direct seam,
not a global replacement of functions that may already have aliases.

Supported non-MX operands:

- Q: E4M3FN `[B, 1, H, 128]`, `1 <= H <= 64`, query scales already folded
  into FP32 weights `[B, H]`.
- Cache: uint8 `[pages, 64, 1, 132]`. Physical bytes in each page are
  `[64*128 row-major FP8 key bytes | 64 FP32 scales]`. Page stride may
  exceed the payload size (shared/layer-offset storage). This is NOT
  a per-token interleaved 132-byte record or a preshuffled SGLang cache.
- Context lengths: int32 `[B]` or `[B, 1]`; block table int32 `[B, columns]`.
  Valid positions must map to allocated pages; padding is masked.
- FP32 logits `[B, max_model_len]`, weighted sum of per-head ReLU dot
  products. Empty rows and positions beyond the row's length are zero.
- Only `next_n=1` is supported. Unsupported geometry/dtype fails explicitly.
  Optional indices group scheduling only; table row i belongs to Q row i.

Metadata is opaque zero scheduling storage for the Torch implementation;
it is NOT a fabricated native kernel capability. MX consumers retain the
existing BF16 output contract. The candidate chunks both scores and gathered
keys; it adds no data-dependent host reads or synchronization.

## Governance and rollback

Catalog: `vllm_paged_mqa_logits`, backend `emu`, status **candidate**.
The snapshot and generated tier matrix track the same entry. Runtime
selection is visible through `registry.register`/adapter dump. No new env
variable is introduced. `HCU_DISABLE_OPS=vllm_paged_mqa_logits` refuses
provider selection in a fresh process; it does not fall back to CUDA and
cannot hot-rewrite already captured graphs.

The Torch route is deliberately a correctness baseline. Serving-memory footprint
and throughput have not been qualified; do not advertise this candidate
as equivalent to the existing optimized SGLang or C4 indexer routes.

## Tests and remaining gates

```sh
python -m pytest tests/kernels/test_vllm_paged_mqa_fp32.py
python -m pytest tests/test_ops_cpu.py
python tools/gen_tier_matrix.py --check
```

The first suite checks independent FP32 math, scale/layout interpretation,
non-contiguous/shared-page storage, context masking, chunking, empty batches,
ABI/kill-switch registration and preservation of MX BF16 behavior. A changing
Q/weights/lengths/block-table graph replay test is skipped without a device.
CPU success does not qualify HCU eager/Graph numerics or the complete model.
Run the suite on real HCU, the repository kernel/adapter/vendor gates and
model text/VL loading and end-to-end accuracy before promotion or merging.

## Recorded focused validation (2026-11-17)

BW1100/gfx938, Python 3.12.12, Torch 2.13.0, HIP 7.2.26365,
vLLM 0.33.0+dtk2610.torch2130.2611101131.g7364da:

- Paged-MQA suite plus existing CPU ops: **76 passed**, with real H32/H64,
  per-slot scales, nonzero storage offsets, eager numerics and changing-input
  Graph replay (Q, K bytes, scales, weights, lengths and tables).
- Companion plugin registration/seam/real Triton-writer-to-adapter tests:
  **20 passed**; this is cross-repository evidence, not an adapter-only gate.
- Focused tests used source overlays in an isolated temporary directory;
  installed framework/adapter packages were not rewritten.

These are operator and seam results, NOT a full kernel/adapter/vendor sweep,
installed-wheel qualification, checkpoint loading or serving-accuracy result.
Keep candidate status and the dependent model PR in draft until those gates
are completed. No benchmark or performance promotion is implied.
