# vLLM/plugin non-MX FP8 paged MQA (candidate)

## Default AITER provider

The installed HCU vLLM bridge defaults to the vendor AITER Opus Paged MQA
interface. A startup JSON file is optional. The existing public `ops` setting
can explicitly select `aiter`, `triton`, or `emu`.

The native path accepts gfx936/gfx938, non-MX E4M3FN Q, page64/D128 row-major
FP8 keys followed by FP32 page-tail scales, and 32/64 query heads. Shared
physical pages with gaps between layers use the existing official vLLM Triton
provider. The bridge avoids copying the entire allocated KV cache per call.
Unavailable native modules and unsupported metadata also use that provider.
Native launch failures propagate. AITER selects its own kernel configuration.

Zero-length graph rows receive safe temporary lengths and page tables on the
device. An independent FP32 output masks those rows and all tail columns to
positive zero. Query, lengths, and page-table changes are consumed on replay.



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

Catalog: `vllm_paged_mqa_logits`, backends `aiter`, `emu`, and `triton`, all **candidate**.
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


## Opt-in reuse of the official vLLM Triton launcher

The default is `aiter`. To explicitly select the existing Triton provider through `HCU_CONFIG`, use:

```json
{"version": 1, "ops": {"vllm_paged_mqa_logits": "triton"}}
```

This explicit provider reuses
`vllm.v1.attention.ops.rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits_triton`;
it does not replace global Core names, route through the differently laid-out
SGLang cache, or add a private environment variable. The metadata function
returns `None` after validating page64 because this launcher reads device
lengths directly. Startup `HCU_DISABLE_OPS` takes precedence over config.

The fast face is HIP/E4M3FN, next_n=1, H32/H64, D128 and aligned page64
storage. Contiguous lengths and unit inner query/weights/table strides are
required. Other valid generic ABI faces retain the existing Torch reader;
MX is explicitly refused. Kernel/import errors on the supported fast face
are not silently swallowed. Importing the provider getter stays Torch-free.

The official helper returns workspace-backed scratch and leaves padding
undefined. The adapter must return an **out-of-place**, zero-masked FP32
tensor so an earlier result cannot be overwritten by a subsequent call.
Never initialize/reset vLLM's workspace manager inside a production provider.
Tests initialize it only in a fixture, including changing-input Graph replay.

Focused HCU measurements and full-model accuracy are separate gates. An
operator microbenchmark is not an end-to-end throughput claim. Keep both
catalog entries candidate pending full repository, deployment and model gates.

The catalog lists `aiter` first. Explicit `emu` and `triton` choices use the
existing op selection. Existing configuration files retain their explicit choice.


## Pinned community DFly Core: separate compatibility gate

The HY4 Flash DFly qualification also checks the Core merged by
vllm-project/vllm PR #68296 (`b067b6f6fa509a8c98c5a79d75b2aa3742d539c6`).
An installed v0.33 wheel without that method cannot accept `method="dfly"`;
a source version label alone does not prove speculator/attention/cache ABI
compatibility. Do not turn a paged-MQA operator result into a DFly accuracy claim.

The existing mHC bridge admits the two inspected delayed-pre signatures:
the original v0.33 signature and the same signature with a final optional
`fn_packed=None` projection cache. Other names/defaults remain refused before
publication. The HCU wrapper deliberately uses the original dense `fn`, as
Core's `forward_native` fallback does; it does not reinterpret or claim to
accelerate the packed cache. No custom-op schema, math, default tier, environment
variable or vendor capability is changed. Contract/lifecycle tests cover both
ABIs, wrong cache defaults, unknown tails and dense-vs-packed argument routing.
This does not qualify the newer Core's full mHC path or the HY4 DFly model.


The same Core probe found a second known ABI: `rocm_sparse_attn_prefill`
adds a trailing `kv_cache_dtype="auto"`. The bridge admits only that exact
extension or the original 12-parameter signature. The wrapper preserves the
scheme in stock calls; MXFP4 and other unqualified schemes must not enter the
BF16 native candidate. Wrong defaults and unknown signature changes still fail
before publication. Existing vendor error propagation and owner/lifecycle
checks remain intact. These ABI contracts are CPU/mock-tested; neither adds
new math or certifies a newer Core/native-wheel combination.

A whole newer Core is not automatically compatible with an older DTK wheel:
the isolated fixed-SHA configuration probe also encountered the independent
MiniMax fused-wrapper ABI guard. It was not bypassed. Use matching binaries or
a scoped DFly backport, and qualify the actual model before claiming support.


## Companion HY4 DFly source-overlay evidence (2026-11-21)

The companion plugin's explicit v0.33 DFly backport completed EvalScope 1.15
HumanEval: **161/164 (98.1707%)**, high reasoning, native thinking budget 4096,
total model length 8192, five speculative tokens, FLASHMLA_SPARSE, TP8/eager,
on BW1100/gfx938. All responses stopped normally with no truncation, empty
finals, syntax failures or request errors. The exact vendor runtime was
`0.33.0+dtk2610.torch2130.2611101131.g7364da` (Torch 2.13/HIP 7.2).
This is a companion source-overlay result, not independent installed-wheel,
image, standard-dispatch or full-model Graph qualification of this adapter.
The identical-budget target comparison is still running; neither DFly speedup
nor a matched-control accuracy gain is established. That historical qualification used `emu`. The current default is described
in the AITER provider section above.
The inspected mHC/FlashMLA ABI extensions do not certify arbitrary newer Core
or binary combinations. No additional native capability, environment workaround,
MiniMax ABI whitelist relaxation or global DeepGEMM rebinding is introduced.


### Current-main integration gate (2026-11-21)

The MR also integrates upstream main `af8b0ea8ca383dd929b15255e22e69e860bab173`,
including its package-layout migration. The new seam/tests use
`hcu_adapter.infra` imports; upstream's current health/all-reduce fixtures are
preserved rather than reverting them to older fixtures. The same focused Linux
CPU lane plus current built-in health contracts reports **221 passed,33 skipped**;
L0 on a Git-index-byte-verified LF snapshot is **GREEN (0 warnings)**.
An initial Windows-converted archive failed shell-LF checks; it is retained as
invalid packaging evidence, not suppressed or counted as passing. Archive export
must disable checkout newline conversion and match Git blobs before qualification.

The 161/164 end-to-end result above belongs to the earlier immutable accuracy
overlay. These new main-integration CPU/structural checks do not transfer that
result to the final merged adapter or establish HCU model/image qualification;
those validations remain pending. No running accuracy service was changed.

## Exact owned-logits padding candidate — 2026-11-24

The explicitly selected non-MX HIP Triton provider now copies its workspace
logits through a private tiled mask/copy kernel. Valid FP32 payload bits are
unchanged; columns at or beyond the original per-row length become positive
zero. Output storage remains fresh and independent of the Core workspace.
The scorer, TopK, attention arithmetic, context limits and generic fallbacks
are unchanged. The private module is imported only after native-path admission.

Actual gfx938 checks: 20 bitwise/ownership/pitched-offset/int32+int64/empty and
changed-input HIP Graph tests passed; 259 existing public-provider/alignment
tests passed, zero failures/skips, with 14 retained Torch deprecation warnings.
Graph-replay component timings at width 262144 and Q1/8/64 showed reduced
copy/mask time; these timings do not measure the scorer or model throughput.
Full-model accuracy and FULL_AND_PIECEWISE throughput remain separate gates.
Run the new focused tests with:

```sh
python -m pytest -q tests/kernels/test_vllm_paged_mqa_mask.py
```

The private launch uses an explicit tensor-device guard and the current stream
on that device. The explicit two-device regression (ambient device0, delayed
producer on a nondefault stream of device1) runs separately so a one-GPU test
lane acquires no new skipped case:

```sh
HIP_VISIBLE_DEVICES=0,1 python tools/check_paged_mqa_mask_multidevice.py
```
