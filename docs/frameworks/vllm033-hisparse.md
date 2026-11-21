# Core033 HiSparse native HIP provider (candidate)

## Scope and integration

The companion Plugin may call:

```python
from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
ensure_hisparse_ops()
```

Only call this for an explicitly enabled **target-only, eager HiSparse** request,
not ordinary model/Flash/DFly startup. Importing the module does not import Torch,
vLLM or sgl_kernel, compile, load a native library, or inspect/select a GPU.
`ensure_hisparse_ops` first loads the host's `vllm._C`, checks the four exact
Core033 schemas and CUDA dispatch entries, then explicitly loads the paired
precompiled native library if all four are absent. HIP Torch uses the `CUDA`
dispatch key. A partial/mismatching set is refused, never overwritten or mixed.
It does not add CPU/fake implementations to make capability checks pass.

The dispatcher fragment defines the unchanged names in `_C_cache_ops`:

- `hisparse_resolve_residency`
- `hisparse_invalidate_written_slots`
- `hisparse_gather_plan`
- `hisparse_gather_compact`

The loader is process-local, lock-serialized, idempotent, fail-latched after an
initialization error, and rejects inherited state before touching locks after
fork. Use spawn/fresh processes, not a fork-inherited provider. Explicit staged
qualification can supply `ensure_hisparse_ops(library_path=...)`; production's
no-argument API finds `sgl_kernel.hisparse_ops` without importing sgl_kernel's
vendor bootstrap. There is **no runtime JIT or fallback**. An absent artifact
is an actionable startup error. `HCU_DISABLE_OPS=vllm_hisparse` vetoes admission;
restart to change an already published dispatcher provider. It is not unloaded
or switched under a running engine.

## Native build

This is a separate extension under the existing `setup_hip.py` HIP build,
installed with the paired native sgl-kernel wheel. It does not redefine or load
`sgl_kernel.native_ops`. It uses the installed Torch ATen ABI, **not** the
upstream CUDA stableTorchABI: rebuild for the paired Torch/DTK wheel.

For an isolated, CPU-only compiler lane (no install, no device execution):

```sh
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
MAX_JOBS=2 AMDGPU_TARGETS=gfx938 DAS_PLATFORM_TAG=dtk2610.torch2130 \
python setup_hip.py --hisparse-only build_ext \
  --build-temp /tmp/hisparse-build/obj --build-lib /tmp/hisparse-build/lib
```

Select architecture(s) explicitly; otherwise existing setup architecture
detection may query a GPU. `--hisparse-only` avoids even constructing the
unrelated extension build. No site-packages, DTK header, official Core file,
running worker, model service or global installation is changed.

## Porting decisions and boundaries

Source lineage: the inspected 1438-line Core033 `csrc/hisparse_kernels.cu`
and its four `torch_bindings.cpp` schemas. Keep the vLLM copyright and Apache
license on the derived native source. The provider is a correctness-first HIP
port, not blind hipify of the CUDA/PTX implementation:

- Hash-union construction is block-parallel. Metadata hit/eviction/miss ordering
  and LRU publication are deliberately serial within one block. No ballots,
  shuffles, masked scans, `__syncwarp`, PTX, or hardware warp-size assumptions
  remain. This sacrifices metadata throughput; no performance claim is made.
- Byte copies use independent **software groups of 32 threads**, not warp
  collectives. On wave64 two groups occupy a wave; on wave32 one does. Vector
  loads/stores use ordinary HIP operations; CUDA L2-only cache hints are not
  transplanted. Parallel coherency and actual wave execution still require HCU
  validation.
- `gather_*` verifies HIP-registered pinned CPU storage and obtains its mapped
  device alias with `hipHostGetDevicePointer`. The host VA is never passed as a
  device pointer. The caller must retain pinned storage and not overwrite it
  until the stream finishes. This port does not silently register pageable RAM,
  synchronize transfers, invent a pinned ring, or guarantee concurrent CPU/GPU
  coherence. Mapped interior pointers, asynchronous source lifetime, and
  multi-stream reuse remain device validation gates.
- All device tensors must share the hot cache's HIP device. Row bytes are dense,
  positive and 16-byte aligned; 3D physical pages may have extra pitch but must
  not overlap. Grid/index narrowing, nonnegative block parameters, int16 slots,
  state bounds, and shared-memory budget are checked. Use the current stream
  with a device guard and check every launch without synchronizing. Invalid
  device metadata that could cause an unbounded state/cache access traps.
- Per-request rows and state ownership follow Core033's contiguous-run contract.
  Multiple independently active requests must not alias the same state region.
  Host scalar checks do not read GPU metadata back to validate this contract.
- Include **#69057's stable valid-front attention-index packing semantics**,
  implemented as a serial in-place filter, not its CUDA warp scan. Both the
  resident-only early return and swap path pack valid indices in input order,
  fill the tail with `-1`, and recompute `valid_counts`. The physical copy plan
  (`hot_indices`, `miss_mask`, compact swaps) retains original column positions.
  Without the optional counts+attention pair no packing is performed.

## Not supported / not yet qualified

This is an explicit **candidate**, not production-ready HiSparse support.
There are no owner parameters or migrations to newer backend abstractions.
The Plugin must fail closed for **HiSparse+DFly**: the inspected #69057 runtime
baseline already has `owner`-aware binding while the pinned Core033 does not.
Compaction correctness alone does not bridge that ABI gap.

CPU loader/schema tests, a serial host execution model of the actual native
function bodies, and HIP compilation are separate evidence from GPU correctness.
The host model cannot detect device races, stream/DMA lifetime bugs, hardware
wave divergence, HIP Graph capture/replay, or model accuracy. No GPU/Graph/model
execution was authorized in this development lane. The Plugin's target-only /
eager admission remains necessary; loading four dispatch entries does not grant
new architecture, DFly, Graph or full-model capabilities.

Required later gates: independent reference comparisons for every output/state,
holes/padding/resident-only/eviction/shared layers, mapped-host offset/lifetime,
page pitch and changed inputs, wave32/wave64 where supported, actual target-only
model accuracy, then separately qualified Graph and owner-aware DFly support.
Never promote catalog state based on a compiler or CPU result alone.

## Prepared HCU numerical lane — NOT RUN during this task

`tests/kernels/hisparse/test_native.py` contains executable real-device tests
with an independent Python/list reference and exact integer/byte comparisons:
all resolver outputs, counts, stats, LRU and ownership; holes and stable packing;
resident-only/mixed/padding/global indices; multi-request permuted state rows;
eager changed inputs; rewritten-host-row invalidation and recopy; plan/compact
copies and missing-host zero-fill; physical page gaps and non-unit output strides;
pinned CPU interior offsets 0/3/19; 2D staging; empty launches and bad-input refusal.
Whole backing storage is compared, including gaps and guard bytes. Initial cached
payloads match ownership. The backing pinned allocation is retained through an
explicit stream synchronization before CPU mutation or checking.

Only after the occupying model job has naturally finished and GPU use has been
authorized, run from the immutable adapter checkout, without installing anything:

```sh
HCU_ADAPTER_HOST=none VLLM_PLUGINS=none PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" python -m pytest -q tests/kernels/hisparse \
  --hisparse-library /absolute/isolated-build/lib/sgl_kernel/hisparse_ops.cpython-312-x86_64-linux-gnu.so \
  --hisparse-device 0
```

Select the free logical device with the existing visibility mechanism as needed.
Do not run this command while another model owns the card. Collect-only with all
GPU visibility empty is safe and does **not** constitute numerical validation.
Requires-HIP/no-device skips are explicitly **not passed device gates**. A missing
or incompatible library fails rather than silently skipping. This is eager-only;
it neither exercises nor grants Graph or DFly qualification. The independent
reference's known-answer CPU tests do not substitute for running this lane.
