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
  transplanted. Eager gfx938 coverage is recorded below; other hardware wave
  configurations and concurrent execution remain separate qualification gates.
- `gather_*` verifies HIP-registered pinned CPU storage and obtains its mapped
  device alias with `hipHostGetDevicePointer`. The host VA is never passed as a
  device pointer. The caller must retain pinned storage and not overwrite it
  until the stream finishes. This port does not silently register pageable RAM,
  synchronize transfers, invent a pinned ring, or guarantee concurrent CPU/GPU
  coherence. Mapped interior pointers are covered by eager device tests below. Asynchronous
  source lifetime beyond the tested synchronization discipline and multi-stream
  reuse remain device validation gates.
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
function bodies, HIP compilation and eager GPU correctness are separate evidence.
The host model cannot detect device races, stream/DMA lifetime bugs, hardware
wave divergence, HIP Graph capture/replay, or model accuracy. The target-only /
eager admission remains necessary; loading four dispatch entries does not grant
new architecture, DFly, Graph or full-model capabilities.

State/cache ownership is a caller ABI prerequisite: active request rows must be
contiguous, independently active leaders must use distinct state rows and
non-overlapping writable hot regions, and replay destinations must not race.
Resolve/invalidate/gather/attention that share state must be stream-ordered or
explicitly event-ordered. Pinned host storage must remain live and unmodified
until the producing stream finishes. The launcher's scalar shape checks cannot
prove these metadata content and lifetime conditions.

Required later gates: paired repository-free installed plugin/adapter delivery,
actual target-only model accuracy for FP8 and W4A8, full backend attention replay,
concurrent/multi-stream reuse, other supported hardware configurations, then
separately qualified Graph and owner-aware DFly support. Never promote catalog
state based on compiler, CPU or isolated native results alone.

## Recorded eager HCU qualification — 2026-11-22

The exact runtime source at commit `371360626b16e153c860b3bb37a0d9479239199b`
(kernel SHA256 `f86627a25f49482419dbe0d385ee0f7c9431c4c82bed24ab69d7ba063e47da36`)
was rebuilt in an isolated directory for **gfx938**, with `MAX_JOBS=2`, GPUs
hidden for compilation, no site installation and no existing package modification.
Source verification covered 922 tracked files with zero mismatches. The paired
Torch was `2.13.0+dtk2610.2611161236.g6ac259`; installed Core was
`0.33.0+dtk2610.torch2130.2611101131.g7364da`.

The fresh binary SHA256 was
`db27ecdb0b1dac6713590299b2aaa99191081b710625d85d234832b66002f48f`.
All four exact dispatcher entries registered without initializing a GPU context
in the CPU-only registration probe. Main-agent execution then used logical device0
in the container's existing visibility mapping: **44 passed / 0 failed / 0 skipped**
(21.04 seconds pytest time). This includes the original36 native cases and eight
additional BF16/opaque-FP8-DS byte-copy layout cases. The qualification snapshot
kept the compiled source immutable; the extra test was staged separately.

This is an isolated **eager native test result**, not an installed plugin test,
attention numerical parity, full-model accuracy, Graph, DFly, sanitizer pass or
performance claim. Independent static review found no confirmed actionable new
defect in this snapshot, subject to the caller ABI prerequisites above; it did
not approve Ready.

### Reproduce the isolated native lane

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

After ensuring the selected card is free and GPU use is authorized, run from
an immutable adapter checkout, without installing anything:

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

`test_typed_copy.py` additionally covers BF16 rows of576 elements (1152bytes)
and opaque FP8-DS MLA uint8 rows of656bytes, physical block64/256, plan/compact
paths, pinned interior storage, padded page pitch, cross-block destinations,
missing-host zero-fill and whole-allocation guard bytes. These are exact payload
transfer/layout checks, not attention or quantization numerical validation.
