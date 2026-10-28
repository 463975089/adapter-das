# Multi-host Bootstrap Design

## Context

`sitecustomize.py` currently contains 3191 lines of sglang bootstrap logic. It
is imported by every Python process whose `PYTHONPATH` contains this repository,
even when sglang is absent. The same environment can contain both sglang and
vLLM, because sglang consumes parts of `vllm._custom_ops`; installed-package
presence therefore cannot identify the active host.

The migration is split into two merge requests so structural movement and new
vLLM runtime behavior can be reviewed independently.

## MR 1: Structural bootstrap migration

MR 1 changes ownership and dispatch without changing the installed sglang
bridge sequence or numerical implementations.

The target layout is:

```text
sitecustomize.py
hcu_adapter/
  __init__.py
  bootstrap.py
  bridges/
    __init__.py
    common.py
    sglang.py
    vllm.py
    deepinfer.py
```

`sitecustomize.py` contains only:

```python
from hcu_adapter.bootstrap import install

install()
```

`bootstrap.install()` is process-idempotent. It first establishes the common
guards, then installs one host bridge. Host selection uses this precedence:

1. an explicit `HCU_ADAPTER_HOST=sglang|vllm|none` operator choice;
2. an inherited automatic selection made by the parent process;
3. an unambiguous launch intent in `sys.argv` or an already imported host;
4. availability probing, selecting a host only when the result is unambiguous;
5. the legacy sglang default when both packages are present and launch intent
   is absent, preserving `python -c 'import sglang'` behavior.

An automatically selected host is propagated to spawned children through a
private adapter-owned environment value. The public override is registered in
the environment registry and documented; users do not need to set it for a
normal deployment. A non-HIP interpreter and a HIP interpreter with neither
host installed are silent no-ops.

`bridges/common.py` owns only cross-framework, pre-import guards: the HCU
environment summary, NCCL topology guard, and tvm-ffi cold-cache recursion
breaker. It must run before either host bridge and must not import sglang or
vLLM.

`bridges/sglang.py` owns the existing sglang installation order. The migration
is mechanical: existing behavior, guards, exception isolation, diagnostics,
and implementation bodies remain unchanged. The existing vLLM custom-op
RMSNorm rebind is extracted to `bridges/vllm.py` and invoked from the same point
in the sglang sequence. In MR 1 this module is a reusable bridge, not the vLLM
platform bootstrap.

`bridges/deepinfer.py` is a side-effect-free future seam; bootstrap does not
select it in either MR.

The package configuration must include `hcu_adapter` and all existing
`sgl_kernel` subpackages in built wheels.

## MR 2: vLLM v0.32 runtime integration

MR 2 is stacked from the MR 1 head on an independent branch. During review its
base is the MR 1 branch; after MR 1 lands, its base moves to refreshed `main`.
It audits the validated
`hcu-vllm-adapter` `compat/vllm-v0.32.0` branch against official vLLM v0.32.0.
Official behavior remains authoritative; the adapter imports only HCU gaps
that are still required.

The vLLM host path uses vLLM's official platform/general plugin entry points,
not eager imports from `sitecustomize`. It adds the narrow HCU platform
identity, v0.32 release compatibility gate, process-local idempotent lifecycle,
and the DTK runtime-library path guard. Optimization providers and legacy DAS
features classified as plugin capabilities remain outside this repository.

The vLLM bridge must not be activated merely because sglang imports
`vllm._custom_ops`. Conversely, a vLLM launch must not execute the sglang bridge.

## Error handling and observability

Importing `sitecustomize` must never make an unsupported environment unusable.
Absent hosts and non-HIP interpreters return silently. Once a selected host is
present, individual legacy bridge failures retain their existing isolated
warning behavior. Explicitly requested but invalid host values fail loudly
with one actionable message.

The common environment summary reports the configured public host choice and
its source, and bootstrap emits one line with the selected active host and
detection source. The existing runtime registry remains the source of truth
for installed sglang bindings.

## Verification

MR 1 requires:

- static assertions that `sitecustomize.py` is the two-line delegator;
- pure host-selection tests, including both-packages-present and spawned-child
  inheritance;
- non-HIP and no-host subprocess no-op tests;
- sglang install-order and per-bridge exception-isolation regression tests;
- packaging tests proving the new package and existing subpackages enter the
  wheel;
- L0, adapter tests, CPU contracts, and an HCU sglang cookbook smoke before
  merge.

MR 2 adds vLLM plugin discovery, v0.32 compatibility, clean-process
idempotency, official-source contract tests, Qwen3-8B TP=2 serving, and the
existing fixed HumanEval accuracy lane before merge.

The two pre-existing L0 failures at base `f4b2c29` (`HCU_MOE_ALIGN_FUSED`
documentation and the `moe` catalog key) are recorded as baseline debt and are
not silently folded into either migration.
