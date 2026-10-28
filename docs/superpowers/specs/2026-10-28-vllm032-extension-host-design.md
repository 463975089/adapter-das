# vLLM 0.32 Extension Host Design

## Intent

Let the multi-host adapter own vLLM 0.32 platform/runtime lifecycle while
loading optional optimization packages through a stable, versioned boundary.
This connects the existing DAS optimization wheel without moving its kernels
or operator implementations into the adapter repository.

## Goals

- Add Extension API v1 under `hcu_adapter.extensions`.
- Discover only the `hcu_adapter.extensions` entry-point group.
- Activate extensions after vLLM version, selected-platform, and DTK runtime
  checks have passed.
- Keep optional extension failures observable but non-fatal; explicitly
  required extensions or capabilities fail closed.
- Preserve process-local, run-once runtime initialization.

## Non-goals and anti-goals

- Do not let an optimization package own `vllm.platform_plugins` or
  `vllm.general_plugins`.
- Do not put optimization math in `bridges/vllm.py`.
- Do not import vLLM, Torch, or optimization packages while importing the
  extension protocol module.
- Do not change `sitecustomize.py`, bootstrap dispatch, or the sglang bridge.
- Do not claim model production support from portable contract tests.

## Approaches

1. **Versioned adapter extension group — selected.** The adapter exposes
   immutable context/spec/report types and deterministic entry-point
   activation. This preserves single ownership of vLLM lifecycle and lets
   optimization wheels be installed or removed independently.
2. Let every optimization wheel register a vLLM general plugin. Rejected
   because allow-listing and ordering can activate only half of the HCU
   lifecycle and creates competing runtime owners.
3. Import the DAS package directly from `bridges/vllm.py`. Rejected because it
   couples the baseline adapter to one optional provider and makes uninstall
   and rollback ambiguous.

## Interface

`hcu_adapter.extensions` defines API version `1`, `ExtensionContext`,
`ExtensionSpec`, `ExtensionReport`, `ExtensionStatus`,
`RequiredExtensionError`, and `activate_extensions()`. Discovery uses entry
point group `hcu_adapter.extensions`. Specs activate by descending priority,
then declared name; reports are returned in deterministic entry-point order.
Duplicate entry-point names, duplicate declared extension names, malformed
specs, incompatible API versions, and activation errors become failed reports.

The vLLM runtime callback creates a context with the installed `sgl-kernel`
distribution version and accepted vLLM version. It records the latest
process-local report tuple. Optional failures do not stop vLLM. A
`RequiredExtensionError` is latched by the existing runtime action and stops
startup.

## Testing and acceptance

- Portable tests cover deterministic ordering, load/activation once,
  duplicates, incompatible API versions, optional failure, required failure,
  and import isolation.
- Runtime tests prove extension activation occurs after version/platform/DTK
  checks, once per PID, and reports are retained.
- Wheel tests prove `hcu_adapter/extensions.py` is shipped.
- Existing vLLM bridge, complete CPU/adapter, official v0.32 source contract,
  and installed-wheel contract remain green.

## Cross-repository contract

The DAS plugin changes its only extension entry point from
`hcu_vllm_adapter.extensions` to `hcu_adapter.extensions`, imports API types
from that namespace, depends on `sgl-kernel>=0.1.0`, and continues to package
no vLLM platform/general entry points.
