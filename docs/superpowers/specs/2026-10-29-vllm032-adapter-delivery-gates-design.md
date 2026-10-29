# vLLM 0.32 Adapter Delivery Gates Design

## Intent

Keep `adapter-das` as the single HCU platform/runtime owner for official vLLM
0.32, while restoring a green adapter test baseline after the DeepInfer host was
added. Document the deployable component boundary so future changes do not move
platform ownership back into `vllm-plugin-das` or `sitecustomize.py`.

## Goals

- Treat `deepinfer` as a first-class bootstrap host in the portable bootstrap
  contracts without changing the existing runtime selection policy.
- Preserve Adapter ownership of `vllm.platform_plugins` and
  `vllm.general_plugins`.
- Preserve Plugin ownership of optional optimizations through the
  `hcu_adapter.extensions` entry-point group.
- Add one operator-facing vLLM 0.32 deployment and responsibility guide.
- Correct stale documentation that still places the vLLM custom-op ABI bridge
  in `sitecustomize.py`.

## Non-goals

- No kernel, catalog, environment-variable, or kill-switch changes.
- No changes to the exact vLLM 0.32 release gate or validated HCU architecture
  list.
- No Plugin repository changes and no compatibility-lock update.
- No DeepInfer operator validation; that requires a separate four-quality-
  dimension workstream.
- No change to the documented legacy host-arbitration behavior when multiple
  frameworks are installed and argv provides no signal.

## Constraints

- `sitecustomize.py` remains the three-line bootstrap entry.
- Adapter remains the only owner of the two official vLLM plugin groups.
- Optional DAS optimizations enter only through `hcu_adapter.extensions` API v1.
- Official vLLM behavior is reused when available; this change adds no duplicate
  implementation.
- Tests under `tests/adapter/` remain pytest-collectable and side-effect free.

## Approaches

### Selected: focused delivery-gate hardening

Update the stale bootstrap contracts, add DeepInfer cases to the existing host
matrix, and add a vLLM 0.32 deployment guide grounded in the already-merged
runtime bridge. This is the smallest reviewable change and restores the main
adapter gate without perturbing validated runtime code.

### Rejected: rewrite the vLLM bridge

The vLLM 0.32 platform/runtime and Extension API are already merged and their
focused tests pass. Rewriting them would duplicate working behavior without a
new failing contract.

### Rejected: combine DeepInfer quality closure

DeepInfer's operator parity, graph replay, scenario, vendor, and model gates are
independent and much larger. Combining them would obscure the vLLM delivery
change and make review and rollback harder.

## Resulting structure

```text
official vLLM 0.32
  -> vllm.platform_plugins / vllm.general_plugins
  -> adapter-das (platform, release/hardware/runtime guards)
  -> hcu_adapter.extensions API v1
  -> optional vllm-plugin-das optimizations
```

The baseline `vLLM + Adapter` stack must remain usable without the optional
Plugin. Plugin activation occurs only after Adapter release, platform, and DTK
runtime guards.

## Acceptance

- `python ci/l0_gate.py` exits zero.
- `pytest -q tests/adapter` has no failure from treating `deepinfer` as invalid.
- Focused vLLM bridge, Extension API, and packaging contracts pass.
- CPU contracts pass.
- Documentation identifies the exact entry-point ownership and startup order.
