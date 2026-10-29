# vLLM 0.32 Adapter Delivery Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the portable Adapter gate after DeepInfer host integration and document the deployable vLLM 0.32/Adapter/Plugin ownership model.

**Architecture:** Keep the already-merged vLLM bridge unchanged. Extend the existing bootstrap contract matrix to cover DeepInfer and correct the stale invalid-host assertion, then add a deployment guide and update the shadow-matrix fact source.

**Tech Stack:** Python 3.12, pytest, setuptools entry points, Markdown, GitHub fork pull-request workflow

**Spec:** `docs/superpowers/specs/2026-10-29-vllm032-adapter-delivery-gates-design.md`

## Global Constraints

- `sitecustomize.py` remains a three-line bootstrap entry.
- Adapter exclusively owns `vllm.platform_plugins` and `vllm.general_plugins`.
- Plugin optimizations enter only through `hcu_adapter.extensions` API v1.
- Do not change kernels, catalog, environment variables, kill switches, or runtime host arbitration.
- Keep all new tests pytest-collectable with no module-level side effects.

## Review Focus

- Explicit `deepinfer` selection must override an inherited different host.
- An actually unknown explicit host must still raise a user-facing `HCU_ADAPTER_HOST` error.
- A DeepInfer selection must not run the early common bridge because its runtime installs common after device-mask alignment.
- vLLM platform/general entry-point ownership must remain Adapter-only.
- Optional Plugin absence must remain a supported baseline documented by the deployment guide.

---

### Task 1: Restore multi-host bootstrap contracts

**Files:**
- Modify: `tests/adapter/test_bootstrap.py`

**Interfaces:**
- Consumes: `bootstrap.detect_host(...) -> str | None` and `bootstrap.install() -> None`
- Produces: portable regression coverage for `deepinfer` as a valid explicit, inherited, argv, availability, and installation host

- [ ] **Step 1: Correct and extend the failing host-selection tests**

Update the existing parameter matrices to include `deepinfer`, replace the stale
invalid value with `invalid-host`, add DeepInfer argv cases, and add an install
test asserting only the DeepInfer bridge runs.

- [ ] **Step 2: Run the focused bootstrap contracts**

Run: `HCU_ADAPTER_HOST=none pytest -q tests/adapter/test_bootstrap.py`

Expected: all bootstrap contracts pass; before the test correction the baseline
failure is `test_invalid_explicit_host_is_rejected` because `deepinfer` is valid.

- [ ] **Step 3: Run the full portable adapter contracts**

Run: `HCU_ADAPTER_HOST=none pytest -q tests/adapter`

Expected: zero failures; hardware-dependent cases may skip.

- [ ] **Step 4: Commit**

```bash
git add tests/adapter/test_bootstrap.py
git commit -m "test: recognize DeepInfer in host bootstrap contracts"
```

### Task 2: Document the vLLM 0.32 delivery boundary

**Files:**
- Create: `docs/frameworks/vllm032.md`
- Modify: `skills/hcu-adapter-dev/references/shadow-matrix.md`
- Modify: `skills/hcu-adapter-dev/SKILL.md`

**Interfaces:**
- Consumes: Adapter entry points in `pyproject.toml`, Extension API v1 in `hcu_adapter/extensions.py`, vLLM guards in `hcu_adapter/bridges/vllm.py`
- Produces: one operator guide describing ownership, install/startup order, fallback semantics, and verification commands

- [ ] **Step 1: Add the vLLM 0.32 deployment guide**

Document the three-component structure, exact ownership of official vLLM entry
points, `hcu_adapter.extensions` activation order, baseline-without-Plugin
semantics, pinned compatibility bundle fields, and portable/HCU verification
commands.

- [ ] **Step 2: Correct the shadow-matrix and skill structure facts**

Move the documented `vllm._custom_ops` bridge location from `sitecustomize.py`
to `hcu_adapter/bridges/vllm.py`, and list `deepinfer` in the bootstrap host set.

- [ ] **Step 3: Verify documentation and entry-point contracts**

Run: `HCU_ADAPTER_HOST=none pytest -q tests/adapter/test_vllm_bridge.py tests/adapter/test_vllm_extensions.py tests/adapter/test_packaging.py`

Expected: all focused vLLM and packaging contracts pass.

- [ ] **Step 4: Run L0 and CPU contracts**

Run: `HCU_ADAPTER_HOST=none python ci/l0_gate.py && HCU_ADAPTER_HOST=none pytest -q tests/test_ops_cpu.py`

Expected: L0 GREEN with only known soft warnings and CPU contracts pass.

- [ ] **Step 5: Commit**

```bash
git add docs/frameworks/vllm032.md skills/hcu-adapter-dev/references/shadow-matrix.md skills/hcu-adapter-dev/SKILL.md docs/superpowers/specs/2026-10-29-vllm032-adapter-delivery-gates-design.md docs/superpowers/plans/2026-10-29-vllm032-adapter-delivery-gates.md
git commit -m "docs: define the vLLM 0.32 adapter delivery boundary"
```

## Final verification

- [ ] Run `HCU_ADAPTER_HOST=none python ci/l0_gate.py`.
- [ ] Run `HCU_ADAPTER_HOST=none pytest -q tests/test_ops_cpu.py tests/adapter`.
- [ ] Inspect `git diff upstream/main...HEAD` for scope and secrets.
- [ ] Push `feat/vllm032-adapter-delivery-gates` to the fork.
- [ ] Open a pull request from `alexanderbin123:feat/vllm032-adapter-delivery-gates` to `HYGON-AI:main`.
