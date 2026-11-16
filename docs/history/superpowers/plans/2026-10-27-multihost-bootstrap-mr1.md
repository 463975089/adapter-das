# Multi-host Bootstrap MR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3191-line automatic entry point with a two-line dispatcher while preserving the existing sglang bridge behavior.

**Architecture:** A process-idempotent bootstrap selects one host, runs framework-neutral guards first, and delegates to focused bridge modules. MR 1 mechanically relocates the sglang chain and extracts its existing vLLM RMSNorm helper; it does not add the vLLM platform plugin.

**Tech Stack:** Python 3.9+, setuptools, pytest, importlib, existing `sgl_kernel` registry and environment resolver.

**Spec:** `docs/superpowers/specs/2026-10-27-multihost-bootstrap-design.md`

## Global Constraints

- Preserve the current sglang install order and per-bridge exception isolation.
- Keep `sitecustomize.py` to the two-line delegator requested by the user.
- Do not introduce numerical code into bootstrap or common bridge modules.
- Do not activate the vLLM platform plugin in MR 1.
- Non-HIP and no-host processes are silent no-ops.
- Normal deployments require no manually configured environment variables.
- Do not fix the two unrelated base-revision L0 failures in this MR.

## Review Focus

- Both sglang and vLLM installed: launch intent or inherited selection chooses exactly one host; covered by Task 1 selection tests.
- Spawned sglang worker with generic multiprocessing argv: inherited selection retains sglang; covered by Task 1 propagation test.
- Generic HIP Python without either host: no environment mutation or warnings; covered by Task 2 subprocess test.
- A common guard fails: host bridge still follows the documented isolation policy; covered by Task 2 call-order test.
- Built wheel omits nested packages: wheel inspection must include `hcu_adapter.bridges` and `sgl_kernel._adapters`; covered by Task 4 packaging test.

---

### Task 1: Host selection contract

**Files:**
- Create: `hcu_adapter/__init__.py`
- Create: `hcu_adapter/bootstrap.py`
- Test: `tests/adapter/test_bootstrap.py`
- Modify: `sgl_kernel/_adapters/hcu_env.py`
- Modify: `docs/env-vars.md`

**Interfaces:**
- Produces: `detect_host(argv, loaded_modules, available, environment) -> str | None`.
- Produces: public `HCU_ADAPTER_HOST=auto|sglang|vllm|none` resolution and private child propagation.

- [ ] **Step 1: Write failing selection tests** for explicit choice, inherited choice, sglang/vLLM argv, one-package availability, both-package legacy fallback, and neither-package no-op.
- [ ] **Step 2: Run `python3 -m pytest -q tests/adapter/test_bootstrap.py` and verify failures are caused by the missing package/API.**
- [ ] **Step 3: Implement the pure selector and child-propagation helper; register and document the public host key without wiring automatic startup yet.**
- [ ] **Step 4: Rerun the Task 1 tests and require all pass.**
- [ ] **Step 5: Commit as `bootstrap: add deterministic multi-host selection`.**

### Task 2: Common guards and thin automatic entry point

**Files:**
- Create: `hcu_adapter/bridges/__init__.py`
- Create: `hcu_adapter/bridges/common.py`
- Create: `hcu_adapter/bridges/deepinfer.py`
- Create: `hcu_adapter/bridges/sglang.py` by mechanically relocating the legacy entry-point body.
- Modify: `sitecustomize.py`
- Modify: `tests/adapter/test_sitecustomize_sequence.py`
- Test: `tests/adapter/test_bootstrap.py`

**Interfaces:**
- Consumes: Task 1 `install()` and host selection.
- Produces: `common.install() -> None`, `sglang.install() -> None`, process-idempotent `bootstrap.install() -> None`, and the two-line `sitecustomize.py` delegator.

- [ ] **Step 1: Replace the old structural expectations with failing tests for the exact delegator, relocated sequence, common-before-host order, idempotency, non-HIP no-op, and HIP/no-host no-op.**
- [ ] **Step 2: Run the focused tests and verify the old entry point fails the new assertions.**
- [ ] **Step 3: Mechanically wrap the legacy chain in `sglang.install()`, extract the three common guards, implement bootstrap orchestration and the side-effect-free DeepInfer seam, then reduce `sitecustomize.py` to two lines.**
- [ ] **Step 4: Run both adapter test files and require all applicable tests pass.**
- [ ] **Step 5: Commit as `bootstrap: move sglang chain behind thin sitecustomize`.**

### Task 3: Extract the existing vLLM custom-op bridge

**Files:**
- Create: `hcu_adapter/bridges/vllm.py`
- Modify: `hcu_adapter/bridges/sglang.py`
- Modify: `tests/adapter/test_sitecustomize_sequence.py`
- Test: `tests/adapter/test_vllm_bridge.py`

**Interfaces:**
- Consumes: Task 2 `sglang.install()`.
- Produces: `vllm.install_rms_norm_bridge() -> bool` preserving the existing sglang-facing custom-op bridge at its original sequence point.

- [ ] **Step 1: Add failing contract tests for both RMSNorm wrapper signatures, destination-passing behavior, and the unchanged location in the sglang sequence.**
- [ ] **Step 2: Run the focused tests and verify failure because bridge modules are absent.**
- [ ] **Step 3: Extract only the existing RMSNorm block to `bridges/vllm.py` and delegate from its original sequence point without enabling the vLLM platform path.**
- [ ] **Step 4: Run adapter, CPU layernorm, and registry tests; require no new failures or skips.**
- [ ] **Step 5: Commit as `bootstrap: isolate sglang-facing vllm custom ops`.**

### Task 4: Packaging and governance alignment

**Files:**
- Modify: `pyproject.toml`
- Modify: `ci/l0_gate.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/governance.md`
- Test: `tests/adapter/test_packaging.py`

**Interfaces:**
- Consumes: Tasks 1-3 package layout.
- Produces: a wheel containing all `hcu_adapter` and `sgl_kernel` packages and documentation matching runtime ownership.

- [ ] **Step 1: Add a failing wheel-content test for `hcu_adapter`, `hcu_adapter.bridges`, and `sgl_kernel._adapters`.**
- [ ] **Step 2: Build the wheel and verify the new test fails against current explicit package configuration.**
- [ ] **Step 3: Switch setuptools to package discovery, update L0 path ownership checks, and update architecture/governance documentation without changing operator behavior.**
- [ ] **Step 4: Rebuild and inspect the wheel; run packaging test, L0, adapter suite, and CPU contracts. Record the two known base L0 failures separately and require no new failures.**
- [ ] **Step 5: Commit as `packaging: ship multi-host adapter modules`.**

### Task 5: HCU smoke, MR evidence, and submission

**Files:**
- Modify: `docs/board.md`
- Modify: `docs/models.md` only if the cookbook smoke produces a new supported result.

**Interfaces:**
- Consumes: complete MR 1 branch.
- Produces: reviewable evidence and a GitHub MR targeting `HYGON-AI/adapter-das:main`.

- [ ] **Step 1: Run static syntax checks, focused tests, full CPU suite, wheel inspection, and compare L0 output with the recorded base revision.**
- [ ] **Step 2: On an owned/idle HCU container, run the existing sglang cookbook smoke with eager mode disabled only if the current validated recipe requires it; do not touch shared foreign processes.**
- [ ] **Step 3: Review the diff for mechanical-only sglang movement, tokens/secrets, unrelated files, and upstream-supported duplicate behavior.**
- [ ] **Step 4: Push `feat/bootstrap-multihost` to the user's fork and open an MR against `HYGON-AI/adapter-das:main`, listing baseline RED items and HCU evidence.**
- [ ] **Step 5: After MR 1 merges, create the MR 2 branch from refreshed upstream `main`; do not stack its diff on an unmerged fork branch.**
