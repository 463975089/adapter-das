# vLLM 0.32 Extension Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned optional-extension host to the adapter's official vLLM 0.32 runtime callback.

**Architecture:** A dependency-light `hcu_adapter.extensions` module owns discovery, validation, ordering, activation, and reports. `bridges/vllm.py` invokes it only after baseline runtime checks and stores reports in process-local state.

**Tech Stack:** Python 3.11+, `importlib.metadata` entry points, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-10-28-vllm032-extension-host-design.md`

## Global Constraints

- Adapter remains the sole owner of `vllm.platform_plugins` and `vllm.general_plugins`.
- Extension discovery group is exactly `hcu_adapter.extensions`; API version is exactly `1`.
- Importing `hcu_adapter.extensions` must not import vLLM, Torch, or any optimization package.
- Do not modify `sitecustomize.py`, `hcu_adapter/bootstrap.py`, or `hcu_adapter/bridges/sglang.py`.
- Optional failures are reports; explicitly required extension/capability failures stop startup.

## Review Focus

- Duplicate entry-point and declared names must fail before activation.
- A forked child must not inherit extension reports or activation state.
- Empty injected entry points must not trigger metadata discovery.
- Extension activation must happen after platform validation and DTK configuration.
- Malformed and incompatible optional extensions must not break baseline vLLM startup.

---

### Task 1: Extension API v1

**Files:**
- Create: `hcu_adapter/extensions.py`
- Create: `tests/adapter/test_vllm_extensions.py`

**Interfaces:**
- Produces: `EXTENSION_API_VERSION`, `ExtensionContext`, `ExtensionSpec`, `ExtensionReport`, `ExtensionStatus`, `RequiredExtensionError`, and `activate_extensions(context, entry_points=None)`.

- [ ] **Step 1: Write failing portable tests for immutable types, deterministic priority ordering, once-only loading, duplicate names, incompatible API, optional failure, required extension/capability failure, explicit empty discovery, and clean import isolation.**
- [ ] **Step 2: Run `HCU_ADAPTER_HOST=none python3 -m pytest -q tests/adapter/test_vllm_extensions.py` and verify failures are caused by the absent module.**
- [ ] **Step 3: Implement the dependency-light API and deterministic activation.**
- [ ] **Step 4: Run the focused test and verify it passes.**
- [ ] **Step 5: Commit as `vllm: add versioned optimization extension host`.**

### Task 2: Runtime activation and report state

**Files:**
- Modify: `hcu_adapter/bridges/vllm.py`
- Modify: `tests/adapter/test_vllm_bridge.py`
- Modify: `tests/adapter/test_packaging.py`

**Interfaces:**
- Consumes: Task 1 `ExtensionContext`, `ExtensionReport`, `RequiredExtensionError`, and `activate_extensions`.
- Produces: process-local `extension_reports()` and runtime activation after baseline guards.

- [ ] **Step 1: Write failing tests for ordered runtime activation, optional report retention, required failure latching, fork reset, adapter version lookup, and wheel inclusion.**
- [ ] **Step 2: Run the focused bridge/packaging tests and verify the new assertions fail.**
- [ ] **Step 3: Extend process-local state with reports and call `activate_extensions()` after version/platform/DTK checks.**
- [ ] **Step 4: Run focused tests and the official vLLM v0.32 source/installed-wheel contracts.**
- [ ] **Step 5: Commit as `vllm: activate optional extensions after runtime guards`.**

### Task 3: Documentation and full adapter gates

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/board.md`

**Interfaces:**
- Consumes: Tasks 1-2 runtime boundary.
- Produces: documented adapter/plugin ownership and verification evidence.

- [ ] **Step 1: Document the extension group, lifecycle order, failure policy, and separate plugin ownership.**
- [ ] **Step 2: Run full CPU/adapter and L0 gates, recording pre-existing failures separately.**
- [ ] **Step 3: Verify the MR1 structural boundary remains unchanged.**
- [ ] **Step 4: Commit as `docs: record vllm extension ownership boundary`.**
