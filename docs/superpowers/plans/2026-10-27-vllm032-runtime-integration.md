# vLLM 0.32 Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum official vLLM v0.32 platform and general-plugin integration to the multi-host adapter without changing the sglang chain or importing plugin-only optimizations.

**Architecture:** `hcu_adapter.bridges.vllm` remains the single vLLM bridge and exports vLLM's official entry-point callbacks. The module exposes `HcuPlatform` lazily from that same file, so the prescribed bridge layout stays intact while sitecustomize and plugin discovery do not import vLLM prematurely.

**Tech Stack:** Python 3.11+, setuptools entry points, vLLM v0.32 plugin API, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-multihost-bootstrap-design.md`

## Global Constraints

- Do not modify `sitecustomize.py`, `hcu_adapter/bootstrap.py`, or the sglang installation chain.
- Target PEP 440 release tuple `0.32.0`; allow pre/dev/post/local suffixes and reject other epochs/releases.
- Use official `vllm.platform_plugins` and `vllm.general_plugins` entry points.
- Keep plugin-only optimizations outside this repository.
- Preserve the MR 1 sglang-facing `_custom_ops` RMSNorm bridge and do not activate it on the vLLM host path.

## Review Focus

- Non-HCU and ordinary ROCm hosts must remain no-ops without importing vLLM runtime modules.
- A forked child must not inherit the parent's successful or failed lifecycle latch.
- An explicit `VLLM_CUDART_SO_PATH` must never be overwritten.
- A changed vLLM platform class or unsupported release must fail closed before runtime work.
- Entry-point metadata must use one shared allow-list name so platform and general callbacks activate atomically.

---

### Task 1: Pin portable vLLM bridge contracts

**Files:**
- Modify: `tests/adapter/test_vllm_bridge.py`
- Modify: `tests/adapter/test_packaging.py`

**Interfaces:**
- Produces: executable contracts for v0.32 compatibility, HCU detection, DTK runtime path, process-local idempotency, and entry-point metadata.

- [x] **Step 1: Add failing tests for accepted/rejected release versions, HCU detection, explicit/derived runtime library paths, parent/child lifecycle behavior, and the two official entry points.**
- [x] **Step 2: Run the focused tests and verify they fail because the MR 2 APIs and metadata are absent.**
- [x] **Step 3: Commit as `test: pin vllm 0.32 runtime contracts`.**

### Task 2: Implement the official vLLM plugin seam

**Files:**
- Modify: `hcu_adapter/bridges/vllm.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Task 1 contracts.
- Produces: `hcu_platform_plugin() -> str | None`, `hcu_runtime_plugin() -> None`, and `HcuPlatform`.

- [x] **Step 1: Implement the v0.32 compatibility gate, lazy HCU detection, DTK runtime guard, and PID-local run-once latch in `bridges/vllm.py`.**
- [x] **Step 2: Expose the minimal `RocmPlatform` subclass lazily from `bridges/vllm.py`, delegating official behavior and selecting upstream Triton attention only when no backend is explicit.**
- [x] **Step 3: Register both official entry points under the shared `hcu_adapter` name.**
- [x] **Step 4: Run the focused tests and verify they pass.**
- [x] **Step 5: Commit as `vllm: add official v0.32 runtime seam`.**

### Task 3: Prove isolation and document the stacked MR

**Files:**
- Modify: `docs/board.md`
- Modify: `docs/architecture.md`
- Test: `tests/adapter/test_bootstrap.py`
- Test: `tests/adapter/test_sitecustomize_sequence.py`

**Interfaces:**
- Consumes: Task 2 entry points and MR 1 dispatch.
- Produces: documented ownership and regression evidence that the sglang/sitecustomize paths are unchanged.

- [x] **Step 1: Document vLLM callback ownership, lazy platform import, and the plugin optimization boundary.**
- [x] **Step 2: Run adapter/CPU/L0 gates and the official vLLM v0.32 source contract tests.**
- [x] **Step 3: Inspect the MR 1..MR 2 diff and verify `sitecustomize.py`, `bootstrap.py`, and `bridges/sglang.py` are unchanged.**
- [x] **Step 4: Commit as `docs: record vllm 0.32 runtime boundary`.**
