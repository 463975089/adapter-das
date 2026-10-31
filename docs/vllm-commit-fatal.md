# Fatal extension commit failures

`hcu_adapter.extensions.ExtensionCommitError(message, reports=())` is the
public signal that activation published state and cannot safely continue in
this process. An owner raises it after its commit boundary. The host preserves
actual earlier ACTIVE/current FAILED reports and stops later activation, even
for an optional owner. This signals failure; it performs no rollback.

`is_fatal_extension_error(error)` follows cause/context, including suppressed
context, cycles and available native exception groups. It recognizes only the
public signal, without message/OOM heuristics or a new Python minimum. Ordinary
third-party clean optional failures keep their existing FAILED/continue
behavior. Unmarked hidden third-party side effects cannot be inferred.

The existing HCU runtime stores fatal reports and latches the startup error,
like a required startup failure. Public `hcu_adapter.bridges.vllm.
require_runtime_healthy()` is a read-only query of this failure. Unknown or
healthy state is allowed; it is not evidence activation finished. Queries
never activate, repair or import a backend. Reports remain diagnostic and
changing them cannot erase the failure latch.

Diagnostic rendering cannot replace a startup failure. `run_once` writes a
fixed refusal before evaluating an exception's string; a formatting failure,
including a formatting `BaseException`, retains that refusal before the running
marker is cleared. The callback's existing `Exception` catch boundary is
unchanged. Discovery/activation classify the original error before rendering
it once under a guard, preserving the actual FAILED report and typed fatal
signal even for wrapped errors with a broken or stateful `__str__`. An ordinary
unmarked optional error still remains optional when only its diagnostic string
is broken. No new recovery, reset, rollback or health-check interface is added.

Every HCU attention selection queries health before backend/config inspection.
Thus catching a failed boot does not permit Flash through a stale ACTIVE
report, nor bypass failure with default TRITON/MLA/sparse. Healthy defaults,
public method signatures and Flash unique-owner/config checks are unchanged.

Only the existing `vllm-runtime` action tightens fork semantics. Its failed or
in-progress state is refused in a forked child because registrations may have
been inherited. Reports and locks reset; failure does not heal. Healthy runtime
success resets/reinitializes as before; unrelated actions retain their original
reset semantics. A same-thread health query during activation is reentrant and
read-only. A selector in another thread waits for the runtime gate, then sees
its final failure. A fresh spawn/exec has independent state.

Tests exercise actual fork with child alarms and bounded parent waits, actual
public host/report/runtime paths, concurrency and CPU metadata doubles. They
do not establish GPU kernels, model accuracy, engine spawn or AOT reload.
Arbitrary direct kernel execution after ignoring boot errors is outside this
activation boundary. There is no production reset/unpoison or generic rollback
API, and no exact Plugin wheel admission rule.


## Public dependencies and compatibility

This independent candidate is based only on published PR17 (`35f8f056`).
Its existing extension host already provides contexts, ordered reports,
`RequiredExtensionError` and process-local runtime actions. The fatal/health
implementation calls no model-load interface and requires neither PR19 nor
PR20. It introduces no numerical operator, model/loader/runner, FP8 fallback,
Flash owner/registry policy or HIP-admission changes.

PR20's environment-required startup policy is a separate composable change.
Tests using a required public `ExtensionContext` exercise the existing PR17
API, not PR20 env parsing. Actual metadata/general-plugin/spawn fixtures here
exercise clean optional, explicit fatal and wrapped fatal behavior without
adding environment-required production policy. W4 independently needs PR19's
post-load/mutation hooks; that consumer dependency does not make this ABI
require the W4 lifecycle mechanism. No exact wheel/vendor/model/SHA admission
or private registry protocol is introduced.

The public error constructor accepts `message` and optional `reports`; it keeps
reports as an immutable tuple. The public classifier accepts an exception and
the runtime-health query takes no required arguments. Callers may validate this
minimum ABI and accept future compatible optional parameters. Neither API grants
permission to resume an unsafe activation or to inspect/reset private registries.

## CPU evidence scope

`test_vllm_commit_fatal.py` covers public clean/fatal behavior, wrappers and
exception groups, retries, caught-error selectors, concurrency and actual fork.
`test_vllm_commit_fatal_discovery.py` builds and installs a tiny test provider to
exercise real distribution entry points, official `load_general_plugins`,
clean/fatal/wrapped behavior and actual spawn/config unpickle before discovery.
The fixture also covers clean/fatal/wrapped/factory failures with unrenderable
errors and required-runtime action failures through the real official loader,
both directly and after spawn/unpickle. Unit contracts cover parent/child
refusal and same-process retry after diagnostic failures, including fork during
string conversion. These checks do not transfer old wheel results to a new one.
Only HCU admission/platform selection are CPU doubles; no EngineCore constructor,
model or GPU kernel runs. The fixture refuses unrelated optimization providers.
Fixture imports neither patch runtime functions nor write evidence. A normal
quant-config `__setstate__` records actual spawn restoration before discovery;
an independent import-purity case checks missing test env/log paths are harmless.
The official loader's once flag may return after a caller catches startup error;
this does not clear Adapter's runtime failure, permit selectors, or retry owners.

Run the affected tests with `HCU_ADAPTER_HOST=none`, empty GPU visibility and
`VLLM_TARGET_ROOT` pointing to the exact audited vLLM checkout for source-byte
contracts when selected. Source validation sets an explicit Adapter `PYTHONPATH`; installed
validation clears it, uses an outside-repo cwd, verifies wheel RECORD/origins in
parent and child, and replaces source-cwd-only import tests with installed-path
companions. Exact commands, selection and artifact hashes accompany the build.
CPU results establish lifecycle/discovery contracts, not GPU/model/Graph or
clean shutdown. No timeout, quality gate or production reset is changed.
Shared bootstrap dispatch is checked for both SGLang and vLLM with CPU bridge
doubles, alongside source structure/import isolation. This is not a SGLang
serving or GPU regression result. Operator-tier, vendor-switch and numerical
quality promotion rules are unaffected: this ABI adds no operator implementation.
