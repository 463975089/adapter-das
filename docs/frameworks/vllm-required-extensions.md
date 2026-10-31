# vLLM startup requirements

Set these variables before launching vLLM. The Adapter passes them through its
official general-plugin callback in the frontend and each worker:

```bash
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=das_hcu
export HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=aiter-int8-moe
vllm serve /path/to/model
```

Both variables accept comma-separated, case-sensitive names. Whitespace around
names and repeated names are allowed. An unset or blank variable requests
nothing; empty elements or whitespace inside a name fail startup before kernel
registration. Set requirements before the first callback; they are startup
configuration, not live enable/disable switches. Fork/spawn workers recheck
their inherited requirements.

Both keys are registered in `hcu_adapter.hcu_env.REGISTRY` and documented in
the [environment manual](../env-vars.md). Behavioral reads use its canonical
resolver. Under normal `HCU_ADAPTER_HOST=vllm` (or auto-selected vLLM),
`bootstrap.install` calls `bridges.common.install`, which emits the shared
summary once before the host bridge; explicit values carry `(explicit)` and
empty defaults use the existing all-defaults line. The empty default requests nothing. There is no hardware/provider
auto rule: a requirement asserts what must already activate successfully and
does not itself enable or select an implementation.

`HCU_ADAPTER_HOST=none` disables that automatic bootstrap/common chain. Calling
the official general-plugin loader directly still resolves and enforces these
requirements, but does **not** emit a common summary or an additional per-value
log. Use `hcu_env.resolve(name)` to inspect `(value, source)` in diagnostics; do
not infer a missing requirement check from a missing common-summary line.
CPU regressions exercise both the actual bootstrap/common chain (only hardware
admission/topology/JIT providers are doubles) and HOSTnone + the actual runtime
callback, including default/explicit/blank/malformed inputs, source resolution,
run-once behavior and refusal before runtime registration. These are not real
GPU, CLI/model or kernel-evidence claims.

Extension requirements use the installed `hcu_adapter.extensions` **entry-point
name** (`das_hcu` above). They do not use the spec's declared name (`das-hcu`).
Capability requirements use exact `ExtensionSpec.capabilities` strings. Missing
or failed requirements stop startup, and failure is latched in that process.
`hcu_adapter.bridges.vllm.extension_reports()` retains the complete reports,
including unrelated optional failures. A fresh process is required after a
latched startup failure.

The host rejects duplicate entry-point names, declared extension names, and
capability owners. Two otherwise valid specs advertising the same capability
both receive failed reports **before either activation callback runs**;
priority does not silently choose one. Unrelated specs may still activate.
An incompatible API declaration does not reserve capabilities from a compatible
spec. Extension factories must remain free of registration side effects.

An active capability report means its owner activated successfully. It does not
prove that a particular model selected it, that native kernels ran, or that
accuracy/performance passed. Those claims require model/provider evidence.
The host does not roll back arbitrary side effects inside a failing activation
callback; an extension must make its own installation atomic or fail before
mutation. Optional activation failures remain reports when nothing requires
them.

Portable coverage exercises the real callback and extension host with injected
discovery/platform dependencies, missing/ambiguous requirements, malformed
configuration, failure latching, and a real fork. It is separate from installed
wheel and actual vLLM startup/model validation.
