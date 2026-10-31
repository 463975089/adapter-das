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
