# vLLM model runner factory seam

The exact Adapter Worker keeps its model-load and mutation hooks. Its upstream `_make_model_runner` override delegates runner selection to `hcu_adapter.bridges.vllm_model_runner`; numerical and model-specific execution stays in a plugin.

Register at extension activation with `register_model_runner_factory(owner, matches=predicate, create=factory)`. The predicate receives the resolved VllmConfig. A selected factory receives `(vllm_config, device)` and returns the runner. With no match, the original bound Worker factory runs unchanged. Multiple matches, conflicting ownership, a selected factory returning None, or registration after selection are rejected. Repeated registration of the identical callbacks is idempotent. Active registrations cannot be inherited through fork; use spawn. An empty registry preserves the official default in a forked process.

The extension retains the returned registration and calls `require_model_runner_factory(registration)` from its health checker. Runner selection freezes the registry. Callbacks are invoked outside the registry lock; they must not install a second worker or bypass the Adapter load lifecycle. A V2 runner subclass can use upstream `pcp_manager_cls` for an explicit out-of-tree PCP implementation. Capability validation, model registration, real attention/cache operations and tests remain the plugin owner’s responsibility.

This adds a startup selection mechanism, not a default model runner or a capability claim for upstream backends. Existing configurations without a matching explicit extension continue through the official runner.

Fork detection runs before acquiring the registry lock. A child rebuilds its lock but refuses inherited active registrations, including forks inside predicates or factories. Each callback is followed by an ownership/process check, including false predicates and successful factory returns.

Idempotent registration with the same owner/callbacks returns the original canonical registration object, both before and after freeze. The returned handle must pass `require_model_runner_factory`; value equality is not a substitute for ownership identity.
