# Flash publication and process-local health

vLLM owns attention selection and configuration validation. Adapter requires a
unique ACTIVE `hcu-flash-attention` report and its exact current-process health
binding, then delegates to `RocmPlatform`. No Plugin class path or backend table
is maintained here. Default Triton and official MLA/Sparse routing remain intact.

`ExtensionSpec.check_health` is an optional final API1 field. Before activation,
it must be synchronous and `inspect.signature(checker).bind()` must succeed.
Optional parameters are allowed. A checker returns None and must not re-enter
host lifecycle APIs. Its binding uses the exact ACTIVE report object and PID;
replacement, duplicate and failed reports cannot borrow an earlier binding.
Failure in the post-activation binding sink is commit-fatal, with one FAILED
report for that owner.

Lock order is host snapshot, unlocked owner callback, host verification. The
inflight counter covers all validation, callback, verification and failure
publication. A fixed failure sentinel is stored before diagnostic formatting;
formatting failure cannot erase refusal. Fork clears successful reports and
bindings, while incomplete or failed runtime/health state remains refused.
All selector routes check the runtime latch. Public registry writes have no
CAS: these checks do not promise atomicity against arbitrary concurrent writers.

The Plugin prepares predictable dependencies before public FLASH_ATTN
registration, refuses foreign overrides, and exposes its own health callback.
Registry drift must fail through the actual official selector, even if the
Plugin class would no longer be selected. No RMS, W4, FLA or model-load hook is
required by this independent owner.

Dependencies are published PR21 plus corrected PR23 (both based on PR17), with
the paired Plugin PR326 public-registry delta. CPU acceptance covers extension
ABI/lifecycle, bounded actual fork, and the actual target public selector for
clear/foreign drift, configuration refusal, optional/required failure, spawn and
unpickle. CPU evidence is not a numerical or full-model claim. Run the focused
tests in `tests/adapter/test_vllm_extension_health.py`,
`test_vllm_commit_fatal.py`, `test_vllm_flash_capability.py`, and the paired
Plugin `tests/vllm032/test_flash_public_selector.py` in isolated environments.
