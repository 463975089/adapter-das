# Explicit HCU Flash owner gate

Selecting FLASH_ATTN requires one ACTIVE `hcu-flash-attention` capability owner
in public process-local `extension_reports()`. The paired Plugin provides the
independent `das_hcu_flash` entry point. Missing, failed or duplicate owners
raise before backend import and configuration validation. Complete normal HCU
runtime activation first; backend selection does not activate extensions.

Reports are cleared after fork, so a child cannot inherit a parent's accepted
owner. Ordinary default TRITON and official MLA/sparse selection do not require
Flash. Existing device/configuration validation remains necessary after this
gate. An ACTIVE report is preparation evidence, not a kernel/model claim.

This review is based on Adapter PR17 `35f8f05`; the paired independent Plugin
review is based directly on merged MR321 `153e4d9`. It does not require the
separate FP8, lifecycle, required-startup or quantization changes.

Exact clean-review source tests:49 bridge/Flash CPU tests passed; paired
Plugin55 CPU tests passed on frozen target `a198423b7`. The new owner/fork
assertions previously exposed six expected failures on the original Adapter.
Installed/actual CLI checks for this exact pair remain separate pending gates;
larger integration-candidate GPU results are not review-artifact validation.
An unrelated activation's partial-commit failure may leave dirty state; this
own-capability gate makes no whole-process health or rollback claim.
