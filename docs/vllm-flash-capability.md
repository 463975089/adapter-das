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

Source tests: 78 passed on the frozen v0.32 target; the new owner/fork tests on
the original Adapter yield six expected failures and six passes. Installed,
actual CLI and GPU/model checks are separate pending gates. Runtime acceptance
uses public capability state, with no exact Plugin wheel/version admission.
