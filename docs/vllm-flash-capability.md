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
Installed/actual CLI results for this exact pair are recorded below;
larger integration-candidate GPU results are not review-artifact validation.
An unrelated activation's partial-commit failure may leave dirty state; this
own-capability gate makes no whole-process health or rollback claim.


Exact clean pair installed validation completed: Plugin55 passed;
Adapter48 passed/one source-layout test deselected. The initial absolute
node-id deselection did not match pytest's relative node-id and retained that
source-only failure; the final runner used the exact test name selection,
without production changes. Source49 includes this assertion.

Independent wheels bind12 Plugin/194 Adapter Python files and every hashed
RECORD row;2649 target Python/native files plus parent/child native module
origins were checked. Plugin wheel SHA256
`7e6052468f71d543724309fb887510aa68fb98d046de4704b3dfa7fc295ba5e9`;
Adapter wheel SHA256
`d5c3abd89bd11751a9079ecf5a5a52e33f9bdfb846b70ebde5b0576c110c77d6`.

The actual installed, unmodified official CLI with real HCU/vendor imports
passed. Its process-local reports and selector accepted explicit auto/E5M2
Flash, refused invalid block128, and returned official Triton by default.
No provider/platform/discovery doubles or GPU/model claims are involved.
The PR17 baseline does not include the separate required-startup env gate;
this evidence checks actual reports and selector behavior directly.
