# HCU vLLM runtime admission

HCU runtime initialization now constructs the selected vLLM release's official
`CudaRTLibrary` before extension registration. The constructor
loads the runtime and binds the symbols and ctypes declarations that vLLM
actually uses. Missing, unloadable or incompatible symbol sets raise a
`VllmCompatibilityError`, which the existing process-local runtime action
latches. No extension is activated after this failure.

Keep official resolution: an already loaded eligible HIP library wins over the
configured fallback. `configure_vllm_hip_runtime_path` still preserves an
operator's explicit spelling, including aliases, and derives an existing DTK
path only when none was supplied. Admission passes no explicit library argument
and does not copy the framework's symbol inventory or impose a wheel/build
allowlist. Non-HCU hosts skip this HCU runtime path.

The constructor binds functions; this change calls no HIP allocation, device
selection, synchronization or IPC operation. Successful binding is a prerequisite,
not proof of every native calling convention, kernel, IPC or model behavior.
Those remain separate native/model gates. It does not add generic activation
rollback or change optional capability ownership.

Tests cover startup ordering/latching, non-HCU no-op, preserved aliases,
actual target constructor failures for missing files, invalid ELF and a loadable
non-HIP system library, plus official loaded-library precedence. CPU controls
use explicit platform/lookup metadata doubles; the precedence control also
doubles CDLL binding and forbids invoking any bound function. Installed and
actual HCU CLI evidence is separate from source assertions.

This branch depends on PR17 and contains no other feature-branch changes.
