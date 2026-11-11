# Keep HIP metadata queries free of PyTorch device initialization

On the HCU CUDA facade, SGLang capacity detection entered
`get_nvgpu_memory_capacity`, failed to execute `nvidia-smi`, and fell back to
`torch.cuda.mem_get_info()`. That fallback initializes a PyTorch CUDA context
before the worker selects its device. Argument resolution also queried device
capability directly. These are shared platform paths, not GLM model operations.

The HIP path now queries `hipDeviceTotalMem` from the runtime already loaded by
Torch, takes the minimum total capacity across visible device ordinals, and
returns MiB. Non-HIP capacity detection retains its original implementation.
Audited early property/capability queries use HIP metadata; initialized workers
retain canonical Torch properties. Architecture lookup matches HIP PCI BDF to
SMI, rather than assuming identical enumeration order. Unknown early property
fields fail explicitly.

The same change removes CUDA allocation from vendor weak-reference discovery,
keeps resource trackers out of GPU bootstrap, disables adapter loading in
TVM-FFI build subprocesses, and defers device-dependent configuration logging.
The AST redirects are limited to the listed modules and produce source hashes
and rewritten-call audit records; installed vendor files are not modified.

`aiter.meta_size` always answers from the installed native module and falls
back to the vendor wrapper only when that module is unavailable. It must not
gate on `torch.cuda.is_initialized()`: official `fp8_kernel` queries device
properties at import time, which marks Torch initialized without selecting a
device. The vendor dispatch dummy then opened a context on device 0 in every
TP rank, and kernels on device 0 stalled periodically once about ten contexts
shared it.

## Validation and limits

The DTK 26.10 isolated probe imported the combined adapter/plugin and completed
real decode argument resolution. All eight visible devices returned 65520 MiB;
`cuda_initialized_after_imports` was false and `init_attempts` was empty.
The evidence summary records exact source hashes and its probe scope.
This was a pre-rebase integration probe, not an end-to-end run of the final
rebased PR branches. It does not establish that HIP itself allocates zero
resources, quantify post-restart memory savings, or close the remaining KV
capacity difference. The running GPQA service does not include this new capacity
and argument-resolution fix. No new operator-output comparison was performed.
