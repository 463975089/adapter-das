# Fuse independent sparse-MLA selector banks

Two validated dense/CSR selector banks use one launch, preserving the original
per-bank integer operations, sorting, padding, invalid-id handling and counts.
Each bank has a separate output. Original consumer validation runs first.

The existing fused-selector policy controls admission. gfx938/64-CU inputs
within the bounded generic row/width work domain use this implementation;
other devices, empty/large banks and SWA-only calls retain their native path.
No batch size, request length, new flag or environment variable is encoded.

Validation before model testing: 72 diagnostic cases compare output bytes to
both independent eager and original fused references, including three graph
replays with changing metadata. Diagnostic graph speedups are 1.06–1.66x.
Maintained tests add CSR/dense, non-unit strides and invalid CSR spans. The
full maintained attention suite reports 159 passed and two pre-existing XPASS;
L0 reports GREEN (0 warn). These are operator results, not serving TPOT.

This branch starts from the previously accepted inverse-RoPE version 7617b11.
The separate FP8 vendor-heuristic experiment failed HumanEval (139/164) and is
excluded. Full HE164, all-rank profiler and unprofiled serving gates are still
required before adopting this fusion in the model.
