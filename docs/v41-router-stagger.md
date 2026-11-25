# Deterministic BF16 router projection

This kernel replaces BF16 -> FP32 GateLinear GEMM only. The native
sqrtsoftplus / bias / TopK and weight loading remain in their original owners.
The K traversal matches the qualified native Tensile stagger order: 128 BF16
values of offset per 64-expert block. The tiled loop has masked generic row
loads and writes; it does not encode request batch, prompt length or generation
length. Model dimensions are 5120 hidden / 384 experts. Qualified inputs are
contiguous BF16 on gfx938, inference without autocast; M0..4 or even M32..64.
Other rows, dimensions, layouts, dtypes and devices retain the native consumer.

The conservative admission is based on an exhaustive M1..64 survey, not a
single target shape. Some native shapes choose GSU8 atomic accumulation, whose
partial-tie order cannot be reproduced by a deterministic serial projection.
The wider proposal failed M14/18/24 tie IDs and remains rejected. Byte tests
also found native differences at M6/12/14/18/24/43. Those rows all fall back under
the new contract. No exceptional-shape lookup list is used in production.

Original operator gates require exact native routing IDs/order, routing weights
rtol1e-5/atol1e-6, independently sampled FP64 logits rtol1e-4/atol1e-5 and L2<=5e-6,
and deterministic changing-input/weight/bias/mask graph replays. The new admitted
126-case gate adds native-logit byte identity. Maintained tests cover every
admitted row including zero, repeated live weight changes and input immutability.

Checkpoint consumer validation additionally compares logits byte for byte to
native and retains FP64 error diagnostics. At amplitude8, native checkpoint
logits can themselves exceed the normalized-fixture absolute FP64 tolerance;
this is not reported as native passing that criterion. The immutable initial
failure and native-only control remain beside the corrected consumer fixture.
The original 126-scenario precision gate keeps its tolerances unchanged.

No new production environment variable or service flag is introduced. Whole
model promotion still needs full164 HumanEval >=156, zero failed / truncated /
invalid requests, eight-rank trace activation and no-profiler serving results.
