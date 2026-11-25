# V4.1 Channel-FP8 operator candidates

Experimental worktree based on Adapter `dd100f4` and Plugin `f6d9bbc`.
Hardware: gfx938, 64 CUs, BW1100; PyTorch 2.13 / DTK2610, vLLM 0.33.
These measurements qualify operators; full-model promotion requires the
separate frozen serving and HumanEval164 gates.

## Shared activation and token FP8

The provider preserves both BF16 rounding boundaries of native clamped SiLU.
Its two quantization contracts match the actual downstream quantizer:
upstream divides by a floored scale; LightOp multiplies by `448/amax` without
a scale floor. The active quantizer bridge exposes its backend and graph
policy as metadata; its existing dispatch and defaults are unchanged.

LightOp's overflowing reciprocal produces negative canonical NaNs for zero
values in tiny nonzero rows. Preserve those FP8 codes as well. Unaligned
LightOp widths retain the original activation/quantizer because that native
scalar conversion has a different overflow contract. Upstream supports
arbitrary positive widths through 8192; the LightOp fusion requires alignment
to 16 columns. Neither path fixes the number of rows or context length.

Plugin tests passed 330 cases, including every finite BF16 gate value,
generic and 2304-column kernels, actual downstream GEMM, unsupported-width
fallback and graph replays with changed inputs. Adapter bridge tests passed
24 CPU cases. Sixty paired microbench shapes improved 1.46–3.40x; the
42x2304 LightOp activation/quantization measured 9.088 -> 3.888 us.

## FP4 cache-page staging

Keep the exact existing byte-copy kernel and increase the live-page launch
from roughly one to sixteen programs per CU, bounded by row and page counts.
All bounds still come from current device lengths on replay. Allocation,
cache format, slot-zero tail bytes and output-table numbering are unchanged.

Ten row/context/page-layout combinations, each with five launch alternatives
and three changed-metadata graph replays, matched every baseline byte.
Fourteen public-entry tests additionally passed an independent CPU byte-plane
oracle, including empty batches, partial pages and strided tables.
For 32-record physical pages and 42 rows at length32769, paired timings were
522 -> 350 us. Short one-row/1024-token copies remained approximately4 us.
The same rule was tested with 128-record pages and varied row counts up to257.

## Remaining validation

The unchanged serving baseline completed 192 requests at input65536/output1024,
global concurrency56 / DP8 / EP8, with request TPOT P50 20.938610 ms and
P95 22.269926 ms. Full real-prefill HumanEval baseline passed159/164, with
zero request errors or truncations. Candidate acceptance must preserve every
baseline-passed task, meet the frozen serving gates and be checked in a fresh
profile. No whole-model speedup is inferred from these microbenchmarks.
