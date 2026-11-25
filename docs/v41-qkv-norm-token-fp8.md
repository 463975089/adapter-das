# BF16 Q/KV normalization and token-FP8 producer

`qkv_norm_token_fp8_triton.norm_quantize` fuses the native Q/KV RMSNorm with
dynamic per-token quantization of Q. Q is explicitly rounded to BF16 before
amax and FP8 conversion. KV retains its native BF16 output. Upstream division
and scale floor, and LightOp reciprocal multiplication without a floor, remain
separate contracts. No framework imports, process-wide BLAS changes or new
environment variables are used.

Rows, input stride, Q/KV dimensions and epsilon come from the caller. Q/KV
widths up to 8192 are bounded by the qualified reduction size. LightOp scalar
tails are refused; an owned consumer must retain its native path for them.
Inputs/weights are read only. Grad/autocast and unqualified devices are refused.

Validation on gfx938, 64 CUs:

- 35 native normalization/quantizer comparisons, including tiny/zero rows,
  padded strides, Q widths 17 through 8192, rows 1 through 1024 and three
  changing Graph replays. FP8 bytes, FP32 scale bytes and KV BF16 bytes match.
- 20 maintained native/Graph tests and L0 GREEN.
- 18 actual attention/indexer projection-consumer cases with checkpoint weights
  and native Channel-FP8 Linear modules. Outputs match exactly before and after
  three changing Graph replays, including empty rows.
- At Q=1280/KV=512, M=42, norm plus LightOp quantization measured 6.470 us
  versus 3.908 us fused (interleaved Graph/event timing). This is an operator
  result; full-model HumanEval, trace and unprofiled TPOT remain separate gates.

Raw artifacts and reproducible scripts are in
`run/dsv4.1/033/fp8_optimization_20261123` in the shared workspace:
`microbench/qkv_norm_token_fp8.json`, `microbench/qnorm_real_consumers.json`,
`probe_qkv_norm_token_fp8.py` and `probe_qnorm_real_consumers.py`.
