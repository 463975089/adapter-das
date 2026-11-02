# gfx936 official dense FP8 fallback validation

This records a local compatibility result, not a performance acceptance or a
general FP8 capability claim. Runtime source is `1a0445235e3b2bcda60b90b4490e938cdc45121e`,
based on Adapter PR17 `35f8f0569df4ffbed045bf0dc18974eaafd5ab1a`.
vLLM target is `a198423b7be0ccf9c9ac3fb01c02ac03ab145e93`, Python3.12,
Torch2.11/DTK26.10, one gfx936 HCU. No DAS Plugin is installed.

The original Channel-FP8 model failed selection because no eligible official
dense FP8 kernel was listed for this device. Direct Torch tensor and rowwise
`_scaled_mm` also rejected gfx936. The Adapter now registers the existing
official `TritonFP8ScaledMMLinearKernel` through `register_linear_kernel` as
the last ROCm FP8 candidate. Native predicates, device identity, model loading,
weight transforms and all existing selection priorities remain official.

Validation completed on 2026-11-01:

- CPU suite: 200 passed, 3 skipped; L0 GREEN. The three skips are not hardware
  passes.
- Actual official provider and complete Adapter→official QuantFP8/apply chain:
  12 cases each, 72 changed-input Graph replays in total. BF16/FP16 outputs,
  M1/17/65, nonunit tensor/token/channel scales and optional bias; worst NRMSE
  0.00179320 and minimum cosine 0.99999833 against decoded-FP8 FP32 GEMM.
  The complete-family reference reuses the official quantizer.
- Isolated Adapter wheel SHA256
  `136f0c5b03e65b92dac330620238d25830ce71ac96a65b8917bf54072c899f39`:
  all 173 packaged Adapter Python files match source and installation.
- `/model/Qwen3-VL-4B-Instruct-Channel-FP8/Qwen3-VL-4B-Instruct-Channel-FP8`
  starts with BF16 output/KV, `TRITON_ATTN`, Graph and prefix caching enabled;
  the log selects the official Triton FP8 linear class.
- HumanEval/0–31, exactly32 tasks, greedy/seed0/max2048/thinking disabled:
  raw0/32, format-only normalized31/32; task26 fails. Both scores use the
  namespace/chroot scorer under nobody with empty environment, no GPU/model
  visibility or external network. Raw fenced-format failure remains recorded;
  normalization performs no semantic repairs.
- Four image requests correctly identify swapped red/blue squares/circles.
  This is image-input sanity only, not general visual accuracy or encoder Graph
  validation.
- Thirty-six serial cold/warm prefix requests pass at lengths63–3073, including
  both sides of the tested16-token page boundaries. Cache counters advance and
  generated token IDs match. Page16 comes from the official cache/backend
  configuration; this API does not expose actual per-group manager state or
  per-request cached_tokens, so it is not a fine-prefix or MTP claim.

The model generates only about4.2tokens/s in this configuration. No controlled
performance comparison has passed. Contexts beyond3073tokens, concurrency and
general visual quality remain open. gfx938 has not been tested and is not enabled.
The original native Torch/selector failures remain valid evidence.

The durable local evidence root is `/data/models/vllm032-upgrade-20261031`:
`torch-fp8-validation/`, `adapter-fp8-linear-attestation.json`,
`qwen3vl4-fp8-official-fallback-server.log`,
`humaneval32-qwen3vl4-fp8-official-fallback[-scores]`, and
`probe-qwen3vl4-fp8-official-fallback-images`, and
`probe-qwen3vl4-fp8-official-fallback-prefix3072`.

The two repository probes are `tests/tools/_probe_vllm_fp8_linear.py` and
`tests/tools/_probe_vllm_fp8_family.py`. They require an explicitly selected
exclusive device and prepared target environment; they do not start models or
substitute for the model gates above.
