# HY4 MQA logit layout conversion

The DTK FP8 scorer returns full K columns. Core consumes request-local columns,
so the Adapter shifts each row by its request start and masks invalid columns
with `-inf`. Long rows now use a 512-element tile instead of 256. Rows narrower
than 512 retain the existing launch. Output shape, strides, bounds, dtype and
allocation remain unchanged; no device-to-host read or new environment flag is
introduced.

gfx938 component validation, using the installed v0.33 Core and LightOp with
`LIGHTOP_MQA_LOGITS_BETA=1`:

| FP8 Q | FP8 K | Compressed width | Full scorer + conversion before / after |
| --- | --- | --- | --- |
| `[378,32,128]` | `[164189,128]` | 84835 | 2.351 / 2.119 ms |
| `[1134,32,128]` | `[164189,128]` | 84835 | 7.107 / 6.411 ms |

Seven interleaved timing samples per variant; median time reduction is about
9.8%. Complete FP32 logit outputs, including invalid columns, are bitwise equal.
These are synthetic component inputs with trace-derived dimensions, not model
TTFT or a distributed throughput measurement. gfx936 performance is unmeasured.

Thirteen HCU regression cases passed: request windows, strided source/output,
row offsets, widths around 256/512, empty rows/columns, changed-input CUDA Graph,
and the real Core indexer body with FP8 cache/FP32 scale, top-k and ragged decode.
Run with the paired Adapter and vLLM installed:

```bash
HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter pytest -q \
  tests/vendor/test_hy4_compact_logits_gpu.py \
  tests/vendor/test_hy4_sparse_indexer_gpu.py
```

Whole-model HumanEval and unprofiled serving performance remain separate gates.
