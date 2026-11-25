# Pipeline the deterministic BF16 router projection

Use a 512-element K load tile and two software-pipeline stages. Express the
native per-64-expert K stagger in element offsets, including wrap within a
load tile; the ordered FP32 accumulation and exact FP32 logits are preserved.
The existing generic row, shape, dtype, device and fallback admission is
unchanged. The consumer still owns the same GateLinear instance and live weight.

The original frozen router gate is unchanged. Discovery compared 27 tiling
configurations. Full qualification passed 126/126 cases spanning all supported
positive row counts and six scenarios, including ties, padding, extreme values,
CPU-FP64 reference and three graph replays with changed inputs/weights/bias.
Route IDs and logits are exact; native routing-weight tolerances are unchanged.

Paired graph timings of the complete projection-plus-native-router chain were
1.12–1.16x faster than the accepted previous kernel on all supported row counts.
The standalone M42 projection measured approximately 25.8 to 20.9 microseconds.
These are operator measurements; full HE164/profile/TPOT gates remain required.
