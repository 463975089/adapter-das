# vLLM 0.33 unquantized AITER compatibility candidate

This Adapter candidate uses the public
`Platform.get_unquantized_aiter_moe_compatibility()` interface proposed in
OpenDAS/vllm. A Core wheel without that interface continues its existing MoE
selection; installing this Adapter alone does not admit HCU AITER experts.

Select `--moe-backend aiter` for explicit admission. Automatic selection still
obeys the official `VLLM_ROCM_USE_AITER` and `VLLM_ROCM_USE_AITER_MOE` policy.
This does not enable global AITER, alter CDNA identity, or admit quantized AITER
experts. The official selector, method, loading, modular kernel, routing,
prepare/finalize and TP reduction remain Core owners.

Initial scope is gfx936, BF16/FP16, gated SiLU/GELU, ordinary TP partitions with
supported dimensions, no extra padding, expert bias, LoRA, EP/EPLB/DP/PCP,
router-weight-on-input or fused shared experts. Unsupported explicit selection
fails before conversion; automatic selection retains official fallback. Separate
ordinary shared experts remain with the official runner. Preprocessed runtime
weights are refused because their format provenance is not yet qualified.

Weights remain canonical raw across postprocessing. Official shuffle(16,16) and
vendor asm_shuffle are distinct permutations; relabeling one as the other is
incorrect and can also eliminate a valid Triton fallback. Vendor neutral
SwiGLU `None` is passed only after rejecting non-neutral semantics. Native
execution errors propagate. The process-local publication guard refuses retry
after partial registration, including BaseException and incomplete fork states.

For this provider, startup materializes `VLLM_MOE_SKIP_PADDING=0` before workers
spawn, keeping valid routes for Graph padding. A previously cached conflicting
value causes a clear error. No graph mode or model scheduler is replaced.

CPU contracts: `tests/adapter/test_vllm_moe_aiter.py`. The initial isolated
Core+Adapter stack passed 19 installed CPU contracts and a single-card component
probe: 14 configurations, 56 expert reference states, 14 whole-runner forwards,
42 changed-input Graph replays and 5 manual parameter-update/postprocess checks.
Original cosine/global-relative/absolute gates were retained. All 14 observed
vendor plans were Triton, including E256/H2048/N512/top8. This is not ASM/CK,
distributed TP, public checkpoint reload, model HumanEval32 or performance
acceptance. The candidate remains under integration validation.
