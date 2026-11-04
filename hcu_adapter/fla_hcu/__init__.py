# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sglang-das HCU fla kernels, vendored for the adapter (gfx938).

Source: maxiaobetter/sglang-das branch main_glm53_flash_feature_cleanup_tx.
Ported files (imports rewritten to this package, bodies verbatim):

- kda.py / chunk_delta_h.py / chunk_intra.py /
  chunk_intra_token_parallel.py / l2norm.py  — KDA prefill chunk pipeline
  (fixes the gfx938 wrong-Aqk of the stock chunk_kda that
  HCU_KDA_PREFILL_IMPL=emu works around)
- fused_sigmoid_gating_recurrent.py — KDA decode recurrent kernel
- cumsum.py / index.py / op.py / utils.py — fla helpers (fork versions)
- conv1d_triton.py — causal_conv1d_fn / causal_conv1d_update with the
  fork's HCU fixes (stock v0.5.1 kernels fail their red gates on
  gfx938: update cos 0.26, fn varlen initial-state dead)

The fork's optional aiter prebuilt fast paths (SGLANG_USE_AITER_* env,
present only in the sglang-das image) stay off by default; on this
stack the pure-triton arms run.
"""
