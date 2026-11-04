def install():
    from .kernels.qsa_scatter_triton import scatter_rows
    from .registry import register, MECH_REBIND
    import functools

    def patch(module):
        cls = module.QSATokenToKVPool
        if getattr(cls, "_hcu_scatter_installed", False):
            return
        cls._hcu_scatter_installed = True
        def set_key(self, layer_id, loc, token_k):
            scatter_rows(self.get_qsa_key_state_buffer(layer_id), loc, token_k)
        def set_compressed(self, layer_id, loc, compressed_k):
            scatter_rows(self.get_qsa_compressed_k_buffer(layer_id), loc, compressed_k)
        def set_positions(self, loc, positions):
            positions = positions.long()
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(3, -1)
            if positions.ndim != 2 or positions.shape[0] != 3:
                raise ValueError("QSA RoPE positions must be [tokens] or [3, tokens]")
            scatter_rows(self.qsa_rope_position_buffer, loc, positions.transpose(0, 1))
        for name, function in (("set_qsa_key_state_buffer", set_key),
                               ("set_qsa_compressed_k_buffer", set_compressed),
                               ("set_qsa_rope_position_buffer", set_positions)):
            setattr(cls, name, function)
            register("qsa_scatter", "triton", module.__name__+".QSATokenToKVPool."+name,
                     MECH_REBIND, source=__name__)
    from sglang.srt.mem_cache import qsa_kv_pool
    patch(qsa_kv_pool)
    from sglang.srt.layers.attention.qsa import mqa
    mqa.HAS_TILELANG = False
    register("qsa_mqa", "torch", mqa.__name__+".HAS_TILELANG", MECH_REBIND, source=__name__)
    if not getattr(mqa.qsa_mqa_prefill, "_hcu_boltops", False):
        from .kernels import qsa_mqa_boltops
        original_prefill, original_decode = mqa.qsa_mqa_prefill, mqa.qsa_mqa_decode
        @functools.wraps(original_prefill)
        def prefill(*args, **kwargs):
            return qsa_mqa_boltops.prefill(original_prefill, *args, **kwargs)
        @functools.wraps(original_decode)
        def decode(*args, **kwargs):
            return qsa_mqa_boltops.decode(original_decode, *args, **kwargs)
        prefill._hcu_boltops = decode._hcu_boltops = True
        mqa.qsa_mqa_prefill, mqa.qsa_mqa_decode = prefill, decode
        from sglang.srt.layers.attention.qsa import qsa_indexer
        qsa_indexer.qsa_mqa_prefill, qsa_indexer.qsa_mqa_decode = prefill, decode
        for module in (mqa, qsa_indexer):
            for name in ("qsa_mqa_prefill", "qsa_mqa_decode"):
                register("qsa_mqa", "boltops", module.__name__+"."+name,
                         MECH_REBIND, source=__name__)
    from sglang.srt.layers.attention.qsa import sparse_attn
    if not getattr(sparse_attn._get_best_config, "_hcu_lds_guard", False):
        original_config = sparse_attn._get_best_config
        @functools.wraps(original_config)
        def best_config(total_q):
            block, warps, stages = original_config(total_q)
            arch = getattr(sparse_attn.torch.cuda.get_device_properties(0), "gcnArchName", "").split(":", 1)[0]
            if arch == "gfx936" and block == 64 and stages == 2:
                if total_q <= 64:
                    stages = 1
                else:
                    block = 32
            return block, warps, stages
        best_config._hcu_lds_guard = True
        sparse_attn._get_best_config = best_config
        register("qsa_prefill", "triton", sparse_attn.__name__+"._get_best_config", MECH_REBIND, source=__name__)
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    if not getattr(QSAIndexer, "_hcu_prep_installed", False):
        QSAIndexer._hcu_prep_installed = True
        def use_fused_prep(self, tensor):
            return False
        QSAIndexer._use_fused_prep = use_fused_prep
        register("qsa_scatter", "triton", "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer._use_fused_prep",
                 MECH_REBIND, source=__name__)
