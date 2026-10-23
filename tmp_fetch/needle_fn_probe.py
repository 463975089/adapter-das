import io
p = "/sglang-v0522/python/sglang/srt/layers/attention/dsv4/indexer.py"
s = io.open(p, encoding="utf-8").read()
anchor = "        query_rows = q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]"
probe = (
    "        import os as _os\n"
    "        if _os.environ.get('HCU_DUMP_MQA') == '1' and not getattr(self, '_hcu_fn_logged', False):\n"
    "            self._hcu_fn_logged = True\n"
    "            _mod = getattr(fn, '__module__', '') or str(fn)[:70]\n"
    "            _qual = getattr(fn, '__qualname__', '')\n"
    "            print(f\"[HCU fn-probe] fp4={use_fp4_indexer} aiter_fp4={use_aiter_fp4} fn={_mod}:{_qual}\", flush=True)\n"
)
assert anchor in s, "anchor missing"
assert "_hcu_fn_logged" not in s, "already needled"
s = s.replace(anchor, probe + anchor)
io.open(p, "w", encoding="utf-8").write(s)
print("NEEDLE_OK")
