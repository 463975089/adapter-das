import torch, torch.nn.functional as F, sys

torch.manual_seed(0)
dev = "cuda"
FP8 = torch.float8_e4m3fnuz
H, D, BS = 32, 128, 64

def make_case(B, R, ctx, int8=False):
    pages_per_req = (ctx + BS - 1) // BS
    n_pages = B * pages_per_req + 8
    if int8:
        kvb = torch.randn(n_pages, BS, 1, D, device=dev) * 0.5
        sk = (kvb.abs().amax() / 127.0).item()
        kv = torch.clamp(torch.round(kvb / sk), -127, 127).to(torch.int8)
        scale = torch.full((n_pages, BS, 1, 1), sk, device=dev)
    else:
        kv = (torch.randn(n_pages, BS, 1, D, device=dev) * 0.05).to(FP8)
        scale = (torch.randn(n_pages, BS, 1, 1, device=dev).abs() * 1e-3 + 1e-4)
    payload = kv.view(torch.uint8).reshape(n_pages, BS * D)
    scalebytes = scale.reshape(n_pages, BS).view(torch.uint8).reshape(n_pages, BS * 4)
    cache = torch.cat([payload, scalebytes], dim=1).view(n_pages, BS, 1, D + 4)
    if int8:
        qb = torch.randn(B, R, H, D, device=dev) * 0.5
        sq = (qb.abs().amax() / 127.0).item()
        q = torch.clamp(torch.round(qb / sq), -127, 127).to(torch.int8)
    else:
        q = (torch.randn(B, R, H, D, device=dev) * 0.05).to(FP8)
    w = torch.randn(B, H, dtype=torch.float32, device=dev)
    lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    bt = torch.arange(B * pages_per_req, dtype=torch.int32, device=dev).reshape(B, pages_per_req)
    return cache, q, w, lens, bt, (sq if int8 else 1.0)

def ref_math(cache, qf, w, lens, bt, ctx, pdtype=FP8):
    # qf: [B, H, D] float tensor (already dequantized); pdtype: payload byte dtype in cache
    B = qf.shape[0]
    flat = cache.view(-1, BS * (D + 4))
    gathered = flat[bt.long()]
    kv = gathered[..., :BS * D].contiguous().view(pdtype).to(torch.float32).reshape(B, -1, D)
    sc = gathered[..., BS * D:].contiguous().view(torch.float32).reshape(B, -1)
    scores = torch.bmm(kv, qf.transpose(1, 2))
    scores = F.relu(scores) * w.unsqueeze(1)
    scores = scores.sum(2) * sc
    pos = torch.arange(scores.shape[1], device=dev).unsqueeze(0)
    scores = scores.masked_fill(pos >= lens.unsqueeze(1), 0.0)
    return scores[:, :ctx]

def bench_graph(fn, iters=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters

def parity(ref, out, ctx):
    r, o = ref[:, :ctx].float(), out[:, :ctx].float()
    nanc = torch.isnan(o).sum().item()
    d = (r - o).abs()
    m = r.abs().max()
    big = r.abs() > 0.05 * m
    rel5 = (d[big] / r.abs()[big]).max().item() if big.any() else 0.0
    k = 64
    tr = r.topk(k, dim=1).indices
    to = o.topk(k, dim=1).indices
    ov = (tr.unsqueeze(2) == to.unsqueeze(1)).any(2).float().mean().item()
    return nanc, d.max().item(), rel5, ov

def run_case(tag, B, R, ctx):
    print(f"\n=== {tag}: B={B} R={R} ctx={ctx} ===", flush=True)
    cache, q8, w, lens, bt, sq = make_case(B, R, ctx)
    qf = [q8[:, r].to(torch.bfloat16).float() for r in range(R)]
    ref = torch.cat([ref_math(cache, qf[r], w, lens, bt, ctx) for r in range(R)], 0)
    del qf
    t_us = bench_graph(lambda: torch.cat([ref_math(cache, q8[:, r].to(torch.bfloat16).float(), w, lens, bt, ctx) for r in range(R)], 0))
    print(f"torch   : {t_us:8.1f} us  (ref)", flush=True)
    # lightop plain, bf16 q
    try:
        from lightop.gemmopt import paged_mqa_logits as lop
        qb = q8.to(torch.bfloat16)
        if R == 1:
            out = lop(qb, cache, w, lens, bt, None, ctx, clean_logits=False)
            us = bench_graph(lambda: lop(qb, cache, w, lens, bt, None, ctx, clean_logits=False))
        else:
            qb5 = qb.reshape(B * R, 1, H, D)
            bt5 = bt.repeat_interleave(R, 0).contiguous()
            lens5 = lens.repeat_interleave(R, 0).contiguous()
            w5 = w.repeat_interleave(R, 0).contiguous()
            out = lop(qb5, cache, w5, lens5, bt5, None, ctx, clean_logits=False)
            us = bench_graph(lambda: lop(qb5, cache, w5, lens5, bt5, None, ctx, clean_logits=False))
        nanc, mad, mrel, ov = parity(ref, out.view(B * R, -1).float(), ctx)
        print(f"lightop : {us:8.1f} us  nan={nanc} maxabs={mad:.4f} rel5={mrel:.4f} top64ov={ov:.3f}", flush=True)
    except Exception as e:
        print(f"lightop : FAIL {type(e).__name__}: {str(e)[:180]}", flush=True)
    # deepgemm fp8 verify arm (R>1): metadata accepted for speculative rows
    if R > 1:
        try:
            import deepgemm
            weff = w.repeat_interleave(R, 0).contiguous()
            out = deepgemm.paged_mqa_logits(q8, cache, weff, lens, bt, ctx, clean_logits=False)
            nanc, mad, mrel, ov = parity(ref, out.view(B * R, -1).float(), ctx)
            us = bench_graph(lambda: deepgemm.paged_mqa_logits(q8, cache, weff, lens, bt, ctx, clean_logits=False))
            print(f"dgm-fp8 : {us:8.1f} us  nan={nanc} maxabs={mad:.4f} rel5={mrel:.4f} top64ov={ov:.3f}", flush=True)
        except Exception as e:
            print(f"dgm-fp8 : FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
    # deepgemm int8 (gfx936 documented path), q scale folded into weights
    try:
        import deepgemm
        cache8, qi, w2, lens2, bt2, s8 = make_case(B, R, ctx, int8=True)
        weff = (w2 * s8).repeat_interleave(R, 0).contiguous()
        ref8 = torch.cat([ref_math(cache8, qi[:, r].view(torch.int8).float() * s8, w2, lens2, bt2, ctx, pdtype=torch.int8) for r in range(R)], 0)
        variants = []
        if R == 1:
            variants.append(("int8", dict()))
        else:
            variants = []  # handled by fp8 arm below
        out = None
        for vname, kw in variants:
            try:
                out = deepgemm.paged_mqa_logits(qi, cache8, weff, lens2, bt2, ctx, clean_logits=False, **kw)
                nanc, mad, mrel, ov = parity(ref8, out.view(B * R, -1).float(), ctx)
                us = bench_graph(lambda: deepgemm.paged_mqa_logits(qi, cache8, weff, lens2, bt2, ctx, clean_logits=False, **kw))
                print(f"dgm-int8[{vname}]: {us:8.1f} us  nan={nanc} maxabs={mad:.4f} rel5={mrel:.4f} top64ov={ov:.3f}", flush=True)
                break
            except Exception as e:
                print(f"dgm-int8[{vname}]: FAIL {str(e)[:120]}", flush=True)
                out = None
    except Exception as e:
        print(f"dgm-int8: FAIL {type(e).__name__}: {str(e)[:180]}", flush=True)

def run_int8_case(tag, B, R, ctx):
    print(f"\n=== int8 {tag}: B={B} R={R} ctx={ctx} ===", flush=True)
    from lightop.gemmopt import paged_mqa_logits as lop
    from sglang.srt.layers.attention.dsv4.paged_mqa_pers_jit import persistent_int8_paged_mqa_logits as pers
    cache8, qi, w2, lens2, bt2, s8 = make_case(B, R, ctx, int8=True)
    weff = (w2 * s8).repeat_interleave(R, 0).contiguous()
    ref8 = torch.cat([ref_math(cache8, qi[:, r].view(torch.int8).float() * s8, w2, lens2, bt2, ctx, pdtype=torch.int8) for r in range(R)], 0)
    c8v = cache8.view(torch.int8).view(cache8.shape[0], 64, 1, 132)
    arms = []
    def arm(name, fn):
        try:
            out = fn()
            nanc, mad, rel5, ov = parity(ref8, out.view(B * R, -1).float(), ctx)
            us = bench_graph(fn)
            print(f"{name:14s}: {us:8.1f} us  nan={nanc} maxabs={mad:.4f} rel5={rel5:.4f} top64ov={ov:.3f}", flush=True)
        except Exception as e:
            print(f"{name:14s}: FAIL {type(e).__name__}: {str(e)[:140]}", flush=True)
    q4 = qi if qi.dim() == 4 else qi.view(B, R, 32, 128)
    arm("lightop-int8", lambda: lop(q4, c8v, weff, lens2, bt2, None, ctx, clean_logits=False))
    for sms in (0, 120):
        arm(f"pers-jit sms{sms}", lambda s=sms: pers(q4, c8v, weff, lens2, bt2, ctx, num_sms=s))

if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    run_int8_case("decode", 16, 1, 4096)
    run_int8_case("decode-bs64", 64, 1, 4096)
    run_int8_case("verify", 16, 5, 4096)
