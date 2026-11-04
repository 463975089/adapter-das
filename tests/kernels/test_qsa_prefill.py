import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("total", [64, 96])
@pytest.mark.parametrize("prefix", [0, 7])
def test_public_qsa_prefill_fits_lds_and_matches_softmax(total, prefix):
    from hcu_adapter.qsa_pool import install
    from sglang.srt.layers.attention.qsa import sparse_attn as S
    install()
    torch.manual_seed(total + prefix)
    lengths = [total//2 + 3, total//2 - 3]
    kv_lengths = [n + prefix for n in lengths]
    q = torch.randn(total,8,256,device="cuda",dtype=torch.bfloat16)
    k = torch.randn(sum(kv_lengths),1,256,device="cuda",dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0,lengths[0],total],device="cuda",dtype=torch.int32)
    cu_k = torch.tensor([0,kv_lengths[0],sum(kv_lengths)],device="cuda",dtype=torch.int32)
    ids = torch.arange(max(kv_lengths),device="cuda").expand(total,-1).clone()
    causal = torch.cat([torch.arange(n,device="cuda")+prefix for n in lengths])
    ids = ids.masked_fill(ids > causal[:,None],-1).to(torch.int32)
    scale = 256**-0.5
    if prefix:
        actual = S.sparse_gqa_fwd_interface_triton_ck(q,k,v,ids,cu_q,cu_k,
            torch.tensor(kv_lengths,device="cuda",dtype=torch.int32),scale)
    else:
        actual = S.sparse_gqa_fwd_interface_triton(q,k,v,max(lengths),ids,cu_q,scale)
    expected = []
    q_start = k_start = 0
    for n,m in zip(lengths,kv_lengths):
        scores = torch.einsum("qhd,kd->hqk",q[q_start:q_start+n].float(),k[k_start:k_start+m,0].float())*scale
        mask = torch.arange(m,device="cuda")[None,:] > (torch.arange(n,device="cuda")+prefix)[:,None]
        scores.masked_fill_(mask[None,:,:],float("-inf"))
        expected.append(torch.einsum("hqk,kd->qhd",scores.softmax(-1),v[k_start:k_start+m,0].float()))
        q_start += n
        k_start += m
    torch.testing.assert_close(actual.float(),torch.cat(expected),rtol=0.025,atol=0.025)
