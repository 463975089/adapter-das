# SPDX-License-Identifier: Apache-2.0
"""Native inverse RoPE arithmetic over several heads per program."""
import triton
import triton.language as tl


@triton.jit
def _mask_copy_inverse_rope_heads(Result, Output, Main, Extra, Positions, CS,
                                 HEADS: tl.constexpr, HAS_EXTRA: tl.constexpr,
                                 GROUP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * GROUP + tl.arange(0,GROUP)[:,None]
    base = (row * HEADS + head) * 512
    count = tl.load(Main + row)
    if HAS_EXTRA:
        count += tl.load(Extra + row)
    n = tl.arange(0,512)[None,:]
    value = tl.load(Result+base+n,(n<448)&(count>0)&(head<HEADS),other=0.)
    tl.store(Output+base+n,value,(n<448)&(head<HEADS))
    pos = tl.load(Positions+row).to(tl.int64)
    k = tl.arange(0,32)[None,:]
    a = tl.load(Result+base+448+2*k,(count>0)&(head<HEADS),other=0.).to(tl.float32)
    b = tl.load(Result+base+448+2*k+1,(count>0)&(head<HEADS),other=0.).to(tl.float32)
    cosine = tl.load(CS+pos*64+k)
    sine = tl.load(CS+pos*64+32+k)
    even = a*cosine+b*sine
    odd = b*cosine-a*sine
    tl.store(Output+base+448+2*k,even.to(tl.bfloat16),head<HEADS)
    tl.store(Output+base+448+2*k+1,odd.to(tl.bfloat16),head<HEADS)
