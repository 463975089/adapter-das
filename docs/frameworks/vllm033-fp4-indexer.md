# vLLM 0.33 FP4 indexer compatibility candidate

This document records the accepted compatibility route for DeepSeek-V4.1 FP4
indexer pages on HCU. It is a candidate integration, not a production or
performance claim.

## Known deviation from the adapter layout rule

`adapter-das/skills/hcu-adapter-dev/SKILL.md:195-198` says not to copy a
cache tensor to work around layout. The decode path in this candidate is an
explicit, reviewed deviation from that rule. It exists because LightOp's
`paged_mqa_logits_fp4` ABI accepts only canonical 64-record pages. This is a
compatibility path, not the design answer.

The prefill path does not stage. `dsv41_gather_index_k_fp4(cache, slots,
page_size)` receives the allocator page size and gathers directly. Decode uses
staging only when the allocator page has a record count other than 64.

## ABI evidence

The inspected LightOp implementation contains `constexpr int kPageSlots = 64`
and checks the cache geometry against `[pages,64,1,68]` or `[pages,4352]`.
The source evidence is the inspected LightOp source
`csrc/paged_mqa_logits/paged_mqa_logits_fp4.cu:699-719` in the 0.6.0 source
bundle. The installed binary independently exposes the same rejection strings
in `workspace/issue/pr/20261108_v033_fp4_indexer/evidence/zerocopy_probe/native_abi_strings_symbols.log:10-12`.
The single-card probe rejected 32-record views and accepted the canonical
64-record control. It also rejected `table_block_size=32` and accepted the
64-record, `table_block_size=64` control.

## Staging shape and cost

For a physical page with `R != 64` records, decode gathers one payload
`int8[64]` and one `int32` scale word per logical slot. It assembles canonical
pages with 4096 payload bytes followed by 256 scale bytes, or 4352 bytes per
staged page. Capacity is static and uses
`pages_per_row = (max_context_len + 63) // 64`.

The staging pool allocation is:

```text
rows * pages_per_row * 4352 bytes
```

For example, with `max_context_len=4096`, `pages_per_row=64`, one decode row
costs `64 * 4352 = 278,528` bytes, about 272 KiB. Eight rows cost 2,228,224
bytes, about 2.125 MiB. The actual allocation scales with the layer's static
`max_context_len` and the flattened decode row count.

The operation performs one device-side byte gather per decode call. It is not
zero-copy and no performance improvement is claimed. Invalid columns use slot
zero only as an in-bounds read and remain masked by the native scorer's
`context_lens`.

## Removal plan

Remove staging only after one of these contracts is true and covered by a
native probe and regression test:

1. LightOp `paged_mqa_logits_fp4` accepts 32/128-record pages and a matching
   variable table-block contract.
2. The vLLM allocator is changed to provide uniform 64-record FP4 pages for
   every admitted ratio, with producer and metadata semantics preserved.

Until then, the compatibility route requires review acceptance. CPU/mock tests
prove shape and metadata behavior only. They do not prove native correctness,
end-to-end accuracy, or performance.
