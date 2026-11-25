# V4.1 inverse RoPE head grouping

The existing gfx938 packed-cache decode writes masked BF16 attention output
and inverse RoPE in one program per token/head. This adds an equivalent
sixteen-head schedule to reduce the number of workgroups. It keeps the original
arithmetic, positive-zero empty-row inputs, FP32 rotations and BF16 stores.
Attention, cache interpretation, positions, selectors and lengths are unchanged.

`mla_decode_rope_triton` retains its existing metadata admission and installed
native-provider qualification. Qualification now executes the grouped kernel;
an unsupported provider/device/layout still declines before writing output.
No environment variable, production argument or batch/sequence-size constant
is introduced. The group size is a launch tile over the already admitted
64/128 local heads; token rows remain variable.

Validation evidence lives in
`run/dsv4.1/033/fp8_optimization_20261123/`:

- `INVERSE_ROPE_TILE_GATE.json` was frozen before discovery.
- `microbench/inverse_rope_tiles.json`: 144/144 cases matched both the old
  schedule and installed Core inverse RoPE byte for byte. This includes all
  BF16 encodings, empty/negative lengths, optional extra lengths, both head
  counts, nine variable row counts and three changing graph replays.
- Sixteen-head schedule speedup across measured cases: 0.997–2.432x.
  M42/H64/extra: 15.21 to 7.21 microseconds. M48/H64/extra: 17.24 to 7.78.
  These are isolated operator measurements, not serving TPOT claims.
- Maintained `test_mla_inverse_rope_tiles.py` additionally exercises int32 and
  int64 positions through actual dispatch and checks unsupported-output safety.
- Full HumanEval164, eight-rank activation trace and serving measurements are
  required independently; the preceding v9 result does not validate this change.

The prior one-head kernel remains available internally for direct schedule
comparison; serving and native qualification use the new grouped schedule.
