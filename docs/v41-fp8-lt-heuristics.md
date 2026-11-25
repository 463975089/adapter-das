# Channel FP8 hipBLASLt heuristics

The native CT parameter loader, quantization and Linear providers are preserved.
The existing Adapter scaled-MM custom op can select a faster installed vendor
algorithm during eager warmup. This changes neither checkpoint bytes nor the
token/channel scale contract. No production switch is added.

Admission is gfx938 with 64 CUs, contiguous E4M3 inputs/weights, FP32 token/channel
scales, BF16 output without bias, M1..256, N>=1024 divisible by16, K>=1024 divisible
by128. Other contracts retain the original LightOp/upstream path. These are tile
and overhead bounds, independent of bs=7 or a 65536-token request.

The C++ helper compiles against the installed vendor headers and queries up to
256 supported zero-workspace heuristics. Candidate names must belong to the
qualified F8BS channelwise family without split-K atomic accumulation. Algorithm
indices are never stored across processes/library versions. Select a candidate
only after median device-event timings improve at least5% over the first vendor
heuristic and2% over the current best. The bounded128-entry cache is local to each
host thread/device. It owns descriptors, with no tensor/device-scratch ownership.
Matmul copies launch parameters before returning; separate streams therefore do
not share mutable device state. Unseen capture shapes use the original provider;
there is no compilation, tuning or synchronization during graph capture/replay.

The helper uses the existing DTK library and a host C++ compiler. Missing compiler
or header support retains native execution. The source/header/library identity
keys the build cache, and file locking plus atomic publication supports eight
worker processes. Execution failures are surfaced instead of hidden by fallback.

Before model validation: 46 initial actual-weight/generic-shape cases passed the
frozen FP8 gate (relativeL2<=.005, cosine>=.99999, independent sampled CPUFP64
rtol.015625/atol2e-5), three changed graph replays, multiple streams, twenty
determinism checks and immutable weights/scales. Small matrices showed a0.2us
regression, so final admission retains their native provider. No tolerance was
changed. Maintained tests cover native vendor parity, capture ownership and cold
capture fallback. The real CT post-load Linear gate covers33 cases, including
empty/fallback rows, amplitude0/1e-30/.25/8 and three changed graph replays.

The initial tiny-input cosine fixture underflowed even for native-identical
outputs (including M257, where this optimization never runs). Its failures were
retained; the rerun normalizes both vectors by the same reference magnitude before
computing the same L2/cosine thresholds. This fixes metric underflow without
changing the numeric contract.

The first whole-model startup exposed a native fallback initialization problem:
when every eager call selects the new path, LightOp first creates its per-device
hipBLASLt handle inside a cold-shape capture, which is forbidden. The existing
bridge now eagerly calls the original provider once per device before selecting
the tuned path. A fresh-process regression deliberately avoids the native
control before capture and checks three changed-input replays of an unseen
M257 fallback shape. That test also covers the lazy handle lifecycle missed by
ordinary eager-versus-native comparisons.

No whole-model speedup is established by these operator results. Full164
HumanEval>=156/164, zero request errors/truncations/invalid completions, eight-rank
trace activation and unprofiled serving measurements remain required.
