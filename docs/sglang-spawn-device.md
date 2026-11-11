# SGLang scheduler device selection before bootstrap

SGLang v0.5.22 starts scheduler workers with spawn. Python loads sitecustomize
before the worker target receives gpu_id. Importing AITER during automatic
adapter policy resolution can create compute queues on the current default GPU.
Selecting the assigned GPU later leaves those queues on GPU0 and can exceed the
driver process limit even though TP ranks do not do useful work there.

The adapter binds maybe_reindex_device_id, the existing parent launch context
that encloses scheduler proc.start(). It preserves the original context and its
yielded logical device, and exports a private parent-scoped, one-use device hint.
The spawned interpreter consumes that hint and selects its GPU before either the
common or SGLang bridge runs. The hint is restored on every parent context exit;
non-spawn helpers ignore it. The contextmanager generator retains its identity,
so existing upstream aliases see the same binding. Original global bindings remain
live for later upstream/provider updates. Other adapter hosts are unchanged.

No public configuration is required. Visibility, rank arguments, kernel choice,
AR/norm policy and the official source files are unchanged. This does not remove
all auxiliary GPU contexts; it prevents schedulers from initializing GPU0 before
their assigned GPU. Launch paths outside maybe_reindex_device_id are not covered.

Regression: tests/adapter/test_sglang_spawn_device.py and test_bootstrap.py.
Runtime validation: TP8 Qwen3.8-27B, NEXTN 3/1/4, FA3, page 64, E5M2 KV,
real node1 image/ViT, unprofiled end-to-end requests and separate HIP Event spans.
