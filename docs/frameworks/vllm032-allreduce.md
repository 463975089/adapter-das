# Official AR compatibility candidate

vLLM retains its communicator, ordinary AR, fused AR+RMS/residual, Gemma and
per-group quant math and fusion passes. Adapter owns only HCU admission, ABI,
selection, copy-in defaults, effective factors and safe installation. No Plugin
AR owner or fourth entrypoint is needed. Historical SGLang catalog `allreduce`
production evidence is not vLLM evidence; the two new `vllm_allreduce*` catalog
records are **candidate**, including the OFF/fallback row.

Public host API: `hcu_adapter.bridges.vllm_allreduce.install(*,
require_available=False) -> hcu_adapter.allreduce.State`, `state()` and
`effective_compile_factor()`. `state()` is read-only, never activates anything,
and refuses fatal/inherited attempted native state or effective-policy drift.
State reports availability, selection and fused registration independently;
`fusion_pass='official-global-gate-unchanged'` explicitly does not claim the
official pass is enabled or executed. This is not an ExtensionSpec capability;
required Plugin names/capabilities retain their existing meanings.

Runtime prepares explicit/configured HCU_AITER_AR or a kill veto after the
existing platform/version, DTK path and real `CudaRTLibrary` ABI checks, and
after the existing official FP8 fallback registration, before the current
mHC/indexer installation and optional extensions. The merged-main replay preserves PR22's runtime admission: any
failure there prevents FP8, AR and Plugin installation. AR preparation failure
stops Plugin activation and remains refused by the existing host latch.
With no explicit selection, AR preparation does not install the shim or import
its optional provider; independent FP8 startup behavior is unchanged.
`install(require_available=True)` must be called before any first installation
if ABI availability is required separately. A required OFF/killed backend checks
public provider/target ABI and loaded native identity but does not register fused
ops or select AR. A missing required ABI is an availability error, not a kill
failure. Kill+required with a valid provider remains healthy and OFF.

Existing `HCU_AITER_AR=1` opts in independently of global AITER, while the
official child `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` vetoes it. Kill
`HCU_DISABLE_OPS=aiter_ar` wins over either. Registered config/yaml is resolved
through `hcu_env`; `resolve(..., host='vllm')` uses vLLM's conservative unvalidated
auto rule, never the SGLang source marker. No new equivalent enable variable is
introduced. Controls and source provenance are exposed by framework-free
`hcu_adapter.allreduce.effective_sources()`.

Full-model admission also uses the public HcuPlatform.use_custom_allreduce
method. Official super behavior is retained. On otherwise unsupported HCU it
adds only explicit environment or registered config selection with an already
installed healthy, available AITER ABI and an actually supported HCU architecture.
Default, auto selection, OFF, kill, child veto and missing ABI do not extend that
platform gate. Fatal/drift checks precede admission. Batch-invariant mode
is unsupported for selected HCU AITER and refused before registration; a late
mode change is refused by state/getter/factor readers too. Its communicator
excludes AITER, and enabling
the generic gate there would otherwise allocate native C++ CustomAllreduce.
Unselected batch-invariant configuration keeps the official result. The ordinary
CLI user disable flag remains authoritative. QuickAllReduce keeps
its official architecture check and disabled gfx936 result; no communicator or
architecture predicate is replaced. Direct component initialization bypasses
the worker's config-to-global-enable step and cannot prove model selection.
The platform bridge source is included in selected AR compile-factor identity.

Provider-consumed `AITER_AR_MAX_SIZE_MB`, `AITER_AR_TRANSPORT` and
`AITER_AR_SMALL_MESSAGE` support their defaults and explicit environment inputs.
The installed provider reads the process environment directly; its public
constructor has no corresponding arguments. Preparation therefore refuses any
resolved config value differing from that actual input, before owned mutation.
An explicitly empty input is also refused when resolve would substitute a
default. Nothing temporarily exports config or replaces provider getters. The
current production catalog configures enable/disable, not these three controls;
CPU tests use a test-only catalog row to exercise real config parsing. Tests
execute the installed provider's two getter definitions and small-message read
expression, without importing/booting the provider or claiming constructor,
native or GPU execution. Config-only non-default support remains unavailable.

Ordinary admission defaults64MiB, staging128MiB, transportIPC, graph registration
False (copy-in). Explicit provider size/transport/small-message/registration
inputs are registered and remain opt-in; fabric requires its public capability.
They must agree across ranks according to the provider's existing contract.
The official default staging is128MiB. Ordinary admission is capped by
min(staging/2, provider admission); an explicit constructor max_size overrides
staging, not the configured provider limit. Official fused eligibility remains
64MiB. These source-level bounds do not prove native allocation or correctness.
No communicator or math is copied. Constructor failure may leave native state
and is sticky commit-fatal. Direct constructor calls while OFF are refused
before provider allocation; normal official selection uses the false gate and
keeps its fallback chain. This is **not** proof the chosen fallback is RCCL.

Target dependency is public `rocm_aiter_ops.register_custom_allreduce_ops()->bool`
and `CustomAllreduceRegistrationError(registration_started=bool)` (candidate27c)
only for selected/required admission. OFF/unrequired does not need that seam.
The seam registers only existing3 fused ops. Clean foreign-name failure has
registration_started=False; partial/opaque publication is fatal with no rollback
claim. Both bindings and official compile-factor registry publication are one
conservatively classified commit; any failure after commit begins is fatal.

**Registration and pass admission are separate:** the official pass manager
still requires `rocm_aiter_ops.is_enabled()` (global AITER) for its AR fusion pass.
This shim never enables it or copies that pass. An upstream AR-specific pass
admission seam is a separate open proposal. Gemma's eager consumer uses the
existing AR-specific gate/public getter; fake/dispatcher CPU success is not GPU
or model proof. Size overrides change provider ordinary admission/staging but
do not rewrite official fused `effective_max_size()`.

Selected factors hash the scoped public provider Python/ABI resource chain,
actual loaded module_custom_all_reduce DSO, effective policy and Adapter/official
wrapper source bytes. Unrelated AITER modules are excluded; OFF ignores optional
provider bytes/options. Raw registered getters validate drift even after vLLM
env caching and both ordinary/AOT factor collection use the public factor map.
No framework global getter is changed. Artifacts/config files remain immutable
after admission: file hashes cannot attest overwritten loaded DSO mappings or
transitive HIP libraries. Framework-free shared policy/provenance modules have
no Torch/vLLM/SGLang import; host imports stay in the vLLM bridge.

The computed official env-factor entry
`VLLM_HCU_ADAPTER_AITER_AR_POLICY` is installed in vLLM's public factor map; it is
not an operator environment override. Invalid policy/fatal state returns a
stable `implementation.invalid` marker because official compile_factors skips
getters that raise. That invalid marker remains hashed in both layers;
state/selector/constructor still reject computation. Ordinary getattr may be
memoized by official env caching; actual hash collectors call the registered
getter, which is tested directly. No whole-model engine/AOT cache reuse is
claimed from these CPU collectors.

The official gate also has a DP AG/RS consumer in CudaCommunicator. This bridge
does not rewrite that consumer or its selection order; DP AG/RS is not validated
by ordinary TP2 or fused registration contracts. Record it as a separate future
scenario rather than claiming all gate consumers passed.

## Failure and evidence matrix

| Case | Owned mutation / outcome |
| --- | --- |
| Default auto, no explicit/config/kill input | No install, no factor, no optional provider import; baseline preserved |
| Kill or HCU/official-child OFF | Gate false; optional vendor knobs neutral; direct native constructor refused; official fallback chain |
| Required ABI + OFF/kill | ABI/provider/native identity admitted, available=True and selected=False; no fused publication; missing ABI remains a clean availability failure |
| Invalid selected size/transport, config not consumed by provider environment, missing/async public ABI, provider unavailable, group initialized, env cached, factor/binding/registry foreign owner, missing fabric capability | Clean preparation failure before owned factors/bindings/records |
| Public registration error registration_started=False | Clean failure, no rollback needed |
| Registration partial/opaque/invalid return or later factor/binding/runtime-record failure | Sticky commit-fatal; preserve any actual published state; no rollback claim |
| Native constructor exception | Sticky commit-fatal; retain cause and attempted state |
| Effective policy drift / foreign replacement after install | Read-only state and dispatch refuse; invalid factor remains hashed |
| Healthy declarations-only fork | Existing immutable declarations checked; no activation in state query |
| Attempted native state / in-progress install inherited at real fork | Refusal with fresh child lock; timeout/deadlock controlled CPU doubles |

Source/installed CPU counts and retained failures are in the external handoff,
not promoted to catalog production status. Torch dispatcher FakeTensor tests
check shapes/handles; provider/group doubles check delegation/alias contracts.
They are not numerical GPU results.

Healthy declaration-only fork may retain immutable bindings. Native construction
attempts, inherited in-progress install and failed commit remain refused after
fork with a fresh lock; no silent state reset. This does not resolve unrelated
official plugins_loaded/process-local Extension report rebootstrap questions.

Runtime's real lock order is host action -> AR owner. Public AR readers,
factor, constructor, selector and installer query host health **before** taking
the AR lock; inside that lock they perform owner-local checks only. Same-thread
RLock reentry is insufficient evidence: real ProcessLocalActions/AR locks with
profile-observed health entry are tested with concurrent runtime/readers, as
well as inherited in-progress state. This is a fix for that specific lock cycle,
not a generic guarantee about arbitrary third-party native/import locks.

Minimum publishing dependencies: registered env/config/runtime registry,
existing HCU/version/runtime bridge, and the public commit-fatal/health ABI. This
module uses no model_load API and does **not** depend on lifecycle PR19 or a
Plugin wheel. The integration base3ca is evidence provenance, not a minimum exact
wheel requirement. This publication slice is based on rendering-safe PR23
09bd79a (the fix on60efffb), itself based on PR17. The earlier60efffb alone is
not a safe dependency: diagnostic __str__ failure could bypass its runtime latch.
PR23 supplies the public commit-fatal and read-only runtime-health ABI; PR17
alone lacks those safety semantics. The original independent90af slice had no
PR19, PR20, PR22, FP8 fallback, Flash owner-gate, W4 or Plugin dependency. The original
replay was based on main `b9fec6bed890d3dffef0fc04f28a8aae72371ba4`.
The conflict-resolution candidate replays only its narrow AR delta onto main
`24ac87cbe00f36a280c04cabbfe1f0fe4f3d109f`;
it preserves main's shared lifecycle, thin model-load facade, required resolver,
Flash and FP8 behavior rather than importing their old implementations. No
model-load API or Plugin is used by AR itself. Historical90af CPU packaging
results belong only to90af; new-main source, wheel and installed checks must
be run separately and do not grant native/model admission.

The platform-admission follow-up replays this same narrow AR slice onto main
`7b8d380e0a50d028e29144eb3e0f3705a63dda6a`. Main's intervening two L1.5 tool
changes are retained. Previous423df artifacts and component/model receipts stay
bound to423df, not this new source/wheel. The first full-model attempt failed
before ready in HCU Triton device-library discovery; it also exposed the public
platform veto described above. Neither is a scored AR numerical failure.

HCU/H0/H1, ordinary/fused numerics, changed-input TP2 Graph, same-model A/B
HumanEval32 and complete native zero-launch bootstrap tracing remain pending.

## Release boundary

This candidate targets vLLM 0.32 with the Core !70 public registration seam
(`0.32.0+das.27c12321d.dtk2610`). The checked local vLLM 0.33 source lacks
`register_custom_allreduce_ops` and `CustomAllreduceRegistrationError`.
Selected/required AR on that release is unsupported and must fail admission;
unselected default preparation remains neutral. No private registration, Core
source modification, global AITER enable, or 0.32-to-0.33 evidence transfer is
part of this candidate. The separate native C++ CustomAllreduce candidate has
a different provider and its model result does not establish this bridge result.
