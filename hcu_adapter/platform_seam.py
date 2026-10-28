"""Platform-semantics seam — governance §0.5 is_cuda 路线的落地点.

adapter 统一以 CUDA 语义回答上游 sglang 的平台判定，让新模型/新功能
不经适配天然走 CUDA 分支（卡商追新）。只翻 sglang 的判定 seam，共
两个模块、五个符号：

1. ``sglang.srt.utils.common`` — is_cuda / is_hip / is_cuda_alike。
   必须在任何 sglang 模块 import 之前安装：common.py 在 import 时就用
   is_hip() 选定 FP8_E4M3_MAX 并写 builtins，全仓 `from sglang.srt.utils
   (common) import is_hip` 的常量绑定也发生在各自模块 import 时——
   判定函数换掉后这些绑定自然拿到 CUDA 语义。
2. ``sglang.srt.platforms`` — _is_cuda_available 打开 / _is_rocm_available
   关闭 → current_platform 解析为 CudaSRTPlatform。

明确不翻硬件层：torch.version、triton 编译后端选择、vllm HIP 构建的
平台判定保持 HIP 真值（全局翻 torch.version 会让 triton 走 CUDA 编译
后端直接打爆内核编译，也破坏 vllm op 探测）。内核选择层
（sglang.kernels PlatformInfo.detect）同样保持 HIP 真值，由
sitecustomize 的 capability 桥衔接——「模型层 CUDA 语义 + 内核层
HIP 真值」是本路线的合成态，两侧的翻译工作就是 adapter 的本体。

fp8 fnuz 等已验证 HIP 语义护栏不随翻（governance §0.5 白名单）：
fnuz 是卡上数学事实，不是平台判定。

开关：HCU_PLATFORM_SEMANTICS=cuda（默认，§0.5 拍板路线）| hip
（旧适配路线，降级调试 / A-B 对照用）。
"""

import logging

LOG = logging.getLogger("hcu.adapter")

# seam 目标清单（install 逐一登记 registry；target = dotted symbol）
_TARGETS = (
    "sglang.srt.utils.common:is_cuda",
    "sglang.srt.utils.common:is_hip",
    "sglang.srt.utils.common:is_cuda_alike",
    "sglang.srt.utils.common:is_flashinfer_available",
    "sglang.srt.platforms:_is_cuda_available",
    "sglang.srt.platforms:_is_rocm_available",
    "sglang.srt.model_executor.runner_utils.shared_read_event:make_external_event",
)


def _install_gfx_probe_guard(_common):
    """Neutralize is_gfx*_supported arch probes.

    Their bodies call torch.cuda.get_device_properties(0) after the
    torch.version.hip truthy check (which we deliberately keep truthful)
    and blow up with "No CUDA GPUs are available" when visible devices are
    an offset set (HIP_VISIBLE_DEVICES=8..15) probed before CUDA init —
    exactly sitecustomize's early import order. HCU is gfx93x: every one
    of these NVIDIA/RDMA-next gates is semantically False.

    Patches BOTH namespaces: `sglang.srt.utils.common` and the re-export
    layer `sglang.srt.utils` (its __init__ does `from .common import *`,
    so late `from sglang.srt.utils import is_gfx*_supported` consumers --
    e.g. kernels/ops/quantization/fp8_kernel -- must not pick the
    ORIGINAL probe out of the package namespace). Works without _common
    too (module-import fallback for the legacy-hip/no-GPU paths).
    """
    import sglang.srt.utils as _pkg  # noqa: PLC0415 -- cheap; usually loaded
    import sglang.srt.utils.common as _c  # noqa: PLC0415 -- cheap; usually loaded

    def _false():
        return False

    targets = [_pkg, _c]
    if _common is not None and _common not in targets:
        targets.append(_common)
    for target in targets:
        for _name in list(vars(target)):
            if _name.startswith("is_gfx") and _name.endswith("_supported"):
                setattr(target, _name, _false)
    LOG.warning(
        "HCU adapter: gcn arch probes (is_gfx*) guarded to False"
        " (early-import safe under partitioned visibility)"
    )


def platform_env_defaults():
    """HCU-stack platform defaults for sglang env vars (R-08 A-tier).

    Pure ``os.environ.setdefault`` calls — an explicitly exported value always
    wins (env三原则: correct deployment needs zero envs, explicit override is
    top priority). Extracted from install() so tests can pin the semantics
    without a GPU/sglang import. Landing ledger (occurrences across
    cookbook/_launch_/_test_ scripts at triage time, docs/env-vars.md §R-08):

    * SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python (31x) — rust TreeCore has
      no prebuilt extension on HCU; the python core is the platform fact.
      SGLANG_RUST_BUILD_MODE=never (4x) rides along (same build machinery).
    * SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false (4x) — inbalance check
      fires on asymmetrical TP layouts this stack uses.
    * HCU_DEEPGEMM_BRIDGE=1 (13x) — DTK deepgemm alias+wrapper bridge,
      validated on gfx936 EP-LL and gfx938 GLM-5.3 lines; harmless when no
      DeepGEMM path runs.
    Deliberately NOT defaulted (triage verdicts): HCU_AITER_AR — the hcu_env
    auto rule is already the right default (refuses legacy layouts, allows
    the GraphSafe one); the 12 manual =0 exports predate that rule.
    SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 was landed with R-04 (35037be): the
    upstream all-M precompile session is pure boot tax on the AOT DTK wheel
    (measured ~52s/shape on nmz26 tp4, boot 6:02→3:57; bw7 EP lines saw
    5.7-7.4GB per shape squeeze the mem budget).
    """
    import os

    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    os.environ.setdefault("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "python")
    os.environ.setdefault("SGLANG_RUST_BUILD_MODE", "never")
    os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "false")
    os.environ.setdefault("HCU_DEEPGEMM_BRIDGE", "1")


def install():
    import os
    import time

    import torch

    from hcu_adapter import hcu_env, registry

    # sgl-kernel 版本门 + 平台 env 默认（R-04/R-08 A 类；上方
    # platform_env_defaults 的 docstring 是逐项落账与判由）。
    platform_env_defaults()

    sem, _src = hcu_env.resolve("HCU_PLATFORM_SEMANTICS")
    if sem != "cuda":
        LOG.warning(
            "HCU adapter: platform semantics=%r (legacy adapter route;"
            " set HCU_PLATFORM_SEMANTICS=cuda for the §0.5 is_cuda route)",
            sem,
        )
        # legacy hip 路线也要装 gcn 探针守卫：torch.version.hip 真值分支下
        # is_gfx*_supported 会裸调 get_device_properties(0)，卡分区可见性
        # （HIP_VISIBLE_DEVICES 非从 0 起）+ CUDA 未 init 的早期 import 即炸。
        _install_gfx_probe_guard(_common=None)
        return
    if not torch.cuda.is_available():
        if torch.version.hip is None:
            # 非 HIP 构建且 GPU 面完全不可见（无卡开发机等）：seam 无从谈
            # 起，gcn 守卫同样装上——后续任何 import 碰到 is_gfx* 都能拿到
            # 安全 False 而不是炸。
            _install_gfx_probe_guard(_common=None)
            return
        # HIP 构建但首轮 CUDA 探测瞬时失败：新容器里驱动枚举会与并发编译
        # （tvm_ffi dlpack 等）赛跑，数分钟后同 env 复测 device_count=8。
        # 不因瞬态弃装——下方 seam 只重写 sglang 的平台判定答案，全部是
        # 常量函数，不需要活的 CUDA 上下文；稍候重试后仍不可见则带告警
        # 照装（真正无卡时后续初始化自会响亮失败）。
        for _ in range(8):
            time.sleep(2)
            if torch.cuda.is_available():
                break
        else:
            LOG.warning(
                "HCU adapter: CUDA face unavailable at install() on a HIP"
                " build (torch.version.hip=%r) -- installing seam anyway"
                " (semantics-only, needs no live CUDA context)",
                torch.version.hip,
            )

    from sglang.srt import platforms as _pf
    from sglang.srt.utils import common as _common

    #这两个 import 本身会执行 common.py 顶层 `if is_hip():` ——彼时判定
    # 还是原函数，fnuz 常量已被冻结并写进 builtins；patch 后必须重算
    # （CUDA 语义 = e4m3fn，max 448）。
    _common.FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
    _common.FP8_E4M3_MIN = -_common.FP8_E4M3_MAX
    import builtins as _builtins

    _builtins.FP8_E4M3_MAX = _common.FP8_E4M3_MAX
    _builtins.FP8_E4M3_MIN = _common.FP8_E4M3_MIN

    def _hcu_is_cuda():
        return True

    def _hcu_is_hip():
        return False

    def _hcu_is_cuda_alike():
        return True

    def _hcu_pf_cuda_available():
        return True

    def _hcu_pf_rocm_available():
        return False

    # flashinfer 可用性也随语义收口：find_spec 对 HCU 的 flashinfer 桩包恒
    # True，is_cuda()=True 后上游会信以为真并裸 import（token_dispatcher/
    # flashinfer_utils 无 except 兜底）。HCU 路线是 attention 走 fa3/triton，
    # flashinfer 恒不可用。
    def _hcu_fi_available():
        return False

    _common.is_flashinfer_available = _hcu_fi_available

    # torch HIP graph 拒收 external event（"External events are disallowed in
    # rocm"）；上游 shared_read_event.make_external_event 以 is_cuda 判定后调
    # Event(external=True)。HCU 硬件真值= HIP 无此能力，把上游的构造口替换
    # 为普通 Event（语义降级：外部读端 fastpath 关闭，与 is_cuda()=False 时
    # 上游行为一致——返回 None 正是它自己的降级臂）。
    try:
        from sglang.srt.model_executor.runner_utils import (
            shared_read_event as _sre,
        )

        def _hcu_make_external_event(device_module):
            return None

        _sre.make_external_event = _hcu_make_external_event
    except Exception:
        pass

    _common.is_cuda = _hcu_is_cuda
    _common.is_hip = _hcu_is_hip
    _common.is_cuda_alike = _hcu_is_cuda_alike
    _pf._is_cuda_available = _hcu_pf_cuda_available
    _pf._is_rocm_available = _hcu_pf_rocm_available

    # CUDA-版本特性门（is_ampere_with_cuda_12_3 / is_hopper_with_cuda_12_3 /
    # is_blackwell / is_sm100_supported / ...）：import 时已 lru_cache(
    # partial(_check_cuda_device_version,...)) 冻结，且原实现会拿
    # torch.version.cuda.split —— HCU 上是 None，直接 AttributeError。
    # 语义上这些门判定的是"NVIDIA 特定版本特性"，HCU 不该放行：统一 False，
    # 上游走保守回退路径。按 __wrapped__ 精确识别，未来新增探针自动覆盖。
    import functools as _ft

    for _name in list(vars(_common)):
        _v = getattr(_common, _name)
        _inner = getattr(_v, "__wrapped__", None)
        if (
            isinstance(_inner, _ft.partial)
            and _inner.func is getattr(_common, "_check_cuda_device_version", None)
        ):
            setattr(_common, _name, lambda: False)

    # gcn 架构探针（is_gfx950_supported / is_gfx1250_supported，@lru_cache 无
    # __wrapped__）：原实现走 torch.version.hip 真值分支后裸调
    # get_device_properties(0) ——卡分区可见性（HIP_VISIBLE_DEVICES=8..15）
    # 且 CUDA 尚未 init 的 sitecustomize 早期 import 顺序下直接
    # "No CUDA GPUs are available"。HCU 是 gfx93x，这些门恒 False。
    _install_gfx_probe_guard(_common)

    impls = {
        "sglang.srt.utils.common:is_cuda": _hcu_is_cuda,
        "sglang.srt.utils.common:is_hip": _hcu_is_hip,
        "sglang.srt.utils.common:is_cuda_alike": _hcu_is_cuda_alike,
        "sglang.srt.utils.common:is_flashinfer_available": _hcu_fi_available,
        "sglang.srt.platforms:_is_cuda_available": _hcu_pf_cuda_available,
        "sglang.srt.platforms:_is_rocm_available": _hcu_pf_rocm_available,
        "sglang.srt.model_executor.runner_utils.shared_read_event:make_external_event": _hcu_make_external_event,
    }
    for target in _TARGETS:
        registry.register(
            "platform_detect",
            impls[target],
            target=target,
            mechanism=registry.MECH_SEAM,
            detail=f"semantics={sem}",
        )

    # `sglang.srt.utils/__init__` does `from .common import *` when IT is
    # first imported -- which happens as a side effect of our very own
    # `from sglang.srt.utils import common` above. The package namespace
    # therefore still holds the ORIGINAL callables; sync them (same for any
    # other already-imported re-export layer).
    import sglang.srt.utils as _pkg_utils

    _sync = {
        "is_cuda": _hcu_is_cuda,
        "is_hip": _hcu_is_hip,
        "is_cuda_alike": _hcu_is_cuda_alike,
        "is_flashinfer_available": _hcu_fi_available,
    }
    for _n, _fn in _sync.items():
        if getattr(_pkg_utils, _n, None) is not _fn:
            setattr(_pkg_utils, _n, _fn)
    LOG.warning(
        "HCU adapter: platform semantics seam installed "
        "(sglang answers CUDA; hw layers stay HIP-truthful)"
    )
