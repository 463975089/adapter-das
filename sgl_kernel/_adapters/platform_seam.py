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


def install():
    import os

    import torch

    from sgl_kernel._adapters import hcu_env, registry

    # sgl-kernel 版本门（engine.py assert_pkg_version）防的是陈旧 AOT wheel；
    # 影子库路线下 kernel 面由本仓兜底，dist-info 版本无意义。走上游声明
    # 的逃生 seam，自动置位不落部署 env（环境三原则）。
    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")

    sem, _src = hcu_env.resolve("HCU_PLATFORM_SEMANTICS")
    if sem == "hip":
        LOG.warning(
            "HCU adapter: platform semantics=hip (legacy adapter route;"
            " set HCU_PLATFORM_SEMANTICS=cuda for the §0.5 is_cuda route)"
        )
        return
    if sem != "cuda":
        raise ValueError(
            f"HCU_PLATFORM_SEMANTICS={sem!r} invalid (cuda|hip)"
        )
    if not torch.cuda.is_available():
        return

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
