---
name: hcu-adapter-dev
description: hcu-sglang-adapter 仓库开发规约：算子垫片/影子库新增、三档实现策略、自动化测试要求、高危算子快关与精度定位流程、sglang cookbook (NVIDIA 参数) 直接支持的降级矩阵。改这个仓库的任何代码前先读本 skill。
---

# hcu-sglang-adapter 开发规约

不修改 sglang 官方代码，让社区版 sglang 直接跑在海光 HCU（DCU/DTK/HIP）上。
本仓库的价值主张：**上游 import 什么算子名，我们就提供什么算子名，内部可
路由到 triton / vendor / torch 三档实现，并对全部出口 API 做统一质量验证**。
"adapter 测试全绿 ⇒ 官方 sglang 无大问题" 是本仓库的验收标准。

## 黄金法则（历史 bug 换来的，违反必踩坑）

1. **同名 ≠ 同语义**：每个垫片必须对拍参考实现（cos/rel 断言），不能只测
   "能跑不崩"。案例：rope GPT-J cache 恒为半分 [cos|sin] 布局（fbc540c）。
2. **eager 对 ≠ graph 对**：凡进 decode 链路的算子必须过 capture-replay
   变输入探针。案例：AiterCustomAllreduce 图内重放损坏（1eb038d）。
3. **GPU 路径禁主机同步**：`.item()`/`nonzero`/布尔掩码索引会毁掉 CUDA
   graph capture。
4. **高危 vendor 算子默认不信**：custom all-reduce、flash attention 这类
   "size 灵活、场景多" 的 vendor kernel，接入即配压测 + 快关。
5. **import 永不失败**：未覆盖算子在**调用时**抛带指引的 NotImplementedError。

## 任务路由（做某类事之前先读对应参考文档）

| 你要做的事 | 先读 |
|---|---|
| 新增/修改一个算子垫片或影子库 | `references/shadow-matrix.md` |
| 写或改任何 kernel / 测试 | `references/test-suite.md` |
| 加一个快关（kill switch） | `references/kill-switches.md` |
| 排查精度问题（输出垃圾/分叉/漂移） | `references/precision-triage.md` |
| 评估 sglang cookbook NVIDIA 参数能否直接用 | `references/cookbook-matrix.md` |

## 仓库结构

```
sgl_kernel/            同名影子包（sglang 的 AOT 默认内核路径）
  _adapters/           元路径影子库：aiter / flashinfer / flash_attn 填空+守卫
  _adapters/kill_switch.py   HCU_DISABLE_OPS 快关注册表
  elementwise/layernorm/rotary/moe/kvcacheio.py   triton/torch 实现
sitecustomize.py       HIP 桥接线（vllm._custom_ops、AR gate、DSA、调试设施）
tests/
  test_gpu_kernels.py / test_ops_cpu.py   sgl_kernel 自有算子契约测试
  test_vendor_allreduce.py / test_flash_attn_gpu.py  高危 vendor 压测
  _*_probe.py          排查探针（手动，路径可用 HCU_ADAPTER_ROOT/HCU_CKPT 覆盖）
  run_triage.py        精度问题四级快速定位入口
skills/hcu-adapter-dev/  本 skill
```

## 协作规约

- `tests/test_*.py` 是自动化门禁（pytest 可跑、断言带数值谓词）；
  `tests/_*.py` 是排查工具（手动、可硬编码路径但须支持 env 覆盖）。
- 改 kernel 必须附带或更新对应契约测试；涉及 vendor 高危面的同时更新压测
  与快关注册表，三件一起提交。
- push 前最低标准：`pytest tests/test_ops_cpu.py`（任何机器）+
  节点上 `pytest tests/test_gpu_kernels.py` + `python tests/run_triage.py
  --stage ops`。
- commit 信息带根因短句（参考 fbc540c / 1eb038d 风格），便于回溯 bug 类。
- 文档与代码同库同 PR：改行为必须同步改本 skill 对应参考文档。
