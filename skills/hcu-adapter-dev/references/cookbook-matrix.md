# sglang 官方 cookbook（NVIDIA 参数）直接支持：降级矩阵

目标：用户完全按 sglang 官方 cookbook 的 NVIDIA 参数启动，不追求高性能，
adapter 保证**能直接跑且结果正确**。评估口径分四层：

## A. 可直接透传（无兼容工作）

TP/EP/PP、`--mem-fraction-static`、`--chunked-prefill-size`、
`--max-running-requests`、调度类、`--cuda-graph-max-bs`、采样参数、
`--json-model-override-args`（截层等）。**已验证**：W4A8 GLM-5 TP8 +
CUDA graph ON + 零旗标 e2e 连贯（短/长 prompt 双变体）。

## B. 参数在、后端换（垫片内部消化，用户无感）

| cookbook 写法 | HCU 实际路径 | 状态 |
|---|---|---|
| `--attention-backend flashinfer/fa3` | triton（或未来 vendor 档） | 需在 server_args 层提示性降级（规划：sitecustomize 感知后打 warning；行为已可用） |
| speculative NVIDIA kernel 路径 | triton/torch 等价或禁用 | 按需接入 |
| fp8/fp4 量化路径 | gfx938 物理有 fp8；算子档替换 | W4A8 已走通此模式（量化差异在算子内消化） |

## C. 物理差异，需显式处理

- CUDA graph 本身是**独立质量属性**：不是"开了就能用"，需过 graph-safety
  套件（AR 教训）。当前默认开且已验证。
- NCCL "multimem all-gather disabled" 类探测日志 = 降级提示，非错误。
- aiter AR 在 gfx93* 默认拒绝（快关自动规则），cookbook 不需要用户改任何
  参数。

## D. 暂不支持（诚实清单）

- NVIDIA 专有融合（如 flashinfer 融合 AR 的某些 fp8 路径）：等 triton
  等价实现或明确报错指引。
- MoE fp8/w4a16 marlin 类视 lightop vendor 档接入进度（shadow-matrix.md
  盘点）。

## 维护规则

- 上游 sglang 升级后：跑 `run_triage.py --stage ops,e2e` 回归；cookbook
  新参数进 B/D 层归档。
- 结论口径：**方向可行**——参数面大部分可透传，差异集中在算子层，而算子
  层正是本仓库的职责边界；风险主要在社区码演进速度，应对 = 锚定 import 面
  + 契约测试门禁。

## E. 新容器初始化规程（混合栈防火墙，黄金法则 #7）

```bash
# 1) 杀掉镜像自带的 das 定制 sglang 与 kernel 包（防混栈：hcu_alloc_decode_kernel
#    这类"奇怪的 ImportError"全部源于此）
SP=/usr/local/lib/python3.10/dist-packages
rm -rf $SP/sglang $SP/sglang-*dist-info \
       $SP/sglang_kernel $SP/sglang_kernel-*dist-info \
       $SP/sgl_kernel $SP/sgl_kernel-*dist-info 2>/dev/null
# 2) 眼见为实：官方源 + adapter 解析正确、版本入日志
PYTHONPATH=<adapter>:<official-sglang>/python python - <<'P'
import sglang, sgl_kernel
print("sglang:", sglang.__version__, sglang.__file__)   # 必须是官方源路径
print("adapter:", sgl_kernel.__version__, sgl_kernel.__file__)
P
# 3) 发射命令里固定 PYTHONPATH=<adapter>:<official>/python（顺序不可倒）
```
镜像 sglang 是 dtk 定制线（含 hcu_alloc_decode_kernel 等私改），与官方 release
共存时 `sys.path` 顺序稍有变化就会解析到它——删除比 mv.disabled 更彻底。
