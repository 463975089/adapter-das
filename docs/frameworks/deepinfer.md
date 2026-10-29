# DeepInfer bridge 操作手册

**首版范围：DeepInfer 上游接口的 HCU 适配接入。** 详细边界见[提交范围](deepinfer-scope.md)。

本入口使用共用 adapter 的 bootstrap、环境注册表、catalog 和运行时 registry。DeepInfer 的导入别名、factory、metadata 和 worker 生命周期位于 `hcu_adapter/bridges/_deepinfer/`；张量实现位于 `hcu_adapter/kernels/`。新增代码遵循本仓 [治理](../governance.md) 与开发 skill。

算子分组、档位文件及 plugin 兼容入口见[算子文件导航](deepinfer-kernels.md)。

## 源码接入

本次迁移基于 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。需要兼容的 HIP Torch、Triton、LightOp、DeepEP/DeepGEMM 等外部库和配套 `deepinfer._C`；仓库不包含权重或二进制。迁移后的整模型与 Graph 回归尚待节点验证，不把历史结果当作本版本验收。

```bash
: "${ADAPTER:?设置共用 adapter 根目录}"
: "${OFFICIAL:?设置配套 DeepInfer 源码根目录}"
export HCU_ADAPTER_HOST=deepinfer
export PYTHONPATH="$ADAPTER:$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"
export HCU_CONFIG=/absolute/path/deepinfer.json
```

多框架环境或通用 Python 启动器显式选择宿主，避免自动探测选到其他框架。`sitecustomize` 仅安装轻量 worker hook；重依赖在 engine 导入时、设备范围设置后安装。显式安装可使用：

```python
from hcu_adapter.bridges.deepinfer import install_runtime
install_runtime(artifacts="/absolute/path/native-artifacts",
                physical_device_count=8, allow_cuda_graph=True)
from deepinfer.entrypoints.llm import LLM
```

每个 worker 先设置实际 CUDA/HIP 可见性，再导入 engine。显式参数优先于 `hosts.deepinfer` 配置；每算子 `ops` 选择优先于默认 backend。重复安装需相同参数；安装失败应重新启动进程，不能继续使用半安装状态。

## 配置

JSON 示例（YAML 同结构，需 PyYAML）：

```json
{
  "version": 1,
  "hosts": {
    "deepinfer": {
      "artifacts": "./native-artifacts",
      "physical_device_count": 8,
      "allow_cuda_graph": true,
      "tilelang_backend": "triton",
      "grouped_gemm_backend": "triton"
    }
  },
  "ops": {"tilelang_ops.norm_tl": "triton"}
}
```

`artifacts` 相对配置文件所在目录解析；`physical_device_count` 为正整数，`allow_cuda_graph` 为布尔值，两项 backend 可为 `triton`/`torch`。旧独立 adapter 的顶层 `runtime` 配置迁移到 `hosts.deepinfer`；不同时加载两份 adapter。公共平台语义为 `HCU_PLATFORM_SEMANTICS=cuda`，DeepInfer bridge 内翻译为其 GPU 语义。

`ops` 只允许 catalog 中可选择的实现。固定接线入口若尚无可切换实现，选择不匹配会报错，不会悄悄忽略。SGLang 与 DeepInfer 同名算子的量化、布局或返回契约可能不同，目录中保留不同契约条目；共用 registry 并不意味着可任意互换所有实现。

## 原生扩展与诊断

在配套 HIP 构建环境中执行以下工具，产物留在忽略目录；也可复用与现有部署匹配的产物：

```bash
python3 tools/build_deepinfer_extension.py --upstream "$OFFICIAL" \
  --workdir "$BUILD_DIR" --output "$NATIVE_ARTIFACTS" --toolchain "$TOOLCHAIN_ROOT"
```

`TOOLCHAIN_ROOT` 需要包含 `usr/bin/gcc-13`、`usr/bin/g++-13` 和对应运行库；还需当前 DTK/PyTorch 的编译依赖。工具在隔离 staging 中构建固定上游 `_C`，不修改上游工作树。`HCU_ADAPTER_DUMP` 支持 `1` 输出 stdout 或带 `{pid}` 的文件路径。无 `{pid}` 的 DeepInfer 路径自动添加 PID。初始 dump 是 adapter 安装时快照；plugin 安装后需再读取 `hcu_adapter.ops.registry.dump()` 或 plugin 报告中的覆盖记录。

`HCU_DISABLE_OPS` 支持具体算子或 `deep_gemm`、`deep_ep`、`worker.runtime` 前缀组；DeepInfer 必需路径被禁用时会拒绝安装，不会静默换成语义不明的实现。

## Shell 入口

配置示例位于 `examples/deepinfer/config.json`。将配置保存到部署目录，并把
`artifacts` 改为实际扩展产物路径；相对路径以配置文件目录为准。随后运行：

```bash
DEEPINFER_SOURCE="$OFFICIAL" bash examples/deepinfer/run.sh \
  /absolute/path/deepinfer.json python3 /absolute/path/upstream-example.py
```

脚本只选择宿主、配置和源码搜索路径，原样执行指定的上游命令；不会安装模型 plugin。
命令和模型参数取自配套 DeepInfer 版本的使用说明。

## 验证状态

本分支从原有 DeepInfer 适配工作区迁入，重新对接 adapter-das 的统一包结构。
本次静态检查不能替代节点验收。旧版 adapter + plugin 的 DSpark、精度与性能记录
不作为本分支的通过结论。适配实现来自原 DeepInfer adapter 工作区（基线 `c2f659f`），
重新对接 adapter-das `f3bf30c`；归属、限制及迁移条件见[首版范围](deepinfer-scope.md)。

模型量化加载、特定 KV 布局、bounded replay 和模型策略继续由独立 plugin 组合。
当前目录不包含 channel-FP8 模型插件，也不将它的模型策略设为其他框架的默认行为。
