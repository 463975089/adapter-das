# GLM-5.3 CT W4A8 P/D 配方

本目录是四节点候选配方，正在验证，尚不能作为完整服务验收记录。
底座固定为 `HYGON-AI/sglang-das` 的 `v0.5.22.plugin`，提交
`15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`；adapter 镜像为
`10.16.1.152:5000/jenkins/model_test_env/vllm:0.28.1rc1-ubuntu22.04-dtk2610-py3.12-20261103-1558`。

每节点 8 卡。prefill 使用两节点 PP2/TP8/CP8/EP8、DeepEP normal，
decode 使用另两节点 TP16/DP16/EP16、DeepEP low_latency、EAGLE 5/1/6、
decode graph max batch 16。decode 的两节点规模是本任务的配置，
不同于来源部署文档的四节点 TP32。prefill 的 PP2 不启用 MTP，graph 关闭。
Prefill 显式设置 `SGLANG_ENABLE_CP_V2=1`，与原生 GLM 的策略 CP 及每 rank
物理补齐方式一致。旧 CP 的 32-token 请求曾在首层 attention 出口出现部分行全零；
两组 CP v2 短前缀对照恢复，但完整数值精度仍在验证。

先配置宿主代码目录和 checkpoint，再在每个角色的两台节点设置对应的
`NODE_HOST`、`NODE_RANK=0|1`、`DIST_INIT_HOST`。同一角色必须共用
`DIST_INIT_HOST` 和 `DIST_PORT`。以下路径和地址均需替换为实际部署值：

```bash
export ADAPTER_DIR=/path/to/adapter-das
export PLUGIN_DIR=/path/to/sglang-plugin-das
export SGLANG_DIR=/path/to/sglang-das
export MODEL_PATH=/models/GLM-5.3-CHANNEL-INT4-w4a8-MTP-int4
export NODE_HOST="<this-node-ip>"
export NODE_RANK=0
export DIST_INIT_HOST="<role-node0-ip>"
export PORT=30861 DIST_PORT=30961 BOOTSTRAP_PORT=30969
bash "$ADAPTER_DIR/examples/glm5.3-w4a8/pd/prefill.sh"
```

decode 两节点改用各自的地址、rank，设置 `PORT=30862 DIST_PORT=30962`，
运行 `decode.sh`。模型服务应分别就绪后再启动 P/D router；本目录不自动
申请节点、停止已有服务或修改容器依赖。

脚本保留调用者的 `PYTHONPATH`，可在启动前选择已验证的隔离 FlashMLA 包。
当前节点还需要配套的 SHCA Mooncake/verbs/provider/net plugin；版本、
哈希、原安装备份与验证范围见
[通信记录](../../../docs/glm5.3-w4a8-network.json) 和
[开发记录](../../../docs/glm5.3-w4a8-development.md)。
将隔离包目录放入 `PYTHONPATH`、配套动态库目录放入 `LD_LIBRARY_PATH`，
并设置 `IBV_DRIVERS=shca`、`CPATH` 指向匹配的 SDK；保留上游 ABI 检查，
不要同时暴露通用和 SHCA 两个 Mooncake distribution。
当前 SHCA 网络配置可设置 `NCCL_NET="NCCL RDMA Plugin v8"`，
使初始化失败时不回退到 socket 网络。不能将这些节点的依赖布局直接视为其他
SHCA/DTK 环境的通用安装方案。

两种角色均使用 `--quantization compressed-tensors` 和 `deep_gemm` runner，
保留专家 W4、动态 token INT8 激活以及普通 Linear 的 W8A8/ignore 规则。
`HCU_W4A8_MOE=native`、`HCU_SLIMQUANT=0`、
`SGLANG_OPT_USE_TOPK_V2=0` 固定在配方中；不额外指定
`--dsa-topk-backend torch`。Ninja 使用镜像已有 wheel 内的原生二进制，
Triton/Torch 编译缓存默认在各容器本地。

配方通过 plugin 的 `SGLANG_HCU_GLM_CT_BF16_ROUTER=1` 选择原生 BF16
gate 投影，通过 `SGLANG_LIGHTOP_TOPK=1` 选择已有的公共 LightOp 专家路由。
前者只覆盖 GLM DSA compressed-tensors 的 BF16 gate，FP32 权重优先；
后者在可用的静态 bias、归一化分组路由分支生效。二者在 plugin 中默认关闭，
需使用包含对应支持的 plugin 版本。组件对照与完整模型验收分别记账，见
[MoE 对齐记录](../../../docs/glm5.3-w4a8-moe-alignment.json)。

指定底座及 plugin 已能识别 DeepEP 的 vendor `quant_type` API；配方设置
`HCU_DEEPEP_BRIDGE=0`，关闭面向旧 SGLang 的 `use_fp8` wrapper，
避免其隐藏原生签名并将 W4A8 INT8 误判为 FP8。CT 配置、checkpoint
名称和格式继续保留，合格的 channel W4A8 DeepEP 专家复用底座的
`HCUW4A8Int8DeepEPMoEMethod`，不通过改变模型的量化标签选择路径。

当前 gfx936 节点的连续 W4A8 HIPC 需要 COV5。使用固定 DeepGEMM
提交 `b061ec70272a001fca622ea814d3ee163e24bb05` 的源码，在同一镜像中执行：

```bash
HCU_ADAPTER_HOST=none python3 "$ADAPTER_DIR/tools/build_deepgemm_w4a8_cov5.py" \
    --source "$DEEPGEMM_SOURCE" --output "$COV5_OUTPUT"
export PYTHONPATH="$COV5_OUTPUT${PYTHONPATH:+:$PYTHONPATH}"
```

`DEEPGEMM_SOURCE` 指向上述源码，`COV5_OUTPUT` 必须是源码树外的新目录。
构建器校验实际编译的 vendor 源文件和头文件字节，生成独立模块并原子发布，
不写回原源码或安装包。启动时仅在该模块可导入时接入，锁定镜像 DeepGEMM
完整版本及 Python wrapper 哈希，保留公共函数、校验和缓存别名。仅该 wrapper
的私有默认 mode 变为原 HIPC 1002，显式 mode 1006 拒绝；其他算子及共享
vendor 配置保持原样。启动日志记录实际模块路径，registry 记录
`w4a8_contiguous_gemm`。移除该目录或在新进程设置
`HCU_DISABLE_OPS=w4a8_contiguous_gemm` 可跳过接入；恢复原路径不代表
当前节点的原 COV6 算子已经可用。

同一镜像的 Aiter custom all-reduce 也需使用 COV5：原 COV6 模块在这些
节点上没有写入输出。相同源码的独立 COV5 构建已通过 8-rank 实际 embedding
分片的精确求和及变输入 graph 检查，完整模型精度尚未通过。

```bash
HCU_ADAPTER_HOST=none python3 "$ADAPTER_DIR/tools/build_aiter_car_cov5.py" \
    --output-dir "$AITER_CAR_COV5_OUTPUT"
export AITER_JIT_DIR="$AITER_CAR_COV5_OUTPUT/jit"
```

`AITER_CAR_COV5_OUTPUT` 必须是新目录。构建器锁定镜像 Aiter 的版本、
402 份源码/配置摘要和原二进制，通过其公开 JIT/COV 参数只重建 CAR；
其他模块引用同镜像安装文件。各节点必须使用相同的镜像资产。
仅设置编译选项不会替换已有二进制，启动进程须使用上述 `AITER_JIT_DIR`。
本方案不修改 vendor 或 SGLang 源码，也不增加运行时 wrapper。


配方通过 `HCU_RMS_NORM_IMPL=lightop` 对齐原生 BF16 plain RMSNorm，
通过 `HCU_FUSED_ADD_RMS_IMPL=lightop` 单独选择 residual RMSNorm。
plain 路径的实际输入、输出 buffer 和 Graph 验证范围见
[组件记录](../../../docs/glm5.3-w4a8-plain-rms-alignment.json)；整模型精度仍须独立复测。
