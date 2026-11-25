# 非分页 DTK FlashAttention 桥

本页定义 adapter 的非分页 FA2/FA3 接口、支持范围与回退约定。
`HCU_FA_IMPL=vendor` 显式选择 DTK flash_attn2.8.4；已验证 gfx936/DTK26.10
的 BF16/FP16 dense/packed 路径，`auto` 保持 Triton。分页缺口仍有效。

- 4D BLHD、CU=None 时按 batch 构造独立 CU；3D packed 保留原 CU。
- FA2 三元返回归一为 `(out, LSE)`；packed LSE 为 TH，dense 为 BHL，
  保持外部 strided out 的对象、存储与 guard。
- 更新公开 dispatcher 的 from-import 缓存，覆盖预导入的真实 diffusion 消费者。
- FA3-only 参数不得静默丢弃，未支持的非分页接口交回原 emulation；
  不捕获 vendor 运行错误后重试。
- vendor 桥只接受已声明关键字，以及中性的 `alibi_slopes=None`、`dropout_p=0`；
  非空 ALiBi 或非零 dropout 显式报 `NotImplementedError`，其他未声明关键字
  报 `TypeError`。原 emulation 会忽略这些 FA2 参数，不能以回退代替拒绝。

新进程使用 `HCU_FA_IMPL=triton` 回退；快关与优先级统一见
[快关注册表](../skills/hcu-adapter-dev/references/kill-switches.md#hcu_disable_ops快关总闸)。
设备测试入口为 `tests/kernels/test_fa_dense_batch.py` 与
`tests/vendor/test_flash_attn.py`，CPU 参数路由回归见
`tests/adapter/test_fa_vendor_kwargs.py`，回归约束见
[测试参考](../skills/hcu-adapter-dev/references/test-suite.md#非分页-diffusion-接口回归)。
模型启动、性能表、图像复现限制及证据目录统一见
[MiniMax-H3 部署第8节](image-dtk2610.md#8-minimax-h3-image-的原生-diffusion-平台候选)。

## SGLang Diffusion FA / FA2 后端

gfx936 上的 HCU diffusion 平台在 DTK FA2 可用时按调用方请求返回对应后端。
`FA` 保留上游 FA 标识、dense、Ring 和 Skip Softmax 接口，常规 packed varlen
通过 DTK `flash_attn_varlen_func` 执行。显式 `FA2` 用 DTK
`flash_attn_func` 执行 dense，并与 `FA` 共享连续 packed varlen 路径。
该路径以 sglang-das `d76d3f185d` 为行为基线，packed Q/K/V 和设备上的
int32 序列边界使用连续布局，切分边界和最大长度由调用方提供。

该路径需要真实 DTK wheel，支持 BF16/FP16。DTK FA2 不可用或设置
`HCU_DISABLE_OPS=vendor_fa` 时，FA 交给上游 resolver，显式 FA2 报错。
已有 FA3 桥仍通过 `HCU_FA_IMPL` 选择 provider。
