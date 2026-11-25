# tier 处置表（emu 退出运行时 · 零降级）— 活文档

> 依据 governance §3b-8/9（2026-11-19）。本表是 emu 清零与降级点拆除的
> 司法台账：每完成一项把处置改为完成态并留 commit 锚点。
> 分类：**A** = 有兄弟档（fork/triton/vendor/cpp），只需拆运行时降级（W2）；
> **B** = emu 是唯一实现 → 补 triton 或寻源 vendor（W3 核心）；
> **C** = 纯开发/诊断参考 → 移出运行时路由，保留 dev 工具。
> 状态：`待处置` → `进行中` → `完成(档位/commit)`。

| # | emu 实现 | 兄弟档 | 分类 | 处置 | 状态 |
|---|---|---|---|---|---|
| 1 | `hcu_adapter/bridges/sglang_mqa_emu.py` | — | B(补triton/寻源vendor) | 改判：桥本体是 triton 面装载器（HCU_MQA_EMU_IMPL=triton 默认），运行时 fallback 已拆 | **完成**（W2-2） |
| 2 | `hcu_adapter/features/dsa/dsa_mqa_logits_emu.py` | `mqa_logits_triton.py`（根名不同未被台账兄弟列匹配）+ lightop `gemmopt.paged_mqa_logits` | B(补triton/寻源vendor) | **vendor 寻源（lightop paged 臂已在文件内，:191）+ triton 修复**：triton 档存在但被 DTK div-4 int64 load bug 打错（`mqa_logits_triton.py` 文档自证 deterministically wrong），本件是带修复的生产正确路径（lightop 臂+torch 兜底，多 except 梯子=§3b-8 违规点，随档位落地一并严格化） | **任务开立**（W3-4：lightop paged 数值对拍 → 接 vendor 档；或 triton 绕过 div-4；落地前本件维持生产正确性不判死） |
| 3 | `hcu_adapter/features/dsa/kpool_write_emu.py` | 上游 kpool writer（SetKAndS 族） | B(补triton/寻源vendor) | **开发 triton（div-4 绕过）**：DTK triton 对 int64 标量 load 的 div-4 编错（g53-tp8 GPU4 2026-10-15 实证：scales 落点全偏 4 倍→index-K cache 全零→GLM-5.3 首 token 平线）；本件=torch 正确性规避，deep_gemm 桥 :162 在产 | **任务开立**（W3-4：triton 修复版（参照 causal_conv1d 的同类 read-dead 修复模式）+ 本件手工对拍；落地前维持生产） |
| 4 | `hcu_adapter/features/dsa/mqa_logits_emu.py` | `hcu_adapter/features/dsa/mqa_logits_triton.py`, `hcu_adapter/kernels/mqa_logits_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 5 | `hcu_adapter/features/gdn/conv1d_fn_emu.py` | fla fork（`fla_hcu/conv1d_triton.py`，根名差异未被兄弟列匹配） | B(补triton/寻源vendor) | **triton 修复任务**：DTK 对 `_causal_conv1d_fwd_kernel` 的 `has_initial_state` 分支编错（gfx938，docstring 自证 stopgap）；conv1d 运行时=fork 档（W2-5 已拒 emu 路由），本件仅剩 varlen+初始态预填充臂在生产兜底 | **任务开立**（W3-5：修复后此臂切回 fork，emu 转纯开发参考） |
| 6 | `hcu_adapter/features/gdn/conv1d_update_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 7 | `hcu_adapter/features/gdn/kda_prefill_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 8 | `hcu_adapter/features/hadamard/hadamard_emu.py` | `hcu_adapter/features/hadamard/hadamard.py`, `hcu_adapter/features/hadamard/hadamard_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 9 | `hcu_adapter/kernels/compress_emu.py` | — | B(补triton/寻源vendor) | **开发 triton**（真数值热路径：窗口 FP32 softmax + 加权 latent 求和 + 环形写回，每步执行；deepinfer 的 tilelang CUDA 核在 HCU 不存在——tilelang_ops 的 backend 即本仓提供者清单，当前 launch 体直跑 emu） | **任务开立**（W3-3；triton 实现 + emu 手工对拍后切档） |
| 10 | `hcu_adapter/kernels/deepselect_emu.py` | `hcu_adapter/bridges/_deepinfer/deepselect.py`, `hcu_adapter/kernels/deepselect_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 11 | `hcu_adapter/kernels/engram_init.py`（原 _emu） | — | B(补triton/寻源vendor) | **机制化改名**（装载期一次性 norm 权重 FP32 融合 + offsets 构造；同 weight_scale 类，无每步数值语义） | **完成**（W3-5） |
| 12 | `hcu_adapter/kernels/engram_runtime_emu.py` | — | B(补triton/寻源vendor) | **开发 triton**（每步数值面：engram_gate_fwd 的 FP32 norm/dot/signed-sqrt 门控 + 整数 hash 表映射；tile_kernels :81 在挂） | **任务开立**（W3-5；低频面可先确认使用率再排优先级） |
| 13 | `hcu_adapter/kernels/gather_emu.py` | `hcu_adapter/kernels/gather.py`, `hcu_adapter/kernels/gather_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 14 | `hcu_adapter/kernels/gemm_emu.py` | `hcu_adapter/kernels/gemm.py`, `hcu_adapter/kernels/gemm_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 15 | `hcu_adapter/kernels/grouped_gemm_emu.py` | `hcu_adapter/kernels/grouped_gemm_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 16 | `hcu_adapter/kernels/index_ops_emu.py` | `hcu_adapter/kernels/index_ops.py`, `hcu_adapter/kernels/index_ops_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 17 | `hcu_adapter/kernels/kv_cache_emu.py` | 上游 `kernels/ops/kvcache/` 家族 | B(补triton/寻源vendor) | **接上游 triton 档**（寻源 2026-11-19：aiter 无 paged KV set/get 原语，仅有 rope_cached*/ep_scatter/quant 族；上游树内已有 cache_ops.py、triton_store_cache.py、fused_fp8_qkv_kv_cache.py、kv_read_table.py、copy/zero_pages、hicache/hisparse——覆盖 load_paged/store_paged 的 gather/quant/scatter 语义）| **寻源完成**；接线任务：DeepInfer fused_transform/kv_cache 消费面换上游调用 + GPU 对拍（W3-1） |
| 18 | `hcu_adapter/kernels/mhc_emu.py` | vllm_dsv41/mhc + mhc_aiter + mhc_lightop | A(拆降级) | emu 件实为**分发器**（:242 即转发 mhc_lightop.hc_projection；HCU_MHC_IMPL=aiter 默认）——vendor 档是运行时；内部 torch 臂梯子随 W2 enforcement 批严格化 | **完成**（W3-6 核实；梯子严格化归 enforcement 批统一处理） |
| 19 | `hcu_adapter/kernels/mla_attention_emu.py` | `hcu_adapter/kernels/mla_attention_dtk.py` | A(拆降级) | 待处置 | 待处置 |
| 20 | `hcu_adapter/kernels/mla_layout.py`（原 mla_layout_emu） | — | B(补triton/寻源vendor) | **机制化改名**（寻源 2026-11-19：上游 0522 与 aiter 均无等价算子；本件是权重装载期一次性字节保持排列——torch 跨步拷贝即 HBM 带宽级，triton 无收益，无仿真语义无降级梯——`_emu` 后缀属命名错位） | **完成**（W3-2：改名+豁免清单+消费点重接；无运行时行为变化） |
| 21 | `hcu_adapter/kernels/moe_gate_emu.py` | `hcu_adapter/kernels/moe_gate_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 22 | `hcu_adapter/kernels/mtp_emu.py` | `hcu_adapter/kernels/mtp_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 23 | `hcu_adapter/kernels/norm_emu.py` | `hcu_adapter/kernels/norm.py`, `hcu_adapter/kernels/norm_lightop.py`, `hcu_adapter/kernels/norm_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 24 | `hcu_adapter/kernels/quant_emu.py` | `hcu_adapter/features/aiter/quant_aiter.py`, `hcu_adapter/kernels/quant_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 25 | `hcu_adapter/kernels/request_indices_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 26 | `hcu_adapter/kernels/rope_emu.py` | `hcu_adapter/kernels/rope_lightop.py`, `hcu_adapter/kernels/rope_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 27 | `hcu_adapter/kernels/sampling_emu.py` | `bridges/_sglang/sampling.py`（top_k/top_p renorm 面，**非同面**——ZOS fused_sample/random_sampling 无覆盖） | 改判 B | **开发 triton**（ZOS 采样数值面：fused_sample/random_sampling，deepinfer 774ce24 锚，device-only；上游参考与 tilelang 核允许数值差——属性测试非位级） | **任务开立**（W3-5） |
| 28 | `hcu_adapter/kernels/sglang_mhc_prenorm_emu.py` | — | B(补triton/寻源vendor) | **改判 vendor 路由面**：实现=全 FP32 torch GEMM（hipBLAS 经 torch.matmul）——DeepGEMM split-K 合约的 HCU 承接，GEMM 本体已在 vendor 库；`_emu` 命名低估了它 | **完成**（W3-6 改判；如需去 dtk/hipblas 直连消除 torch 层再开任务，当前性能路径成立） |
| 29 | `hcu_adapter/kernels/state_and_constraints_emu.py` | `hcu_adapter/kernels/state_and_constraints_triton.py` | A(拆降级) | triton 档已在运行时接线（tilelang_ops:632 mask_logits）；emu=开发参考（§3b-9） | **完成**（W3-3 核实，无需改动） |
| 30 | `hcu_adapter/kernels/verification_emu.py` | — | B(补triton/寻源vendor) | **开发 triton**（ZOS 验证元数据六件套 + sum_smaller_probs 归约；:445-484 全量在挂；与 sampling 同族同优先级——投机解码热路径） | **任务开立**（W3-6） |
| 31 | `hcu_adapter/kernels/vision_attention_emu.py` | bridges/_deepinfer/vision_attention.py（**调用面**非实现档，兄弟列误配） | 改判 B | **开发 triton**（有界内存 FP32 varlen ViT attention 整体在 torch——真数值面） | **任务开立**（W3-6；ViT 路径使用频率先确认再排期） |
| 32 | `hcu_adapter/kernels/vision_emu.py` | `hcu_adapter/kernels/vision_triton.py` | A(拆降级) | rgb_pad_to_patches → **triton 档已接线**（tilelang_ops:668）；replace_input_embeds（:656）= 嵌入替换机制臂（装载/预处理变换，mla_layout 类语义） | **完成**（W3-6 核实；replace 臂如需彻底去 _emu 名可随下批机制改名） |
| 33 | `hcu_adapter/kernels/weight_scale.py`（原 _emu） | — | B(补triton/寻源vendor) | **机制化改名**（装载期 UE8M0 scale ABI 布局，tile-kernels 2.1.0 GPU ABI 对齐，docstring 自证"loading-time candidate, not a GEMM implementation"；无数值语义无降级梯） | **完成**（W3-3） |
| 34 | `hcu_adapter/kernels/zos_metadata_emu.py` | `hcu_adapter/kernels/zos_metadata_triton.py` | A(拆降级) | triton 档已在运行时接线（tilelang_ops:361-406 四函数）；emu=开发参考（§3b-9）；W3-3 曾误判改名已即时撤销 | **完成**（W3-3 核实，无需改动） |

## 运行时降级点拆除台账（W2，governance §3b-8 违规清单）

| 降级点 | 位置 | 处置 | 状态 |
|---|---|---|---|
| shadow 三层梯子 try-descent | `sgl_kernel/elementwise.py` 等 | 解析期定档，缺实现=启动报错 | **完成**（W2-1 elementwise silu 族；W2-6 批收齐：layernorm rms/fused_add 双解析严格化+分发零降档+aiter one-strike 移除+gemma/dtype 缺口改具名报错、rotary CUDA 无核档即报错、gemm_fp8 ptgq v1/v2 CUDA 无 native 即报错；torch 全部退出三个文件的 CUDA 路径，CPU 契约面保留） |
| aiter one-strike（失败置 None 落 triton） | `sgl_kernel/elementwise.py` | 移除，vendor 错误直曝 | **完成**（W2-6 随 9369626c 梯子批落地：分发处注释锚点 "no one-strike vendor disabling"，vendor 失败直曝、无静默禁用；与上行 shadow 梯子条目的 W2-6 收齐备注一致） |
| aiter_moe 无调优配置→stock triton (log-once) | fused_moe 桥 | 硬报错或显式配置 | **完成**（显式 aiter 拒绝/无配置=RuntimeError；auto=确定性选择保留+log-once；包裹层 except→orig 全量吞错拆除；W2-3） |
| mqa triton 臂 torch-emu fallback | `bridges/sglang_mqa_emu.py` | 拆（emu 退运行时） | **完成**（per-call emu 回落拆除，严格分发；W2-2） |
| HCU_FA3_EMU=1 默认 | env 注册表 | 默认翻转+报错 | **完成**（default=0，docs 同步；W2-2） |
| _auto_mqa_emu auto 落 emu | `infra/hcu_env.py` | auto 不落 emu | **完成**（return None；W2-2） |
| R-12 桥隔离静默走上游 | bridges try-isolation | 配置档失败=首用报错 | **完成**（终态=启动拒绝：HCU_STRICT_BRIDGES=1 默认，桥失败即 RuntimeError 拒绝起服——比 per-op 首用报错更强更简（免逐 topic 毒包装表）；0=审计模式留排障；W5-2 修复 W2-5 的半落地雷：_FAILED_BRIDGES 曾引用未定义） |
| 安装链碎行降噪 | bridges/features 12 处成功行 | warning→info（P7 默认静默） | **完成**（W2-5） |
| emu_parity mark + gate 过滤 | pyproject markers / 三 KDA 巨头 / l1_gpu_gate `-m "not emu_parity"` / ci/emu_parity_nightly.sh | CI 250s→0，手工/nightly 通道保留（W4-1） | **完成** |
| gate 零降级断言 | tests/tools/_test_zero_degradation.py（第 7 并发 lane）：桥装完后 _FAILED_BRIDGES 空 + registry 无 emu 落点 | W5-1 落地 | **完成** |
| 容器验收轮（终验） | **bw47 round-7（2b887b9 树，seed v5）：l0 / cpu / kernels / gates(7道含零降级) / vendor 全 PASS，`GATE_EXIT=0`，JIT_BUILT=0（seed 全域命中，零编译）；kernels 326.37s（1937 passed / 4 xfailed=两处 known-baseline），43 家庭代表例转手工 lane；L1.5 上游套件按 `L15_UPSTREAM_MODE=off` SKIP**。过程轮：round-4（38b4542）l0/cpu/l15_exact/gates/vendor PASS + kernels 396s/3 既有基线 + 6 ownership 断位（eb4bc9e 修）；round-5（d747feb）l15_exact 冷 JIT 17 颗模块仍 PASS（~1680s），kernels 395s 3 失败；round-6（081a141）首次 L1.5 off，暴露家庭代表 43 例在 0.5.22 的 8 个既有失败与长跑用例（基树 759ac7d 同败，已入账并转手工 lane） | 终验完成 | **完成（round-7 定稿）** |
| main 合并轮（MR #152 冲突解决） | main 带入 #133/#137/#138/#139/#143（PCP 注意力族 + FP4 indexer 默认档 + DSpark + CPU fixture）。冲突 4 处（l0_gate 机制集合并集；3 件 import 按 infra/ 约定）+ auto-merge 漏网 4 处旧路径（bridges/vllm.py、test_vllm_fp4_indexer_default ×2、test_pcp_mode_admission、fp4_indexer.py:134 相对导入）。**bw47 merge-2（8d63252 树，复用 round-7 .so + seed v5）：l0 / cpu / kernels / gates(7道) / vendor 全 PASS，`GATE_EXIT=0 wall=1083s`，JIT_BUILT=0，kernels 319.53s（1938 passed / 945 skipped / 4 xfailed）**；首跑（be405c7）唯一失败即相对导入解析门禁（1937 passed / 1 failed），修复后归零——门禁自身抓住了搬家语境陷阱 | 合并完成 | **完成（merge-2）** |
| review 修复轮（#152 AI-review） | 5 条（1×P2 + 4×P3）：P2 aiter_moe skipif 装饰器归位（辅助函数挤占→无 CUDA 环境 error 而非 skip，AST+collect 双验证）；P3 gates echo 动态化（${#gate_names[@]}）；P3 env-vars 表格空行；P3 hip_vec_bf16 两处 `#pragma unoll`→`unroll`（flashinfer 适配件）；P3 fused_add docstring/注册表 torch 候选残留（含 hcu_env.py 同类）。**bw47 容器（f8a6879 树，include/ 变更触发原生全量重建 254s rc=0）：l0 / cpu / kernels / gates(7道) / vendor 全 PASS，`GATE_EXIT=0 wall=1076s`，JIT_BUILT=0；gates echo 实证输出 7 concurrent**（首跑容器 22min 到期于 gates 5/7，复跑续完） | 修复完成 | **完成（GREEN）** |
| main 合并轮 2（#149） | main 带入 #149（HY3 INT8 dispatch 保留 quant_type、breakable prefill graph 改 HIP stream-query 保留图、nextn idle KV metadata、FA E5M2 原子注册）。冲突 3 文件 6 处 + 功能增量落位：bridges/sglang.py 4 处取 HEAD 编排器，#149 增量移入拆分后的家（idle-metadata→sglang_spec_kv.py、quant_type→sglang_deep_ep.py、prefill_bcg 注释→sglang_serve_gates.py）；prefill_bcg_gate.py 取 main 新方案（HIP runtime 查询替代 eager fallback）+ infra import；tier-matrix 再生成；main 带入测试 6 处旧根路径修正。**bw47 merge-4（adcbd86 树）：l0 / cpu / kernels / gates(7道) / vendor 全 PASS，`GATE_EXIT=0 wall=1071s`，JIT_BUILT=0，kernels 325.08s（1936 passed / 945 skipped / 4 xfailed）**；pass-3（49186c3）唯一失败 = test_spec_h2d_pin 第二处 `_install_idle_verify_metadata` 旧家导入（1935 passed / 1 failed），修正落位后归零 | 合并完成 | **完成（merge-4）** |
| fresh_process 搬运 | 9 个 spawn/distributed 测试打标→kernels 段 deselect→gates 第 6 并发 lane（900s 预算，语义不变） | 实测 380s 串行→并发吸收（W4-2） | **完成** |
| packaging 构建复用 | session 级 repocache（树指纹）+ 默认 wheel session 化 | 5 次整仓 copy→1 次（本地实测单副本生效；容器收益随 pip 更大）（W4-3） | **完成** |
| kernels 段 xdist | `-n 4 --dist loadfile`（缺 xdist 显式 WARN；L1_KERNELS_XDIST=0 退避） | **实测 391-396s**（验收轮 3/4，串行基线 1122s → **-65%**；预算已回迁 900s，seed 契约注释更新） | **完成**（W4-3+验收轮实测） |
| GPU compare 助手 | tests/kernels/_gpu_compare.py（GPU 归约+单次同步+失败才拉邻域；bf16-ulp 尺度感知+cosine 模式） | 新标准落地+示范迁移（conv1d_update_emu）；~170 存量点增量迁移（W4-4） | **完成**（标准+示范；存量增量迁移） |
| KDA/conv1d emu 兜底 | gdn 族路由 | 无可用档=报错（gfx938 接受断供） | **完成**（=emu 路由即 RuntimeError，运行时值 fork\|stock；W2-5） |
| P7 启动 tier 表 | `bridges/sglang.py` install 尾部 + `registry.log_summary` | 完成（桥装完后集中打表，一行一算子） | **完成**（W2-4） |
| wizard/validator 拒退役档 | `hcu_config._validate` + wizard | status=retired 选择即报错（3b-9） | **完成**（W2-4） |
| L1 known-baseline（平台/夹具漂移，xfail 非静默） | `test_mla_sparse_decode.py::test_native_decode_graph_fresh_plans[64/128]`、`test_mqa_triton_gfx938.py::test_paged_mqa_pool_layouts` | xfail(strict=False)+具体证据；两处均非适配器读路径回归 | **完成**（2026-11-19 实测：①vllm `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 注册但零写入——探针 cache 非零字节 0/93440，fixture 全零→参考与输出同为全零→cosine 0/0；②0.5.22 SetKAndS/fused_store writer 与本夹具手工 buffer 布局漂移——interleave 与 row-major 各读臂 rel 完全相同（3.14208/3.13919），triton 与 emu 相互一致，dispatch 臂仍绿；修复=按 0.5.22 writer 语义重推夹具，非适配器任务） |
| 家庭代表 43 例（0.5.22 漂移，CI 外） | `l15_family_matrix_v1.json` adapter-pytest（`test_quantized_linear_chain`、`test_dsv41_compression`、`test_l15_execution_chains` 等） | 手工家庭 lane（L1.5 默认 off 后不进 CI）；kernels 段恒定 deselect | **诊断入账**（2026-11-19 基树 759ac7d + 同 seed 实测同败：`quantized_linear_chain[1/17/33-fp8]`+`[1/17/33-ct-fp8]` 6 例、`test_dsv41_compression[pair-state|open-group]` 2 例（ratio=2 447 字节差）；另有长跑/疑似挂起用例使整组 >700s；与 W1-W5 零回退改造无关，修复=按 0.5.22 fp8 线性/压缩语义重推） |
| HiSparse native 用例（`tests/kernels/hisparse` 3 文件，CI 外） | 需单独构建并安装的 HiSparse native 库（`setup_hip.py --hisparse-only` / conftest `--hisparse-library`）；用例文件头自述为"待显式授权的 GPU lane" | 独立 lane（`docs/frameworks/vllm033-hisparse.md`「Reproduce the isolated native lane」）；kernels 段默认 `--ignore` + WARN 行，`L1_KERNELS_HISPARSE=1` 才并入 | **kernels 段已划出**（2026-11-24 实测 main `5ef648b`：通用镜像上 104 个 setup error——`test_native` 36 / `test_registered_host` 60 / `test_typed_copy` 8，#159 带入，PR lane 与 #159 自身的 L1 当时均 skipping 故未暴露；库的构建与专用 lane 的接入待 owner） |
