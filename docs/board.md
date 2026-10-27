# 任务看板（requirements board）

> 协作入口：**按下方 R-xx 编号领任务**：把总表「认领人」列填上你的 git 账户名（Gitee/GitHub 用户名）并回填进展；
> 每条需求的明细自带验收标准，照标准交付。完成且无遗留的任务及时移出本表，
> 知识沉淀进 docs/ 现有文档（models.md / ci.md / env-vars.md /
> BACKEND_CATALOG 等）。R-xx 编号全局连续不复用。
> 本文进仓；初级敏感信息已在发布时脱敏（内部节点以代号指称）。

## 在办任务总表

（按优先级排序）

| ID | 需求 | 优先级 | 认领人（git 账户名） | 状态 | 登记日 | 最近更新 |
|---|---|---|---|---|---|---|
| R-10 | 模型×并行方式验证矩阵（服务器模型资产盘点先行） | P2 | （serving 轨在办，待填账户名） | **进行中（2026-10-18 首批，自跟踪）**：格 1/1b（Qwen3-INT8）红定性=上游 v0.5.21 W8A8Int8 动态 MoE scheme 仅 NPU（compressed_tensors.py:896 raise，非 HIP 特有 CUDA 同拒；dense 线性层路径绿=Qwen3.6-27B 前科，红面仅 MoE experts）→归因不修；格 3（GLM-5.3-INT8 717G）同族同因引用格 1；格 2（GLM-5.3 fp8）**硬 blocked=池内资产残缺**（2026-10-18 opandas 恢复核验：容器挂载直接生效、无空目录残留，但实跑三连取证——①GLM-5.3 目录无 tokenizer（借 glm5.3-channel-int8 完整 tokenizer 组装 overlay 配方已备 /root/g53/GLM-5.3-ovl）②**GLM-5.3 fp8 仅 22/141 片无 index**（命名 of-00141 单片 5G，110G/预期 700G）③替代源 **GLM-5-FP8 亦残缺 139/142 片**；唯一完整的 glm5.3-channel-int8 141/141=CT-int8 即格 3 同因红；**勘误二更：fp8 全量≈700G 级 TP4 装不下须 TP8**——运维补片进遗留跟踪）；**盘点勘误：GLM-5.3 实为原生 fp8 block 非 bf16**（与 glm53f 同格式族）；容器按评审方指示重建（**新镜像 sf_b075-20261018-0048 轮子 1013→1018+公共盘只读挂载+清 222G**）——**回归四门全绿收官**（锚点/并发 6/6/长文 5180tok 双事实对；单流 7 tok/s=10-16 基线 2×）；；**乙档首格判负（2026-10-19）**：hygon/GLM-5.3-Channel-INT4-w4a16 391G 完整可读，auto boot 崩=上游判 W4AFP8 scheme 而该 scheme 仅支持 GROUP 策略、channel（g=-1）负维崩——实际面 int4×fp8-dynamic 仅路由专家量化；命名勘误三更+「moe_wna16 现成可跑」作废（models.md 2026-10-19 行）**镜像升级新雷=tilelang 0.1.15 ThreadSync 编译回归**（mHC 融合内核硬错，0.1.12 同条件仅 Warning）→评审定规跟版本保融合→钉 0.1.12（vendor tgz 留档+bootstrap §5+upstream-watch §5 登记，待报轮子团队）；tvm-ffi 递归链再爆已解（离线验证节点 拷 .so）；战报 models.md 两行+cookbook §1.5+bootstrap §0 runbook 已落 | 2026-10-17 | 2026-10-18 |
| R-11 | 自研 triton 档产线（emu 尾部三件→融合面→marlin 桥）——独立 agent 轨 | P2 | （离线轨+合并跟踪，待填账户名） | 进行中（**前置状态：首件 fa3 prefill/varlen 已合并 master（2bbdea0，2026-10-19 协调人 执行），L1 全量门 GREEN（三跑故事收口，fa_varlen 33 用例 生产节点 A/BW1102 首实跑全 PASSED）**；**fa3 varlen serving A/B 已执行 2026-10-19：判决=triton 臂守 candidate，晋级阻塞**——≤8k ctx 质量+性能平价（解码 8.27 tok/s 两臂逐字同速 30.95s×3、prefill ±10%、锚点等价、LONGCTX 事实对）；**≥11k ctx 臂 B 2/2 boot 毙服务**（emu `_attend_chunked` 单笔 28.88GiB OOM 杀树/watchdog 挂死）vs 臂 A 3/3 过；glm53f 的 q_v paged 面超 triton 合同=内核级 13× bench 本模型不兑现属预期；A/B 顺带揪出**ops 重绑丢 q_v 包参 boot 杀手**（dsa _forward_fa3 decode 捕获崩，2759711 修）；战报 models.md 2026-10-19 行。**墙修复后终验（同日 c59135b）：晋级障碍已摘除——triton 臂 11k×3 绿（11.3s=auto）+全电池绿+LONGCTX-6k 28.4→3.1s（~9×）+锚点与 auto 臂逐字一致（修复前文本微差同源 tile 敏感性）；档位守 candidate（glm53f q_v 面恒 emu，翻默认无感；待行使 triton 核的模型过 serving A/B 再晋级）**。余项：大窗/H=32 调优；BW1102 perf 复核；kpool write/flash_mla 面承接） | 2026-10-18 | 2026-10-19 |
| R-13 | 验收后遗留收编（conv1d-flake / kpool=1 风险 / aiter reduce_scatter 契约缺口 / R-09④ wheel 复评 / R-09①②③ 晋级） | P2 | （serving 轨在办，待填账户名） | 进行中（2026-10-19 协调人 登记；**#6 R-12 守卫已完成：ee836e2 已推**；**#4 paged_mqa 晋级已完成：27c5847+3848d96 已推**；**#2 10k 容量墙已完成：c59135b+战报已推**——q 行子分块击穿（峰值 22→6.8GiB、fp64 金标 0.0046、47 门绿+新钉、serving 终验 11k×3 绿+LONGCTX ~9× 彩蛋，详见明细）；conv1d-flake 有 xfail-strict 桩自动跟踪；格 2 资产残缺补片挂运维遗留；**#1 ar_ext.cu 已改判收口（2026-10-19）：不修复，废弃——aiter GraphSafe 新镜像终验 4/8-rank 15/15 全绿（图重放净）后无修复意义，坏输出亦不配当对比参照，custom AR 七删四改除役（ce7e402），serving 验金绿） | 2026-10-19 | 2026-10-19 |
| R-05 | W4A16 Marlin 内核族接入（生产模型主力量化路线的高性能档） | P3（2026-10-17 评审定维持 P3） | （serving 轨在办，待填账户名） | **实施段已落地（2026-10-19 dev（serving 轨） 自验收：桥+w4a16_moe.py+三段式 9/9+bench 2.1-3.3×，51acb72 已推；摸底段此前 协调人 核证通过）**；**e2e 试验田变更（2026-10-19 同日 10-19 探役）**：池内 392G INT4-w4a16 实为 W4AFP8 面（int4-channel×fp8-dynamic）不适用本桥与 moe_wna16——e2e 挂「等真 group-wise w4a16 checkpoint 入池」，R-05 维持候选 opt-in 不晋级——本段后状态：**候选档 opt-in 落位**（HCU_W4A16_MOE_IMPL=deepgemm 显式；auto 现答 triton），剩 serving A/B+e2e 后评审默认翻转。细节：生产节点 A GPU4 单测级三轮探针——①deepgemm w4a16 masked 族=**数值健康**：C++ low_latency_w4a16_gemm 经 w4a16_marlin_weight repack 后对独立 torch 金标准 rel=0.0022（bf16 噪声级），轮子自带 triton golden（moe_w4a16_groupwise_triton）+repack 自洽，布局全文档化（b [E,N,K//2] u8 lo_even、scale [E,N,K//32] 与激活同 dtype、约定 (q−8)·s、actual_tokens 掩码）→ **桥接首选**；②aiter 纯 triton AWQ 反量化=精确 rel=0（小端 nibble+整零点 (q−z)·s，GEMM 同解码逻辑，配 config json 待接）→ 兜底/参考；③lightop marlin w4a16 C++ 族=**不可用**——scales_zeros 格式无文档且不可复现（20+组合+受控 s=1/z=0 微探针均失配，输出幅度与注入 scale 无关）、小形状 dequant 硬 VM-fault（设备非法访存杀进程）、config 表仅 M1-16 单一 MoE 形状族、moe_gemm_marlin_w4a16 包装源码 NameError；④deepgemm moe_marlin_prefill=w8a8-fp8 族=R-04 已判死。探针资产 tmp_fetch/probe_r05_{marlin,micro,aiter,deepgemm}.py + 容器 /root 同名 | 2026-10-17 | 2026-10-19 |
| R-07 | MTP 红配置（快配置）heisenbug 攻坚——下阶段 | P3（依赖 R-06 方法学前置） | — | 待启动 | 2026-10-17 | 2026-10-17 |
| R-14 | 上游单测收集面扩容——模型族 + 功能面全查证收编（DSv4/GLM-Next/Qwen3Next/KimiK3/Hunyuan4/MiniMax-M3 + PD 分离/dspark/mtp/dflash/EP/hicache/mock canary） | P2 | （serving 轨批 A/B/C/D 已落地；E/F 待领） | 进行中（2026-10-19 评审两轮点名缺口，协调人 全树查证定批；**批 A 已完成（同日，nmz26）**：l15_mech 段落地（gdn/hybrid_linear/mamba/swa/lightning/dsv4 六目录+**known-fail 预算绊线** passed≥90 且 failed≤42 才绿、越线/超时即红）；首跑 38F/99P/15S/173 子绿 826.58s，六族归因零静默——K1 gdn cuteDSL 编译器 `ptr<1>`≠`ptr` lowering 失配×2（平台）、K2 dsv4 JIT 元数据编不过×3（平台）、K3 l1c_evict_priority triton-ROCm 无此拷贝 op×1（平台）、K4 flash_mla `block_table=None` 桥缺×10（候补 R-11 flash_mla 面）、K5 dsv4 数值 ~20（待深查）、K6 mamba2 triton 页界×3（与 conv1d-flake 同族嫌疑）——账本落 docs/ci.md L1.5b；15 skip 全合理（14×flashinfer 平台+1 SM90 门）；第 2 轮**全门 GREEN**（gpu=0）计数复读 38/99，绊线判 PASS。**批 D 已完成（同日，nmz18/BW1101 转场）**：unit 功能面直跑分诊——PD 分离 unit/disaggregation 33 文件 515 测+157 子测 rc=0 直跑全绿零 skip、dspark 13 文件 156+43 全绿（GPU 形状门 BW1101 实测过）、mock canary self_* 七件 23 测全绿；hicache 1 合法 skip+3 HF-gated-401 快错+6 挂（mooncake/boot）、dflash 2 件全 E+挂归批 E——**694 测+200 子测零适配直跑，零静默**；账本 docs/ci.md L1.5c）。**批 B/C 已完成（同日，nmz18 接力）**：批 B 24 文件 23 绿 1 挂（GLM-5.3-Next 生产主线五件全绿 27 测+37 子——零契约面缺口正式补上；DSv4/qwen3_5/Kimi+25/Hunyuan 全绿，AMD 双件合理 skip；挂=kimi_k3_bfa_overlap 归批 E）；批 C 真发现收全——K7 dsv4 数值族（q_rope_store 6+fp8 dequant+批 A K5 同面）、K8 dtype 契约（positions int32）、K2 扩（meta+compress_plan JIT 双双编不过@gfx938）、teardown 挂两处；glm53 fusion 2S=gfx950 上游门 BW1102 同绿不了；L1 纳入定案=l15_units 段（批 B 绿 22+批 D 绿三段 ~10min），账本 docs/ci.md L1.5d | 2026-10-19 | 2026-10-19 |
| R-15 | 性能回归门（相对阈值 + 历史锚定 + 三级防线；微基准进 L1、e2e 进夜间） | P2 | — | 待启动（2026-10-19 评审拍板"需要"；协调人 已完成上游 perf 面查证与设计定案，见明细） | 2026-10-19 | 2026-10-19 |
| R-16 | skill/CONTRIBUTING 与门禁体系同步维护（CLI 上手守则多人化适配） | P2 | — | 待启动（2026-10-19 协调人 完成 skill 补丁首轮=kill-switches AR 除役定位/shadow-matrix 四档状态机/cookbook-matrix 零 env 原则/SKILL 法则 10-11+治理闭环节+仓库结构刷新——本轮即交付大半；本行跟踪剩余：CONTRIBUTING MR 门禁段与 ci.md 同步、flash_attn wheel 段落复核刷新、新同事上手路径实效检验） | 2026-10-19 | 2026-10-19 |
| R-17 | 上游 release 分支策略（rel/v0.5.x 双向同步）+ 主仓分支拓扑治理（仅 main+rel，个人调试 fork）+ **v0.5.22 升级适配役（已 release，首演）** | P1 | 分支策略/拓扑/文档面已落地；**v0.5.22 首跑普查已完成（同日，nmz26/baa9bef）**：单测矩阵全量对账落 ci.md L1.5e——**零例绿变红**，机制老族同构、模型族面全绿；**破损桥名单四条定案**（P0=memory_pool kv_fix needle 失配[boot 级炸点]+add_norm ar_norm seam 符号漂移；P1=dsv4 JIT 家族编不过 3→18 扩面；P2=dspark server -9 归因）；适配 commit 落 rel 待开工；**P0 两根适配完成（同日，rel/v0.5.22@2ad90aa）**：kv_fix 检测 .22 伤口愈合（上游重写 strides 为纯 int）转 no-op、ar_norm ARM 候选门消失（.22 aiter 融合原生化）转优雅降级 dispatch-only——**双树兼容**（.21 回归 patch/ARM 照装、.22 import 干净+canary 六件 22P 全绿）；**双向同步首演达成**：rel 修复 cherry-pick 回 main（1340a42，L0 绿）+**第二轮全量同步（2026-10-21）**：rel@9e5ce08 的 74 颗 rel-only 中 64 颗代码修复/feat/docs 落 main（dsv4.1 全战役：V41 注册表别名与自愈、DSV4 JIT 针脚族 [dpskmeta HIPcub/c_plan uint2/main_norm_rope syncwarp/批2批3 float2 绑定/批4/quant v2 HIP fp8]、flashmla-emu sparse 语义修正[token loc token 级]+页池 as_strided+原子写全家装、dsv4_ep_bf16+triton masked fp8 双核[DeepEP 免 pack]、moe_dualstream_gate[graph replay 乱码根因]、spec_h2d_pin[MTP 步墙]、jit_fp8_ffi 重绑+jit_build_log[冷跑可观测]、fp4→fp8 解包 block_shape 族），7 颗 patch-id 已在 main 自动跳过；**有意截留 10 颗**=2 docs（其内容已随 dac7de1 在 main）+8 dbg(dsv4) 会话探针（插桩用后即弃性质，按 §1.9 纪律不入主干）；同步后 main 与 rel 代码面（sitecustomize/sgl_kernel/ci/tests）零差异——唯一残留=flash_mla.py 91 行 dbg 探针（rel 有 main 无，同截留性质）；L0 GREEN(54 warn)+adapter 单测 47 过 2 skip 本机实证；=R-17 验收 3 之半巩固；余项全收口（同日，L1.5f）：**rel 线 L1 全门 GREEN**（gpu=4，--up-tree /sglang-v0522，mech 预算判 38F/99P 比普查还少 3 败=适配后装配走得更远）；**JIT 三根定性→两根同配方实测治愈（同日 L1.5g）**：CUB 头缺失与 hipcc uint2 分解缺口均被 cuda-mocker 配方收口（用户定性 /opt/dtk/cuda/cuda-12=DTK cuda-mocker，CUDA 语义虚拟层底层 HIP 与 adapter 哲学同构；配方三件套=合成 ROCM_HOME 根[hipcc 壳剥 USE_ROCM/HIP_PLATFORM+注入 SGL_CUDA_ARCH=750]+LD_PRELOAD mock libcudart；零装库零 sglang 改；metadata+conv1d 双 LOAD_OK、conv1d 数值对拍 PASS 9.4e-3；cutedsl=K1 仍账）；dspark -9=boot 资源类登记批 E；**R-09① 终态定案**：.22 上游原生接手（消费链直通 ca_comm 同一 aiter GraphSafe 面），桥降级即终态、ar_norm 晋级动作变「上游默认开+验证 HCU 通」。余=serving 级四门（production 宣布前置）+**锚定轮已跑（2026-10-20 nmz18 复活轮，ci.md L1.5h 全账）**：mech 38F/101P 持平=32F 拆账被推翻——成分=topk_v2 构建败（cuda.cu 引 transform_packed 而 v0522 头已名 transform_paged，针脚失配）连坐 17 败行+c_plan 装上但运行期炸（:562 PrefillPlan）×2+gdn cutedsl K1×3+数值不 close×11；三模块隔离实证未被推翻，全链翻绿压 **topk/c_plan/quant 三模块移植**（topk/c_plan=adapter needle patch 可治；per_token_group_quant=HIP 针脚+arch750 vec16B 静态断言，归 R-19）；同轮附得=**配方揭盖效应**（kernels 5F/160P、l15 39F/87P 均为配方把回退/skip 翻成真跑撞同批缺口，跨机对账必须同配方形态）+**nmz18 复活**（镜像配对铁律推翻：真根因=冷 tvm-ffi 构建自锁+缺 hyhal 挂载双因素，"镜像×挂载面×cache 温度"三件套铁律改写入 bootstrap §4b）+CI 种子扩包 tvm-ffi/torch_extensions（cache_export/import，234K 免疫冷容器死锁）+镜像固化合成根与 hcu_env 收编 ROCM_HOME/LD_PRELOAD 两 env+deep_gemm_dtk 改读 DTKROOT（配方已用 .info 兜住，消除后更净） | 进行中→**第一段落地（2026-10-19）**：①主干定名+调试分支清点双关账（本地/远端清单均只剩 main）；②CONTRIBUTING.md 弃用收编 board「协作与分支规约」节（双线模型三绑定+主仓拓扑+fork-MR+cherry-pick 节拍成文；全仓 6 文档外链改指；弃用与收编同由用户 2026-10-19 拍板执行）；③l1_gpu_gate.sh 树参数化（--up-tree/HCU_UP_TREE，显式树不存在即 GPU 探测前 loud fail 防错树假绿；全局 PYTHONPATH 跟随所选树；防错树三证据干跑在案：坏树拒/坏参拒/树校验先于 GPU 探测）；④rel/v0.5.22 分支从 main(691d080) 建立并推 Gitee（远端清单=main+rel 双分支入拓扑）；⑤主仓拓扑治理事实达成（ls-remote 仅 main↔rel）。余=v0.5.22 树并置/适配普查/全门+双向同步演武+fork-MR 演练（见明细） | 2026-10-19 | 2026-10-19 |
| R-18 | tvm-ffi dlpack 冷缓存导入期自递归——adapter 预热桥（方案3：sitecustomize 代码化预热） | P2 | — | **根因实证+双缓解落地（2026-10-20 nmz18 锚定轮）**：构建器 fork 子进程再 import torch/HSA→父子互 poll pipe 永挂（/proc/syscall+fd 全指纹取证；同型死锁曾误诊为"镜像↔宿主配对"，已改写 bootstrap §4b）；缓解①=cache 种子收 tvm-ffi/torch_extensions（cache_export/import 扩包，234K 播种即免构建）②官方开关 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 兜底已入文档；方案3 预热桥本体仍未做（评审原案），优先级待定（CI 面已被①②免疫） | 2026-10-19 | 2026-10-20 |
| R-19 | JIT kernel vendor 化——sgl_kernel JIT 需求清单化提给 aiter/DTK 维护方（预编译随 wheel 发货，我们只管调用） | P1 | — | 待启动（2026-10-20 用户拍板战略方向；现状账：sglang JIT 35 模块自编，38F 中 32F=编译失败（CUB/uint2 两根=DTK 工具链，需 cuda-mocker 配方自救）；而 aiter wheel 已证明满预编译回零编译是可达态）；配套（同日落）：冷跑可观测性=JIT 逐颗进度行+gate 段心跳+pytest 行缓冲（ci.md「慢段可观测性」），编译风暴与楔死日志可辨，见明细 | 2026-10-20 | — |
| R-20 | adapter 多宿主 bootstrap（sglang/vLLM 双 MR） | P1 | alexanderbin123 | **MR1 待提交**：薄 `sitecustomize`、确定性宿主分发、common/sglang/vllm 桥隔离及完整 wheel 已完成；CPU/adapter 134P+3S，HCU RMSNorm kernel 10P、真实 vLLM ABI/parity PASS；L0 无新增失败（仅基线两项既有债务）。当前节点无 sglang 包/可复用容器，cookbook e2e 留作 MR 合并门；MR2 必须待 MR1 合并后从刷新后的 upstream main 建分支，只接官方 vLLM v0.32 seam，官方已有能力不重复建设 | 2026-10-27 | 2026-10-27 |

## 需求明细

### R-20 adapter 多宿主 bootstrap（sglang/vLLM 双 MR）

**两段交付边界**

1. MR1 `feat/bootstrap-multihost` 只做结构迁移：`sitecustomize.py` 固定为
   三行入口，`hcu_adapter.bootstrap` 选择唯一宿主并按 common→host 安装；
   既有 sglang 顺序链机械迁入 `bridges/sglang.py`，vLLM RMSNorm ABI 桥抽到
   `bridges/vllm.py`，不启用 vLLM platform/plugin 路径。
2. MR2 在 MR1 合并后从主仓最新 `main` 新建分支，面向官方 vLLM v0.32 tag；
   先复用官方 platform/plugin/custom-op seam，只迁移 plugin 独有且有证据的
   优化，遵守“官方已支持就删除过渡桥、不重复建设”。

**MR1 验证证据（2026-10-27）**

- `tests/test_ops_cpu.py tests/adapter`：134 passed，3 skipped（含 wheel 全内容
  检查）；`tests/kernels/test_layernorm.py`：空闲 HCU 上 10 passed；真实
  `vllm._custom_ops` 绑定的 plain/fused RMSNorm HCU destination ABI 与 torch
  参照对拍 PASS。
- L0 与基线 `f4b2c29` 相同，仅有两项既存 RED：`HCU_MOE_ALIGN_FUSED` 文档
  缺口、`wna16_channel_moe.py` 注册的 `moe` catalog 缺口；本 MR 不夹带修复。
- 当前执行面有 8 张空闲 HCU，但没有 sglang 包、sglang 源码树或可复用容器，
  因而未执行 cookbook 服务冒烟；该项必须由具备三绑定环境的 MR watcher/
  owned node 补齐后再合入。未向 `models.md` 写入新支持结论。

### R-07 MTP 红配置（快配置）heisenbug 攻坚——下阶段

**背景（2026-10-17 阶段收官挂起）**
- MTP 安全配置已全绿但 6× 慢（0.58 tok/s）；快配置（开图）输出乱。
- 定位战排除图腐蚀后，真凶定性为**探针敏感 heisenbug**（加侧缓冲拷贝即
  转绿）——现有观察通道会扰动它，需换不扰动时序的取证手段。

**工作内容（前置：R-06 的干净对拍方法学）**
1. 离线重放：用已捕获的输入快照离线喂 fa3 emu / verify 路径，绕开服务态时序。
2. 时序扰动二分：延迟注入定位敏感窗口（哪一段的时序变化会掩盖/触发腐蚀）。
3. 修复后走 R-01 同款收口：Berlin 连续 PASS → e2e 四门 → 性能回收留档。

**验收标准**
1. 红配置（快配置）Berlin/四门全绿，MTP 吞吐恢复数据留档（目标 >3 tok/s）。
2. 根因有书面结论（含为何探针敏感），坑沉淀 one-pit-one-test 单测。

### R-05 W4A16 Marlin 内核族接入（生产模型主力量化路线的高性能档）

**背景**
- 生产模型大部分为 channel-wise + Marlin（W4A16）量化——硬件依据见
  长期方向区「硬件特性备忘」；das 栈有自研 marlin 实现先例。
- 现状：W4A16 路径 adapter 侧靠 triton 兜底（AWQ 需
  `--quantization moe_wna16`）可跑，但缺 marlin 级高性能内核；上游
  marlin 内核为 CUDA 专有，HCU 按四级路由找归宿
  （lightop → aiter → 自研 → triton 兜底）。

**治理选位（2026-10-17 评审对齐：adapter 轨）**
- 判据套用：上游 sglang 在 CUDA 上原生支持 marlin 量化路径（gptq_marlin /
  awq_marlin / W4A16），HCU 缺 = **行为还原 → adapter**（同族先例：channel-wise
  int8、AWQ/moe_wna16）。"动机是性能 + 用自研内核"不改变选仓
  （CONTRIBUTING §3b）；自研 marlin 内核库本体放 vendor 侧（同 lightop/aiter
  轮子），adapter 只放桥（seam + 四级路由 + 登记）。
- **分支条件（摸底必答）**：若生产 marlin checkpoint 实为 das 私有 repack
  布局（非标准 marlin 格式，仅 das 栈可加载），则"加载该格式"为上游没有的
  行为，该部分走 plugin 轨（同 slimquant 先例）。摸底时看 das 树 marlin
  权重格式即可裁定。

**摸底结论（2026-10-17 协调人 完成，das 树实读）**
- das 的 marlin 内核形态：`kernels/ops/quantization/gptq_marlin.py` /
  `moe/moe_wna16_marlin.py` 等 = **自带 `gemm/marlin/marlin.cuh` 源码经
  tvm_ffi JIT 装载**（cache_once，与 bootstrap.md §3 的 tvm-ffi 预热同机制）。
- **格式标准性（必答项）已答**：das 走标准 `gptq_marlin_repack` /
  `awq_marlin_repack`（GPTQ/AWQ → 标准 marlin 布局）——**无私有格式，
  adapter 轨定案**。`slimquant_w4a8_marlin` 为私有格式部分，维持 plugin 轨。
- **候选二（vendor 档一步到位）**：DTK deepgemm 轮子自带
  `grouped_gemm_w4a16_nt_masked_entry`（附 vendor triton ref）+
  `moe_marlin_prefill`——R-04 摸底时容器实测确认，接入评估与 das
  marlin.cuh 路线二选一（bench 定）。

**wheel 侧运行时摸底（2026-10-19 dev（serving 轨），生产节点 A GPU4 单测级）**
- **deepgemm w4a16 masked 族 = 唯一数值健康的 vendor C++ 路径（首选）**：
  `w4a16_marlin_weight`（[E,N,K//2] u8 → 低延迟 W4A16 布局）+
  `grouped_gemm_w4a16_nt_masked_entry`（→ low_latency_w4a16_gemm C++）对
  独立 torch 金标准 rel=0.0022；轮内 triton golden
  `moe_w4a16_groupwise_triton` 同分自洽。约定（探针钉死）：nibble 小端
  沿 K 打包、(q−8)·s（无显式零点）、scale [E,N,K//32] 与激活同 dtype
  （bf16/fp16 均可，须一致）、actual_tokens 行掩码。N%16==0、K%128==0。
- **aiter 纯 triton AWQ w4a16 = 精确（rel=0）**：qweight [N,K//2] 小端、
  qzeros [K/G,N//2] 沿 N 打包整零点、scales [K/G,N]、(q−z)·s；GEMM 内核
  同解码逻辑但需 tuned config json（形状覆盖待查）——兜底/参考档。
- **lightop marlin w4a16 C++ 族 = 不可用（当前轮子）**：scales_zeros
  格式无文档、20+ 候选组合与受控微探针（s=1/z=0）均无法复现（输出幅度
  与注入 scale 无关）；小形状 dequant 直接 VM-fault 杀进程；config 表仅
  覆盖 M1-16×单一 MoE 形状族；`moe_gemm_marlin_w4a16` 包装源码引用未
  定义变量（NameError）。除非厂商补文档，此路封存。
- deepgemm `moe_marlin_prefill`（w8a8-fp8 族）随 R-04 终判死，不再计。

**工作内容（摸底可随 R-04 同批做，同在 das 树）**
1. 摸底：自研 marlin 库形态/接口/dtype 覆盖（int4/int8 权重？MoE 与
   dense Linear 都覆盖吗）、das 当时的接法、上游 marlin 调用面清单
   （repack + gemm 两段）。
2. 桥接：调用面映射 + env 档位枚举登记（命名对齐
   `HCU_KDA_PREFILL_IMPL` 先例）；布局/repack 语义差异在桥内消化。
3. 单测先行（三段式）→ e2e（AWQ/GLM int4 线）→ A/B bench vs
   moe_wna16 triton 兜底 → catalog 晋级。

**验收标准**
1. 数值对拍过（vs triton 兜底与离线 dequant 参考），容差按 test-suite.md。
2. bench 证明性能收益（vs moe_wna16 triton）。
3. 三处登记齐（L0 绿）、L1 绿、e2e 四门通过。

### R-10 模型×并行方式验证矩阵（服务器模型资产盘点先行）

**背景（2026-10-17 评审拍板登记）**
- 单测绿 ≠ serving 通（校准结论）：模型×并行的真实覆盖需要实跑记录在案，
  把"官方支持的模型/并行方式理论上都能顺利验证"变成绿格矩阵。
- 服务器模型资产比预想新——生产节点 A `hygon/` 10-16 刚到一批量化矩阵（GLM
  全家 + DSV3 + MiniMax + Qwen），先盘点后排批。

**盘点结论（2026-10-17 协调人 完成，生产节点 A 主池实测 config.json/体积；16 卡池 未扫——2026-10-19 试连 <远程池>连接受限（网络层拒），补扫等隧道/窗口）**

*甲档·验证邻域内（量化格式已验证 × 架构已跑过，预期顺）：*
| 模型 | 格式/架构 | 体积/资源 | 备注 |
|---|---|---|---|
| Qwen3-30B-A3B-Channel-INT8-w8a8 | CT int-quantized × Qwen3Moe | 30G，单卡 144G 可跑 | **首选**：已验证格式（Qwen3.8-27B 同族）的更大兄弟，最快出绿格 |
| ZhipuAI/GLM-5.3（fp8 block 勘误，**池内残缺 22/141 片**） | GlmMoeDsa | 141 片发布为 ~700G 级（现存 110G），TP8×144G（**原判 135G/TP4 作废**：10-18 实测单片 5G×141） | 大 GLM-5.3 的原生 fp8 全量版替代源 GLM-5-FP8 亦残缺（139/142）；运维补片后方可排期 |
| hygon/GLM-5.3-Channel-INT8-w8a8 | GlmMoeDsa（CT int-quantized 待核） | 717G，TP8×144G 紧（90G/卡+KV），16 卡池 更稳 | 已验证格式 × 新架构 GlmMoeDsa |
| GLM-5.3-Flash（bf16，在跑） | glm5_next | 306G，TP8 | 主线基线，矩阵原点 |

*乙档·新格式/新族（一役级，各挂对应需求）：*
| 模型 | 格式/架构 | 体积 | 去向 |
|---|---|---|---|
| GLM-5.3-Channel-INT4-w4a16 | GlmMoeDsa × CT pack-quantized | 392G | **10-19 实跑判负+勘误三更**（models.md）：实为 int4-channel×fp8-dynamic（W4AFP8），上游 scheme 有但 channel 算术崩；R-05 试验田假设作废 |
| GLM-5.2-Channel-INT4-w4a8 | GlmMoeDsa × **slimquant_w4a8 私有格式** | 363G | plugin 轨（das slimquant_w4a8_marlin）；新族+私有格式双重新，立需求再排 |
| MiniMax-M2.5-bf16 | MiniMaxM2ForCausalLM | 427G | 上游有 M2 族支持基线，新版本模型验证 |
| DeepSeek-V3-w4a8-MOE-Quant | w4a8 私有格式 | 334G | 同 slimquant 轨 |
| DeepSeek-V4.1-Flash-Channel-INT4-w4a8 | w4a8 | 721G | 16 卡级资源，远期 |

*丙档·暂不可用（盘点即排除，不占排期）：*
- **空目录占位**（4KB 未下载）：Hy4-preview-Channel-FP8-w8a8、Kimi-K3-DSpark-INT4。
- **权限不可读**（他人/训练组产物）：WeLM-v4.5-80B-A3B-0813-MTP（213G，MTP 能力
  与 R-07 相关，若能协调权限再启用）、GLM-5.3-Flash-CHANNEL-INT8-w8a8-v3
  （=iter_0033200 训练 checkpoint，非发布版）。
- **Channel-FP8 族**（GLM-5.3-Channel-FP8-w8a8 等）：CT float-quantized fp8 路径
  adapter 未覆盖（已知坑，fp8 serving 层病着——单列不混入本矩阵）。

**工作内容**
1. 首批（甲档，走现有模型套件守卡机制）：Qwen3-30B-A3B-INT8（单卡最快）→
   GLM-5.3 bf16 TP4 → GLM-5.3-INT8 TP8/16 卡池；每模型 launch→生成→锚点一条龙，
   models.md 记行。
2. 并行维度：TP 主线已通；首批顺手补一档 **EP**（Qwen3-30B-A3B-INT8 双卡 EP）；
   dp-attention（GLM-5.3-Flash）= 已知结构缺口，单列不进首批。
3. 乙档随对应需求排期（w4a16→R-05；w4a8→plugin 轨）；16 卡池 模型池补扫。
4. 失败项归因记录（格式未覆盖/架构缺口/资源不足），不静默丢弃。

**验收标准**
1. 首批每模型 models.md 绿行 + 启动配方入 cookbook（按 R-08 回仓原则）。
2. 模型×并行（TP/EP）矩阵表落 docs；乙档去向逐项标注。
3. 不可用项的理由留档（本明细即底账，后续变动在此更新）。

### R-11 自研 triton 档产线——独立 agent 轨（emu 尾部三件 → R-09 融合面 → R-05 桥）

**背景与定位（2026-10-18 评审拍板"开"）**
- 与dev（serving 轨）（serving 轨）并行的第二条工程线，资源错开：dev（serving 轨） =
  生产节点 A GPU4-7 serving 轨；dev（triton 离线轨） = **离线验证节点（BW1101）卡
  4-7 离线单测主场**，bench 与平台判别待 BW1102 窗口复核。
- 活源：emu 清退尾部三件无人认领（fa3 prefill/varlen、kpool write、
  flash_mla/dsa_paged_mqa 面），后续按 R-09 fusion-ledger 缺失项排序承接，
  R-05 marlin 桥候补——即长期方向第 3 条"自研 triton 档"的产线化。
- **卡型警示**：离线验证节点=**BW1101 ≠ 生产 BW1102（gfx938）**——正确性/对拍在
  离线验证节点 有效；性能数字不迁移；gfx938 特有 bug（如 kpool int64 div-4）在
  BW1101 不可复现，相关判别必须标注"待 BW1102 复核"。

**隔离三规**（见使用约定）：独立 clone+分支不推 master；止步 candidate 档
（production 晋级 = dev（serving 轨） serving A/B）；本条目 协调人 维护。

**工作内容**
1. 首任务：fa3 prefill/varlen triton 内核（现役 emu；decode 侧已 triton 化
   可参考——3D packed q/归一化 capture 合规等坑已沉淀在战报与测试）。
2. kpool write triton（前置 = gfx938 int64 div-4 平台 bug 判别，需 BW1102）。
3. flash_mla/dsa_paged_mqa 面 triton 化。
4. 完成后按 R-09 fusion-ledger 头部缺失四项承接（AR+norm 融合/silu+quant
   融合/quant vendor 档/weight_packed_linear）；R-05 marlin 桥候补。

**验收标准（每内核一验）**
1. 三段式单测绿（参照/形状矩阵/容差按 test-suite.md 快查表）。
2. bench vs emu 留档 + BW1102 复跑记录（或显式标注未复核）。
3. 三处登记齐（registry/env-vars/catalog）+ UI 快照同 commit；L0/L1 绿；
   catalog 状态=candidate。

### R-13 验收后遗留收编（R-01/R-02/R-12 出板后开放项跟踪）

**背景（2026-10-19 评审定：已完成项出板，遗留集中收编）**
三项验收收官（R-01 主线闭环 / R-02 emu 替换达成 / R-12 总装入口防护落钉
=commit e22bfb7，协调人 独立复跑本地 3P+2S、容器 L1 同款 env 4P+1S），各自
遗留收敛到本条统一跟踪；知识主体在 docs（models.md 战报 / env-vars.md /
fusion-ledger / catalog）。

**遗留清单**
1. ~~ar_ext.cu 真修复~~ **改判废弃（2026-10-19，ce7e402 已推）**：评审方
   判断实证——aiter.dist GraphSafe 在 sf_b075-20261018-0048 压测
   4-rank+8-rank 各 15/15 全绿（图捕获+新鲜输入重放×4，1eb038d 类不复
   现），serving 前科全绿，latency 与 NCCL 同档（88vs69us 小消息、
   1-16MB 0.93-0.95×）；custom AR 无生产位、坏输出不配当对比参照 →
   不修复，代码除役（triton_ar.py/ar_ext.cu/build+三手测工具删，
   allreduce.py 墓碑，四 env 键撤，catalog 除名）。汇合协议考古走 git 史。
2. **10k 容量墙**：fa3 emu 稠密化 OOM 32GiB（f1 长文 4972 过、10k 探针
   崩；与 fork 无关）。→ **已完成（2026-10-19，c59135b 已推）**：根因=
   emu (maxL×C) 板 q 行零分块（无 tile 时 8192 面实测 peak 22.02GiB，
   叠 serving 99.6GiB 驻留=28.88GiB 单笔 OOM；前役 triton 臂 2/2 毙 vs
   auto 3/3 过的"臂分叉"实为驻留差异不是臂差异）；修=QTILE_TOKENS=1024
   行子分块（内层 KV 流式 LSE 合并原样，逐 bit 同序、图安全）——峰值
   6.80GiB（-69%）、fp64 金标 0.0046、47 门绿+新钉
   test_extend_face_peak_bounded（跨卡缩放不变量）；**serving 终验
   triton 臂 11k×3 绿（11.3s=auto 同速）+全电池绿+LONGCTX-6k
   28.4→3.1s（~9× 彩蛋）+锚点与 auto 臂逐字一致+解码 8.27 tok/s 不变**。
   R-11 晋级障碍随之摘除（catalog 守 candidate 待行使 triton 核的模型过
   A/B）
3. **conv1d-flake**：fork `_causal_conv1d_fwd_kernel` 地址布局敏感误计算
   （L1 布局 cos 0.67 六进程稳定复现，换分配排列即 0.9999）——DTK triton
   codegen 嫌疑。**xfail-strict 观测桩常驻 L1（DTK 修好即 XPASS 翻红）**；
   主动排查候选：fla 上游对照 / DTK 编译器定位 / emu 回退评估。serving
   恒 channel-last 不受影响。
4. ~~**paged_mqa triton 档 production 晋级**~~ **已完成（2026-10-18 深夜，27c5847+3848d96 已推）**：validated（8849dc6 同窗 A/B：解码 +9%/+4%、四门绿+马拉松 0/20）→ production + 默认 vec→triton（hcu_env default/catalog note/env-vars 对账三面）；zero-env 翻转后四门验证绿（boot 即打 triton(default)、锚点 4/5 同基线形态、并发 6/6、确定性 x3、长文 5.9k 双事实对）；短 prompt 256 tok=33.1s/7.73 tok/s 三跑稳定（非同窗观测，仍优于 emu 参考值 34.18s）。emu-vec=显式 opt-out 保留（=vec/loop 双词）。
5. **kpool=1 DSA 模型接入风险（登记性）**：强制门下写行主序 vs 读交错
   将错位——届时 `HCU_DSA_PRESHUFFLE=0` 全链行主序或 writer 归一。
6. ~~**R-12 小打磨（minor）**~~ **已完成（2026-10-18 深夜，ee836e2 已推 Gitee）**：HIP 冒烟臂 bare-shell 守卫——`_sglang_on_child_path()` 收集期沿 _clean_env 组装的 PYTHONPATH 查 sglang 包目录（不真实 import：pytest 进程已被 sitecustomize 污染、探针子进程贵），两 HIP 臂 SKIPif 组合条件；容器双场景对照：bare 3P+3S（修复前 FAIL）、门禁 5P+1S 零回归；守卫契约测试 test_sglang_reachability_probe_contract 常驻；顺修 _clean_env `os.path != ROOT` 笔误。
7. **aiter reduce_scatter 重叠张量契约缺口（2026-10-19 R-04 EP 复验挖出，
   上游/DTK finding）**：EP 拓扑下上游 EP FFN 边界 attn-tp reduce_scatter
   是现树无条件路径，踩 aiter reduce_scatter 重叠张量契约 ValueError——
   现役解法=HCU_DISABLE_OPS=aiter_ar 强 NCCL 等价（af433c6 同役：ar_norm
   臂限纯 TP）。根因在 aiter 轮/上游边界语义，候选去向 operator-requests
   （对供应商提需求标准）跟踪。
8. **R-09④ weight_packed_linear**：随下轮 DTK wheel 复评（本轮 weight-pack
   机制面三重不可用已判死，R-04 终判同源）。
9. **R-09 ①②③ production 晋级**：quant_aiter 桶（15.1×/7.5×）与 ar_norm
   桶（1.38-2.11×、tp8 216 次融合实证）均 candidate——各需 serving A/B
   bench+马拉松后翻默认（quant 须 fp8 模型在池）。

**验收标准（按项）**
2/4/6 已完成（c59135b / 27c5847 / ee836e2，证据见验收记录）；1/3/7/8/9
各项：修复/晋级/复评落地时四门或 L1 证据 + 三处登记同步；5 接入时处理。

### R-14 上游单测收集面扩容——模型族全查证收编

**背景（2026-10-19 评审提缺口，协调人 上游树全查证）**
- 现状：l15_upstream 段只收 `attention/unittests` 四目录（dsa/kda/dense/mla，
  硬编码）；上游 kernels/ops/ 与 unit/models/ 两级**完全未收**；机制目录
  gdn/hybrid_linear/mamba/swa/lightning/dsv4 六个也未收。适配层对多个模型
  族有活桥/活路由，但上游契约面零单测覆盖。

**六族查证底数（容器 /sglang-v0521 实查，2026-10-19）**

| 族 | 上游 arch | 可收测试（未收） | 备注 |
|---|---|---|---|
| DSv4 / v4.1 | deepseek_v4.py(+nextn/dspark/vit) | unittests/dsv4/test_deepseek_v4.py（25 测）；kernels/ops/attention：dsv4 三件（indexer_quant 3 / q_rope_store 4 / dsv41_prefill_sparse_indexer 4）+ compress_plan_bounds / compress_state_runtime_shapes + **dsv4/ 子目录**；unit/models 六件（dsv4_shared_expert_fusion / unified_fp8_q_pair / rope_policy / mxfp4_shared_expert_requant / amd_fused_mhc / amd_wo_a_bf16——后两件 AMD 专属预期 SKIP） | 原始 R-14（dsv4 目录+三件）并入本条 |
| **GLM-5.3-Next**（生产主线！） | glm5_next.py(+nextn) | unit/models：test_glm5_next_bfg_fusion（3 测）/ test_glm5_next_modelopt / GLM 族件 test_glmmoedsa_correction_bias_fp32 · test_glm_moe_gate_fp32 · test_glm_nextn_moe_ptpc；kernels/ops/moe：test_glm53_shared_expert_fusion（2 测） | **生产模型零契约单测，本条最高价值项**；adapter 对它有 DUMP_LAYERS/mHC 活钩 |
| Qwen3.8 / Next | 无 qwen3_8 专名文件；qwen3_next.py(+mtp) 在 | 机制级=unittests 的 gdn/swa/mamba/hybrid_linear 四目录（Qwen3-Next=hybrid GDN 架，其注意力面全在这四目录）；unit 级=qwen3_5 四件（packed_weight_loader / mtp_quant_config / modelopt_fp4 / pipeline_parallel）+qwen4_exp_ple_table + embedding_registration | 上游无 "3.8" 专名测试——3.8 由 qwen3_5 族 arch 承载与否分诊时核；deterministic 类（test_qwen3_next/_qwen35_deterministic）=模型 boot e2e 面**不纳入**（归 R-10 模型轨） |
| KimiK3 | kimi_k3.py(+vl) | unit/models 三件：test_kimi_k3_packed_modules_mapping（1 测）/ test_kimi_k3_bfa_overlap / test_kimi_k3_vision | 无机制级专属目录（走 MLA 通用面）；AMD eval e2e 不纳入 |
| Hunyuan4 | hunyuan_v4.py(+nextn) | unit/models 两件：test_hunyuan_v4（4 测）/ test_hunyuan_v4_nextn_weight_loading | 件小，低风险批 |
| MiniMax-M3 | minimax_m3.py(+vl) | **上游无 unit 级测试**（只有 AMD mi35x/mi30x eval e2e） | 无可收物——登记"上游缺失"，不可能是我们补上游测试，归档即可 |

**功能区查证底数（评审方第二轮点名，同日实查）**

| 功能 | 上游家 | 底数 | 形态/风险 |
|---|---|---|---|
| **PD 分离**（disaggregation） | test/registered/disaggregation（21 件）+ unit/disaggregation（**33 件**） | unit 级 33 件=纯 python 单测（bootstrap/decode 池/kv 校验/role switch…，**首选批**）；顶层 21 件=多进程双服务 e2e（basic/dsv3 件等），挑核心件分诊 | unit 批零依赖可直跑；顶层批需双 boot，走 mock_model 或择件 |
| **dspark** | spec/dspark 目录（**13 件**） | block_accept_estimator/confidence_metrics/kernel_parity/scheduler/sps_table/stacked_ctx_kv_parity/ragged_verify 等——**纯单测大头** | kernel_parity/ragged_verify 可能踩 GPU 形状门（我们踩过 verify 崩点，预期有真实发现） |
| **MTP** | spec/ 顶层 f3 件 + mock_model/test_e2e_spec_eagle | test_frozen_kv_mtp / test_spec_mixed_chunk / test_spec_standalone(+extra)；无 python 级 mtp unit（EAGLE 路径走 e2e） | 项目红线认知：MTP×GLM5 图内 verify 在本栈 heisenbug（R-07 台账）——预期非全绿，登记有价值 |
| **dflash** | spec/dflash（2 件） | test_dflash.py + gemma4 e2e 件 | 主件单文件，小批量 |
| **EP**（高吞吐/低延迟） | ep/（8 件） | test_deepep_large（=LL 侧）/ test_deepep_small(+extra)（=吞吐侧）/ eplb_no_a2a / flashinfer_a2a / mooncake_ep_small | **多进程 requests 面为主（8/10 走 run_script/launch）**——boot 级，预期大部分 SKIP 或改 mock 后纳入；HCU_DEEPEP_BRIDGE 活桥正缺这层验证 |
| **hicache** | hicache/（11 件） | storage 四后端 file/mooncake/3fs/umbp + variants + spec_file/mooncake + pp_with_hicache + qwen35_hicache | storage 后端类纯 python 可直跑；import 依赖（mooncake/3fs/umbp 轮子缺失即 SKIP） |
| mock canary（补充发现，与 PD/CP/PP/TP 共振） | mock_model/（12 件） | test_e2e_{pd,cp,pp,tp,spec_eagle}.py + self_unit canary 六件 + perturb 两件 | **mock 模型 e2e= 无 GPU 权重也能跑部署拓扑的功能面**——hicache 有 qwen35 件同理：功能单测与模型解耦，正合 HCU 依托 mock 拓扑验证的路线 |

**工作内容**
1. 分批纳入（防一次吞撑爆 l15 段 2300s 超时）：
   - 批 A（机制目录）：**已完成（2026-10-19）**——l15_mech 段落地
     （gdn/hybrid_linear/mamba/swa/lightning/dsv4 六目录收集，含
     known-fail 预算绊线 passed≥90/failed≤42）；38 失败六族归因账本
     落 ci.md L1.5b，两跑稳定+全门 GREEN（证据与族表见 docs/ci.md
     L1.5b）；
   - 批 B（unit/models 族件）：**已完成（2026-10-19，nmz18）**——24 文件
     分诊 **23 绿 1 挂**；GLM-5.3-Next 五件（生产主线最高价值）全绿（bfg
     3+27 子/modelopt 4+8/glmmoedsa 9/gate 1+2/ptpc 10）；DSv4 五件全绿
     （AMD 双件合理 skip）、qwen3_5 六件全绿、Kimi 三件+kimi_k25 超底数
     加收 55P、Hunyuan 三件（v3 加收）全绿；唯一挂=test_kimi_k3_bfa_overlap
     （首测点后静止→归批 E 逐件）；
   - 批 C（kernels/ops 族件）：**已完成（2026-10-19，同场）**——真发现
     收全：glm53_shared_expert_fusion 2S（上游 gfx950 门，BW1102 同样绿
     不了；构造面由批 B 守）、dsv41_sparse 4S+dsv4 子目录 18S 全合法；
     **负面两族+一扩**：K7 dsv4 数值（q_rope_store 6/indexer_quant fp8
     dequant 1/批 A K5 ~20 同面）、K8 dtype 契约（`positions must be
     int32` 桥候选失配点）、K2 扩（meta+compress_plan 两个 JIT 都编不过
     @gfx938）；compress 两件 teardown 挂随 K7 深查役——账本 docs/ci.md
     L1.5d；**L1 纳入定案：l15_units 段=批 B 绿 22+批 D 绿三段（PD/
     dspark/canary），批 C 负面件修复后再转正**；
   - 批 D（unit 功能面直跑优先）：**已完成（2026-10-19，nmz18/BW1101 实测，
     占卡转场）**：unit/disaggregation 33 件 + hicache storage 类 +
     spec/dspark 纯单测件 + spec/dflash 主件 + mock canary self_unit 族——
     全账本落 docs/ci.md L1.5c；可直跑面=PD 分离 33+dspark 13+mock canary 7
     文件 **694 测+200 子测全绿零适配**；hicache 1 合法 skip（umbp）+
     3 快错（HF gated 401 上游测试设计依赖外部令牌）+6 挂（mooncake 服务
     连接/boot 级）；dflash 2 件全 setup-E+挂归批 E——零静默；
   - 批 E（多进程/boot 级，择件+改 mock）：disaggregation 顶层 21 件、ep 8 件、
     mock_model e2e_{pd,cp,pp,tp}——多数需双 boot 或多卡，逐件定"直跑/改造/
     登记不做"三态，**不硬塞 L1**；产出建议=独立功能段或夜间/发版档；
   - 批 F（红线预期管理）：MTP 三件 + dspark verify/parity 面——预期非全绿
     （R-07 台账在案），发现即登记，不设晋级门。
2. **首跑分诊三态制**（预期不做全绿承诺）：PASS / 合理 SKIP（AMD/无 GPU/
   依赖缺失）/ xfail-KNOWN_FAILURES 登记；非预期失败逐件归因（桥失配 vs
   上游已知 vs 平台限制）——mini KNOWN_FAILURES 账本落 docs/ci.md。
3. docs/ci.md L1.5 说明同步（批 A/B/C 落地各自同 commit）。

**验收标准**
1. 三批分发连续两跑稳定（不炸段、不拖垮 L1 全门 GREEN）。
2. 分诊三态清单 + KNOWN_FAILURES 账本落 docs/ci.md。
3. GLM-5.3-Next 五件与 qwen3_5 四件全部有处置（PASS/SKIP/登记三态之一，
   零静默忽略）。
4. 功能区（批 D-F）：批 D 可跑面清单 + 批 E 逐件三态处置表 + 批 F 发现
   登记（MTP/verify 类不设晋级门）；全部落 docs/ci.md，零静默忽略。

### R-15 性能回归门（相对阈值 + 历史锚定 + 三级防线）

**背景（2026-10-19 评审拍板，协调人 上游 perf 面查证后设计）**
- 上游有三层性能测试（basic_perf/ 11 件、perf/models/ 2 件、kernels/
  benchmark/）：`check_perf(at_least("output_throughput", …, 2660,
  amd=2100))`——boot 真模型→bench→对**硬编码绝对阈值**断言，跌破即红，
  等于把性能回归做成单测。**机制可偷，数值不可收**：阈值锚定 NVIDIA/AMD
  官方 CI 机器，BW1102 无参照意义，直收必假红。
- 我方已有性能资产（散在役里，非常驻）：A/B 臂间 bench（R-11/R-02）、
  catalog 晋级 bench 留档门、冷编译污染教训（R-11 验收实测 0.0×→热
  1.9×）、L1 pick_gpu 空闲选卡。缺的是把它们固化成机器。
- 设计原则（与北极星挂钩）：性能主战场=host 提交路径与图节点瘦身——
  该层 e2e bench 最钝、**op/图级微基准最敏锐**（单次 launch us 级差异
  e2e 不可见），故微基准常驻优先级高于 e2e 吞吐门。

**设计定案（三件）**
1. **相对阈值，不做绝对断言**：断言①臂间比（triton vs emu ≥N×，防
   "换装悄悄变慢"）②历史比（本轮 vs 基准记录不跌 M%；基准值入库可学
   catalog 档案，换机器/驱动重锚定）。绝对值只记录不设门。
2. **宿主三分离**：跑前 GPU 空闲检查（pick_gpu 现成）+ bench 预热段
   强制（冷编译假阳在案）+ 结果带环境戳（卡型/驱动/时钟/驻留占用）。
3. **防线分级**：L1 进**亚秒级 op/图级微基准**（量 launch 开销/图节点数/
   热点算子 us 级——R-02 vec 化 21.6ms 图发射那类信号的常驻哨）；e2e
   吞吐门放**夜间/发版档**，与 R-10 模型轨合流（同一次 boot 同时收四门
   与性能门）。

**工作内容**
1. 微基准工具面：op 级计时器 + 图节点/launch 计数器（prof_torch/
   hipGraphLaunch 前役手法固化）接进 L1 新段（阈值/source-of-truth
   沿 catalog 或独立 json）。
2. e2e 性能门脚本化：basic_perf 的 run_bench_serving/check_perf 机制
   借鉴，断言改相对；挂夜间档与 R-10 合流。
3. 基准档案：首跑留档各微基准/e2e 参考值（含环境戳），登记重锚定规则
   （换卡/换驱动/大版本必须重锚）。
4. docs/ci.md 段说明同步（同 commit）。

**验收标准**
1. L1 微基准段常驻 GREEN 且耗时可控（段级超时实测定）；无 GPU 扰动
   （跑前后卡况检查在案）。
2. 至少一次真实防尾示范：人为植入性能回归（如临时换 emu 臂）微基准
   翻红（红→绿回放）。
3. e2e 性能门在夜间/发版档跑通一次（哪怕值仅入档），基准档案+重锚定
   规则落 docs（ci.md 或 models.md，不新建文件）。

### R-16 skill/CONTRIBUTING 与门禁体系同步维护（多人协作上手守则）

**背景（2026-10-19 评审定：同事将加入分工协作）**
- skill 是新同事的第一入口；首轮补丁已随本行登记交付（见 R-16 总表行
  状态），本行跟踪**剩余同步面**，防止门禁体系演进后守则再次漂移。

**剩余工作**
1. CONTRIBUTING.md MR 门禁段与 docs/ci.md 对齐（kernels+adapter 收集面、
   l15 上游单测零拷贝条款、一键 L1 用法——CI 扩容 R-14/R-15 落地后复查）。
2. skill references/flash_attn wheel 兼容段（2026-10-10 实测）复核刷新
   （vendor 新轮就位即改写，勿留过期结论）。
3. 新同事实际上手一轮后回捞 skill 缺口（首跑踩坑即 one-pit-one-test
   补进黄金法则/参考文档）。

**验收标准**
1. CONTRIBUTING/ci.md/skill 三处门禁描述一致（一处改三处对）。
2. 无过期日期标注的"实测"结论（凡带日期的段落随复核刷新）。
3. 新同事首PR不因守则缺失返工（守则失效项即为 skill 待补项）。

### R-17 上游 release 分支策略（rel/v0.5.x 双向同步）+ v0.5.22 升级适配役

**背景（2026-10-19 评审拍板；触发=上游 v0.5.22 已 release）**
- 双线模型（评审定）：**adapter master ↔ 上游 main**（每日追新/竞速线，
  upstream-watch.md 巡检规约承载）；**adapter rel/v0.5.x ↔ 上游 release**
  （生产线：版本锁定，serving A/B 验证过的调优沉淀于本线）。上游出新
  release 分支即及时建对应 rel 分支。
- 真实现状=单 master 混线：生产调优（fork 转正/paged_mqa 晋级/ar_norm…）
  全部验在 v0.5.21 容器却落在直跟 main 的 master——上游 main 激进变更
  翻车时生产验证过的资产连坐。分线即解。
- upstream-watch.md 已有"日常追 main + release 日对账"巡检与自主修复
  授权边界；分支拓扑为新增面，落 CONTRIBUTING §1 与 upstream-watch §3
  互链。

**五条策略规则（随本役写入文档）**
1. **回流 cherry-pick 不 merge**：rel→master 定期（每役/周节拍）分析
   rel 调优 commit——通用修复/性能优化回流，版本特化垫片（needle 依赖
   /版本 gate）留 rel；判据可执行化=catalog 状态与 commit 载体复查。
   反向 master→rel 同节拍（新桥成熟/serving 需要即回流生产），双向记录
   （rel 线回流 commit 建议加 `sync:` 前缀）。
2. **三绑定铁律：分支↔上游树↔容器**：rel/v0.5.22 改动必须在
   /sglang-v0522 树+对应容器跑门禁；l1_gpu_gate.sh 现自动探测
   /sglang-v0521——不参数化即静默错树假绿（ci.md 环境铁律教训的双线
   放大版），本役先加 `--up-tree`/env 参数。
3. **支持窗口 N=2**：最新 release 线 + 上一线（仅 bugfix），更老 EOL
   只读归档；每线一间容器。
4. **catalog 档位与线绑定**：production/validated 证据注明验证所在
   rel 线；跨线结论优先复测（线上 validated ≠ 线下 validated）。
5. **主仓分支拓扑（2026-10-19 评审补约束）**：主仓（现 Gitee，后期可能
   迁 GitHub——托管平台无关的规则措辞）**只保留 main 与 rel/v0.5.x 发版
   分支**；日常追新线与发版线的对应关系在主仓分支名上直接可读。个人
   调试/特性分支一律 **fork 到自己的账户仓**调试，成熟后向主仓提 MR
   （主仓不承载个人 WIP 分支）；主仓移除 travel 类调试分支（现有历史
   调试分支在迁分支拓扑时清点处置）。
6. **`ci/upstream_diff.py`（在案计划）= 多线前置工具**：脆弱桥带
   版本/schema gate 一份代码跨版适配是对，靠对账工具才可控；本役顺带
   催生最小版（registry+TEXT_PATCH_ALLOWLIST 驱动的静态对账），把升级
   成本从人肉 needle 普查降为跑对账。

**工作内容（v0.5.22 升级役=策略首演）**
1. 建 rel/v0.5.22 分支；容器并置 v0.5.22 树（/sglang-v0522，双树共存
   有先例）；l1_gpu_gate.sh 树参数化。
2. 适配：needle/签名普查（upstream_diff 最小版优先，人工兜底）→ 适配
   commit 落 rel/v0.5.22 → L0/L1/L1.5 全门 + 四门 → production 宣布；
   v0.5.21 线转上一线（仅 bugfix）或 EOL 判定，记录在案。
3. 策略文档落地：**已完成（2026-10-19，含变更）**——用户同日定
   CONTRIBUTING.md 弃用、内容收编本看板「协作与分支规约」节（原 §1
   改写为双线模型三绑定+**主仓分支拓扑**（仅 main+rel；fork 内任意命名、
   MR 标 base 线）+cherry-pick 节拍；§2-§6 全文平移；README/architecture/
   governance/reading-guide/upstream-watch/skill 全部外链改指 board）；
   upstream-watch §3 release 流程挂分支动作 + ci.md 环境铁律补三绑定句
   同批；**主干定名——已完成（2026-10-19）**：master(187b870) 快进合入 main
   并推 Gitee（零冲突纯 ff），Gitee 默认分支已切 main、远端 master 已删
   （ls-remote 仅剩 main）；**历史调试分支清点——已完成（同日核验）**：
   本地与远端分支清单均只剩 main，零残留可关账。
4. master 未落地的生产调优清点：与 rel/v0.5.22 状态对账，确定各自
   回流/重验清单（首次双向同步演武）。

**验收标准**
1. rel/v0.5.22 生产可用：零 env 四门 + L1 全门全绿（v0.5.22 树）。
2. 策略三文档落地互链；错树防护（树参数）落地并有防错树测试证据。
3. 双向同步各演示 ≥1：rel 调优 cherry-pick 进 master（L0 绿）、
   master 新桥回流进 rel（门禁绿）。
4. v0.5.21 线处置（上一线/EOL）有明确记录；catalog 带线标注成形。
5. 主仓分支清单=main+rel/v0.5.x（无个人调试分支）；fork→MR 流程在
   board.md「协作与分支规约」节成文且一个真实新同事 fork-MR 演练走通
   （含门禁回帖）。

### R-18 tvm-ffi dlpack 冷缓存自递归——adapter 预热桥

**背景（2026-10-19 评审拍板方案3；事故档案 models.md 2026-10-18 运维役行）**
- 递归链解剖（tvm-ffi 轮子 bug，非 adapter 代码）：
  `_optional_torch_c_dlpack.py` 导入执行发现缓存 .so 不存在 →
  `subprocess.run(python, utils/_build_optional_torch_c_dlpack.py)`（导入期
  spawn）；构建脚本 `import torch.utils.cpp_extension` 触发 CUDA 初始化 →
  torch 加载又 import tvm_ffi → 同模块再次导入执行、缓存仍无（父进程还在
  编）→ 再 spawn。每层 ~24s，5 分钟观测 119 进程。
- 关键事实：构建脚本 L732 **自己会设 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`**
  （轮作者认知里的递归闸），但 **spawn 之前没有预设**——只护了子进程端，
  没护进程级重入；脚本注释自证防的是函数级循环 import 而非进程级。
- 现役缓解=bootstrap §3 手工配方（`TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
  直构预热或健康容器拷 .so）——人肉运维面，清缓存即复发。

**工作内容（方案3：预热桥代码化）**
1. 新桥 `sgl_kernel/_adapters/dlpack_warmup.py`：且仅当
   `~/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch{maj}{min}-<device>.so`
   不存在时，一次性 `subprocess.run` 构建脚本并注入
   `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`（= bootstrap 配方的代码化）；
   失败不致命——log-once 降级走纯 python dlpack 路径。device 判定与
   libname 规则与轮子 L120-151 逐字对齐（torch 版本+cuda/hip/cpu）。
2. sitecustomize 挂载：位置在 hip 门内、任何 tvm_ffi 可能被 import 之前
   （hcu_env 之后不受影响；顺序契约测试
   test_sitecustomize_sequence.py 结构层同步更新）。
3. 登记：env-vars.md（新仪器行为=自动预热，可控开关）+ BACKEND_CATALOG
   （模型/运行时桥条目）+ bootstrap §3 改写（配方降级为"手动兜底"，
   推荐自动桥）同 commit。
4. 结构测试钉：test_sitecustomize_sequence.py 增断言——缓存缺失场景（
   仿 env 切 TVM_FFI_CACHE_DIR 到空目录）下预热桥触发且导入不挂。

**验收标准**
1. 冷清缓存容器（含 Windows/WSL 侧不可用场景可只清 /root/.cache/tvm-ffi）
   import `sitecustomize` 一次成功、无 fork 风暴（进程数恒 1）、
   预热后加速层 .so 在位。
2. 递归免疫回归钉常驻（结构测试），L1 全门 GREEN。
3. bootstrap §3 与 env-vars.md 同 commit 更新，运维手册不再依赖手工配方。

### R-19 JIT kernel vendor 化——需求清单提给 aiter/DTK 维护方（预编译随 wheel，我们只管调用）

**背景与拍板**（2026-10-20）：aiter wheel 实证"满预编译、import 秒级、
零运行编译"是可达态（bootstrap §4c，51 颗 .so 随包发）；而 sglang 侧
JIT 35 模块全靠 CI/首跑自编，v0.5.22 上 38F 里 32F 是编译失败（根=CUB
头缺失+hipcc uint2 语法缺口，均 DTK 工具链侧）。用户拍板：JIT kernel
一律提需求给 aiter/DTK 维护方维护，adapter 只做调用面；全接入后 CI 不
再承担预编译时间。

**需求单内容（提给 vendor）**：
1. 清单=sglang JIT 全 35 模块（键控 `~/.cache/sglang/jit/gfx93x/sgl_kernel_jit_*`，nmz26 现名可点；大头=deepseek_v4 家族/dsa/mamba causal_conv1d/elementwise/fast-hadamard/kimi_k3 等，见 L1.5e/L1.5g 账）；
2. 交付形态=预编译模块随 aiter wheel（或独立 sgl-kernel vendor 包）发货；模块名键控须带 wheel/dtk 版本对齐约定（aiter 现行操作名裸键存在遮蔽坑，见 bootstrap §4c 铁律）；
3. 工具链修复同车：hipcc 默认打通 CUB 头路径+支持 `uint2` 结构化绑定（cuda-mocker 配方三件套=过渡期自救，见 L1.5g）；
4. **2026-10-20 锚定轮补单（三模块+一构建器）**：①`dpsk_v4_topk_v2`（v0522 针脚 transform_packed→transform_paged 失配=当前 mech 38F 中 17 败行的首恶）；②`compress_plan` c_plan.cuh :562 PrefillPlan 运行期路径（隔离形状可过、套件形状炸）；③`per_token_group_quant`（`__HIP_SATFINITE/__HIP_E4M3` 针脚+`SGL_CUDA_ARCH=750` 映射触发 vec 16B 上限静态断言=arch 映射层面受限，vendor 侧提映射或降 vec 方案）；④tvm-ffi 选件构建器同车修（fork 子进程再 import torch/HSA 的自锁+无超时，R-18 根因——wheel 预编译 addon .so 随包即彻底免疫）。

**过渡期（vendor 接单前）**：cuda-mocker 配方继续有效（可编/可测/数值对拍 PASS）；cache 播种管道（bootstrap §4d）管住新容器首编成本。

**验收标准**
1. vendor wheel 到位后 35 模块零 JIT 编译直接 import（靶态=aiter 同款开机行 prebuilt=N runtime_built=0）；
2. CI 端到端去除 JIT 编译时间——新容器 kernels/l15 段窗回到纯测试耗时，l1_gpu_gate 的 JIT 相关 known-fail 预算线归零重锚；
3. 需求单 (1)(2)(3) 逐项有 vendor 回执/版本号落档。

## 项目北极星与长期方向（背景板，不单独开需求号）

**北极星（评审拍板，2026-10-17 重申）**：官方 sglang 代码零修改，拉下来挂上
adapter 即在 HCU 上跑通，且**性能正常**——能跑不算数，性能要对得起硬件。
当前性能主战场在 host 侧提交路径与图节点瘦身（hipGraphLaunch 线），不只算子本身。

三条推进线：

1. **调测效率**：官方单测（`l15_upstream` 段）代替端到端做日常验证；模型 e2e
   降频为发版/夜间。校准点：官方测试是**过滤器不是裁判**——发现适配层破坏
   即可，质量金标准是我们自己的三段式单测。
2. **自有单测资产**：三段式单测（参照/形状矩阵/容差）持续汇总，覆盖面**精心
   挑选**——不求全，锚定上游契约面 + 我们的接入面（影子符号面、桥的接缝面），
   目标是对新版本、新模型、新功能**快速支持**；新接入先补单测再上 e2e
   （CONTRIBUTING §3.4b test-first）。
3. **档位演进**：调试期 emu（稳、金标准）→ 性能期 triton 自研 → 成熟转
   vendor（aiter/lightop）。单向晋级、逐 catalog 条目按证据门禁推进
   （对拍 + bench + catalog 晋级）；emu/triton 换下不删、转 ref 角色只留
   离线测试。校准方向不变：emu 校准 triton；emu/triton 暴露 vendor bug；
   vendor↔triton 是性能竞争关系。
   **emu 清退清单注记（2026-10-17 协调人 对账，现役剩 6 处）**：①KDA prefill
   （triton 已写好 473–1009×，卡 R-01 门）②conv1d fn/update（16.4×/4.3×
   同卡 R-01 门）③fa3 prefill/varlen 面（decode 已 triton 化）④paged MQA
   logits（vec 化 emu 现役，vendor 替换=R-02）⑤kpool write ⑥flash_mla/
   dsa_paged_mqa 面。清零路径 ≈ R-01（双解锁）→ R-02 → fa3 prefill →
   收尾。已换装：silu/rms_norm/rotary/moe_gate（triton production）、
   GDN 线性注意力（aiter）、fa3 decode（triton）、MQA vec。
   **更新（2026-10-18）**：R-01 竞态定罪 custom AR（NCCL 缓解 60%→2.5%）
   已置待验收——若验收通过，①②两笔（最肥的）随即解锁换装。
4. **融合沿用（2026-10-17 评审定调）**：is_cuda 语义红利的兑现方式——上游
   融合入口（flashinfer/sgl_kernel 路径）默认**保持融合粒度**承接
   （vendor 融合核优先，其次自研 triton 融合核），拆解/降级为例外且必须
   显式登记（catalog 注记+理由），禁止静默拆解；正确性冲突时先保正确、
   vendor 融合核就绪后回接（mHC 先例：先 UNFUSE 保语义，deepgemm
   `tf32_hc_prenorm_gemm` 融合核可回接）。盘点见 R-09。

**硬件特性备忘（量化/内核选型的决策依据，2026-10-17 评审方提供）**：
HCU 卡的通用 core 与 tensor core **不能 overlap**——NVIDIA 上靠"矩阵核
乘 + 向量核同时处理 scale/量化"藏掉的开销，在本卡上**串行计费**。因此：
- 生产模型量化主路线 = **channel-wise**（scale 处理最少、可落 epilogue）
  + **Marlin W4A16**（量化成本离线 repack 挪走 + 自研 asm 内核按本硬件
  调度写）；block-wise 分组 scale 在 GEMM 主循环里的向量活结构性吃亏，
  对 block-wise fp8 路线的性能预期**不要按 CUDA 经验设目标**。
- 自研 deepgemm/marlin 与 CUDA 版不仅接口可能不同，**性能模型是另一套**
  （R-04/R-05 摸底与验收 bench 都以此为前提）。
- channel-wise int8 Linear 上游原生支持且已双卡型验证 e2e。

## 协作与分支规约（原 CONTRIBUTING.md 收编）

> 2026-10-19 用户定：CONTRIBUTING.md 弃用单文件，内容收编进本看板统一维护；
> 外链（README/architecture/skill）统一改指本节。上位治理见
> [docs/governance.md](governance.md)，MR 门禁细节见 [docs/ci.md](ci.md)。

# 贡献指南（多人协作约定）

> 先读 [docs/architecture.md](docs/architecture.md)（分层与机制总览）与
> [skills/hcu-adapter-dev/SKILL.md](skills/hcu-adapter-dev/SKILL.md)（开发规约），
> 再读 [docs/ci.md](docs/ci.md)（MR 自动测试怎么跑）。本文只讲协作流程与红线。

### 1. 分支与 MR 模型

**双线模型（R-17 评审定）**——主干分支与上游 sglang 分支一一绑定：

- `main`（**追新线**）↔ 上游 `main`：每日追新/竞速，日常巡检与适配落这
  （[docs/upstream-watch.md](docs/upstream-watch.md) 承载）；
- `rel/v0.5.x`（**生产线**）↔ 上游 release 分支：版本锁定，serving A/B
  验证过的调优沉淀于此；上游出新 release 分支即从 main 及时建对应 rel
  线（upstream-watch §3 流程）。支持窗口 N=2：最新线 + 上一线（仅
  bugfix），更老线 EOL 只读归档。

**三绑定铁律**：分支 ↔ 上游源码树 ↔ 容器。`rel/v0.5.22` 的改动必须在
`/sglang-v0522` 树 + 对应容器上跑门禁（`ci/l1_gpu_gate.sh --up-tree` 或
`HCU_UP_TREE` env 指定；缺省自动探测 `/sglang-v0521|/root/...`，指定了
不存在的树则 loud fail 不静默）——错树假绿是环境铁律教训的双线放大版。

**主仓分支拓扑**：主仓只保留 `main` 与 `rel/v0.5.x` 发版分支（托管平台
无关的规则——现 Gitee，后期或迁 GitHub）。个人调试/特性分支一律 **fork
到自己账户仓**调试，成熟后向主仓提 MR；主仓不承载个人 WIP 分支。

- MR 分支命名 `feat/<域>-<短名>` / `fix/<域>-<短名>`（fork 内命名随意，
  MR 标题与 commit 前缀对齐即可）；**MR 必须标 base 线**——追新修法进
  `main`，生产修复进 `rel/v0.5.x`。
- 线间节拍（R-17 规则 1）：rel 通用修复/优化定期 cherry-pick 回 main
  （`sync:` 前缀记录，**不 merge**，版本特化垫片留 rel）；main 新桥
  成熟/serving 需要时同节拍回流进 rel，双向留痕。
- MR 提交后自动过两层门禁（见 docs/ci.md）：
  - **L0 结构门禁**（Gitee Go 云端，秒级）：语法 / LF / 密钥扫描 / 注册表交叉一致；
  - **L1 GPU 数值门禁**（自托管 watcher，容器空闲卡，~10 分钟）：算子单测 +
    五个门禁脚本 + vendor 探针，结果以评论回帖到 MR。
  - L1 全绿（或仅 WARN）即可合入；改动 `sitecustomize.py` /
    `hcu_adapter/bootstrap.py` / `hcu_adapter/bridges/` / `_base.py` / import
    接管逻辑的 MR，另需一次 e2e 冒烟（cookbook 配方）再合。
- 本地提交前先自跑 L0：`python3 ci/l0_gate.py`（零依赖，秒级）。

### 2. Commit 风格

一行式 `域: 描述`，中文英文均可，描述讲"做了什么+为什么"，长背景进 body：

```
cp: prefill context-parallel bridge (HCU_PREFILL_CP=1) - platform gate + rope guard ...
moe: aiter bridge engagement/debug logging (HCU_AITER_MOE_DEBUG)
ci: tiered MR auto-test -- L0 + L1 + watcher
docs: architecture collaboration chapter + html diagram
```

常用域：`cp` `moe` `attn` `rope` `gemm` `quant` `ar`（allreduce）`pd`（分离部署）
`ep` `ci` `docs` `tests` `env` `config`。

### 3. 三类改动的 Checklist

### 3a. 加 / 改算子（`sgl_kernel/*.py`）

0. **平台语义前置**（governance §0.5）：接入点的平台语义走 is_cuda 分支——
   上游新符号（含 flashinfer/trtllm 等专有库调用）按四级路由找归宿：
   lightop → aiter → **triton 先行**（先跑通）→ 响亮 stub；fp8 fnuz 等
   语义护栏白名单锁定不随翻。triton 先行产物在 vendor 就位后转 ref 档。
1. API 签名与官方 `sgl-kernel` **逐字对齐**（含 `out=`、占位参数、destination-passing）。
2. 按三档实现：triton 首档（fp32 累加、零主机同步）→ vendor 封装 → torch 兜底。
3. 单测三层归位：`tests/test_ops_cpu.py`（torch 兜底）、`tests/kernels/`（GPU +
   CUDA-graph capture 回归）、`tests/vendor/`（高危 vendor 压测 + 快关 gate）。
4. 走 op 级 env / yaml 开关的，同步登记：`hcu_env.py` REGISTRY +
   `docs/env-vars.md` + `hcu_adapter.yaml` 示例键 + `hcu_config.py`
   BACKEND_CATALOG——**L0 交叉一致检查会挡缺漏**。
5. 高危算子（allreduce / attention 类）接入即配：压测（尺寸×dtype×graph-replay
   变输入）+ `kill_switch` 快关。

### 3b. 加桥 / 加 plugin（先分仓，再落位）

1. **先读 [docs/governance.md](docs/governance.md)**：双仓边界、薄入口与宿主
   分发终态、裁决链（seam → rebind → text-patch 白名单）、active/ref 角色
   都在那里，本文是执行项。
2. **先分仓**：上游在 AMD/CUDA 上**有**而 HCU 上缺的行为 → adapter 仓的域桥；
   上游**也没有**的能力 → **独立 plugin 仓**（经上游 seam 或注册机制接入，
   可直调 lightop/aiter 等底层库），MR 不进本仓。注意 plugin 的目的同样是
   **性能优化**（算子融合、专项内核、新调度路径）——动机是性能不改变选仓，
   判据始终是"上游有没有这个行为"。plugin 有意上游化时（自研格式/新算子
   push 上游拿正式 seam），按 governance **§5.1 演进轨迹**走：接入 → 证据
   积累（格式契约提前固化为文档+契约单测）→ 上游 PR → 过渡桥退役。
3. **本仓落位**：宿主级行为还原放 `hcu_adapter/bridges/<host>.py`，算子/
   机制桥放 `sgl_kernel/_adapters/<域>.py`（install() 入口 + 开关 + 安装成功
   `registry.register(...)` 登记）；`sitecustomize.py` 固定三行入口，
   `bootstrap.py` 只做探测与分发——**不新增**实现体到两者，review 拒收。
4. **先写根因注释**：上游哪个文件哪个判定漏了什么、为什么桥是解法（现存每座
   域桥都是范本）。没有根因注释的桥不予合入。
4b. **测试先行顺序**（新模块默认"先单测后 e2e"，比"先跑模型再解问题"省一
   整轮起服务-爆-倒查循环；实证：contig-triton/fa3-emu 单测先行均 e2e 首过，
   moe_align/slimquant 反序付出 e2e 倒查代价）：
   - **算子/数学类**（triton 核、emu、量化路径）：单测先行是**硬门槛**——
     有了 ref 对拍 + 形状矩阵 + 容差判据，再上 e2e；脚手架照
     `tests/kernels/test_gemm_int8.py` 三段式（参照/形状矩阵/容差），pattern
     见 skill `references/test-suite.md`。
   - **seam/接缝类**（import 时序、capture 交互、平台判定、跨卡织物）：
     单测表达不了的（如 graph capture 里的主机同步）先以 e2e/探针探路，
     但**"one pit, one test"**——探明的每个坑，修复 MR 必带能离线复现该坑
     的单测，无离线复现不予合入（沉淀进 tests/ 即质量资产，L1 自动收集）。
   - e2e 前先跑既有相关门禁（`tests/tools/_test_*`），L1 全绿是 MR 准入。
5. 优先用 `_base.py` 的机制通道，不动上游源码：
   - `extras`（allowlist 填空）——真包缺的符号我们补；
   - `overrides`——需要整体夺回某函数时；
   - `post_hooks`——包加载后 wrap 类方法 / 判定词；
   - 模型文件级判定词才用 post-hook wrap（按名 import 的模块
     必须走这条，直接改属性不生效）。
6. 桥必须有开关（env / yaml），默认行为与上游一致或已验证更优。
7. **登记运行时注册表**：`registry.register(op, impl, target, mechanism)`——
   op 是 catalog 键，mechanism ∈ seam/rebind/text-patch（text-patch 针串需在
   `TEXT_PATCH_ALLOWLIST` 在册）。这一步换来启动日志"每算子一行谁在跑"
   和 `HCU_ADAPTER_DUMP` 机读 JSON；L0 `registry-ops`/`text-patch` 会挡漏登。
8. **routing glue 都只准写 glue**（sitecustomize/bootstrap/宿主桥中的路由、
   签名适配、断言放宽），不准出现 triton import、`@triton.jit`、`tl.*`——
   算子数学一律放 `sgl_kernel/`
   （governance §1 红线，L0 `glue-numerics`）。
9. ref（诊断）实现同样登记 `role="ref"`，但**永久留在离线测试路径**
   （`tests/kernels/`），任何桥不得把它 bind 进运行时。
10. 存量 sitecustomize 长桥的迁出按 architecture.md §9 P1-P3 节奏做，
    每次迁移容器 e2e 复验；不要在别的 MR 里顺手搬（迁移与改逻辑分离）。

### 3c. 加环境变量 / 配置键

三原则：**正确部署零环境变量**（默认值即最优解）；**显式设置永远最高优先**；
**启动日志打印每个生效值及来源**。落地 = REGISTRY 登记 + env-vars.md 手册行 +
yaml 示例键，缺一被 L0 挡。

### 3d. 推广边界（团队协作红线）

adapter 仓与独立 plugin 仓并存后，防止边界随时间模糊的五条（违反即 review 拒）：

0. **融合沿用原则（R-09）**：is_cuda 路由选中的上游算子融合尽量保留——
   拆解为例外、禁止静默拆解、正确性优先；上游融合面四态对照见
   `docs/fusion-ledger.md`（vendor 融合 / triton 融合 / 已拆解须登记 /
   缺失排期），新增拆解不同步 BACKEND_CATALOG 即违规。
0. **先分仓再写码**：上游没有的新能力（算子融合/专项内核/新后端/新机制，
   即便动机是性能优化）不进本仓，落 plugin 仓经 seam/注册接入或直调底层库；
   本仓只收行为还原（governance §0 判别问句）。
1. **选位不跳级**：行为还原需求按 governance §3 裁决链选实现手段，
   MR 描述里写明选位理由（"seam 不可行因为…"）。
2. **单 active 指针**：一个 catalog 算子同一时刻只有一个生产实现；想换实现
   = 改注册表指针 + 走门禁，不是再抄一份。
3. **登记即存在**：没有 catalog 条目、没有 register 记录的实现不存在
   ——排障只认 `HCU_ADAPTER_DUMP` 的输出（plugin 仓的接入同样要求登记）。
4. **组合归因不重跑矩阵**：集成问题先看 dump（谁在跑），再 yaml 翻单条目
   二分；禁止"全绿矩阵跑三遍"式排查（governance §4）。

### 4. tests/tools 命名语法

`tests/tools/` 是物料库不是垃圾抽屉，新文件必须带 docstring（用途 + 运行方式）：

| 前缀 | 语义 | 生命周期 |
|---|---|---|
| `_test_` | 门禁脚本（可被 CI / 回归引用） | 长期维护，改坏 CI 会红 |
| `_probe_` | 环境/能力/e2e 行为探针 | 随特性存活 |
| `_launch_` | 生产启动配方（模型+旗标固化） | 随配方存活 |
| `_dbg_` | 一次性调试取证 | 用完可删，禁止被引用 |
| 无前缀（`run_triage.py` 等） | 跨任务长期工具 | 文档化于 SKILL.md |

详见 [tests/tools/README.md](tests/tools/README.md)。

### 5. 共享节点与双树纪律（红线）

- **本地仓（编辑树）是唯一改码处**；HCU 容器内的仓只做运行验证，**禁止在容器里
  改代码再手拷回来**——两树漂移曾造成 API 对不上的实 bug（elementwise 事故）。
  容器树与所属分支漂移时，以 Gitee 对应分支为准：`git fetch && git reset`。
- 共享 GPU 节点上**只操作自己标识前缀的容器与进程**（每人用自己的名字
  缩写做前缀命名容器，如 `zhang3_*`、`li4_*`；只动带自己前缀的，别人的不碰）；
  杀进程按端口精杀，
  不做宽匹配 `pkill -f`。长任务必须 `timeout N` 前缀。容器内改码/部署走脚本
  文件，不拼超长内联命令。
- 任何令牌（Gitee push token、CI token）**不落仓库、不进 commit**；临时写入
  `.git/config` 用完即还原。L0 密钥扫描是兜底不是借口。

### 6. 版本升级（跟随上游 sglang）

升级 = 对齐 import 面：官方 `sgl-kernel` python API diff 一遍 → 补齐新符号的
影子实现 → 跑 L1 + cookbook 冒烟。影子库机制的价值就在这：上游模型代码不碰，
只追算子层。上游出新 **release 分支即从 main 建 `rel/v0.5.x` 线**（双线
模型 + cherry-pick 节拍见本文件 §1；大版本普查先跑
`_probe_vendor_inventory.py` 类盘点再动手）。日常追新巡检（对账工具 +
日流程 + 自主修复授权边界）见 [docs/upstream-watch.md](docs/upstream-watch.md)。

## 验收记录（在办任务的过程性证据）

> 已完成任务的验收记录随清行移除，证据沉淀于 models.md 战报 / ci.md /
> 对应 commit。下表只留在办任务的过程性证据。

| 日期 | ID | 结论 | 证据 |
|---|---|---|---|
| （在办过程证据追加处；完结即移除） | | | |
