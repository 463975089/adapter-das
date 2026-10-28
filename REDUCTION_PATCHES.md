# REDUCTION_PATCHES — 减核役批量账（emu→triton/融合，单测先行，serving 集中验证）

单批工作流（用户定向）：逐项 parity（含 fp8 ulp 语义门）+ 秒级微基准 → 批量完成后
**一次**集中 serving A/B（nmz18 dsv41f dp8ep8+DSPARK，四门 + bench_step p50）。
不做逐项 boot 对比实验（用户 2026-10-23 指令：prof 发现问题直接解）。

## 批次账（2026-10-22/23，B 臂 = 全融合）

| # | 改动 | 文件 | 单测判决 | serving 效果 |
|---|---|---|---|---|
| 1 | DSV4.1 sparse rope/norm torch fallback → triton 门（`_fused_dsv41_kernels` needle） | `hcu_adapter/dsv4_sparse_triton_gates.py`（7508c7d） | bitwise 10/10 + probe 3/3 | B 臂组成 |
| 2 | silu_and_mul_clamp × per-token-group-quant 融合为一核（`silu_and_mul_clamp_quant`，GPB=2 重铺修复大 M 断崖）+ **针 v3 挂点修正**（routed arm 锚 `act_and_mul_triton` else 臂，非 `silu_and_mul_clamp`；MISS/ENGAGED 一次性诊断） | `hcu_adapter/dsv4_silu_fused_quant.py`（54d45d0 → 6ef8b3e） | fp8 ulp 门（pair-bytes 逐值 + rtol 0.08）；大 M 0.42×→1×+ | B trace act_and_mul 104ms/步段收回（wall 21.65→19.62s）；B-v2 ENGAGED 四门绿，p50 975ms |
| 3 | MHC `_HC_MIX_DOT_PRECISION`/`_HC_MIX_NUM_STAGES` import-frozen 常量针（tf32x3 崩→ieee fp32 MFMA 路） | `hcu_adapter/dsv4_mhc_stats.py`（f74faec） | B 臂 boot 绿（原 tf32x3 compile 崩） | B 臂可跑的前置 |
| 4 | **MoE combine emu 内置 triton 化**：`moe_sum(_reduce)` 原为 aten 三连 `sum→mul→copy`（=trace in-graph `reduce_kernel<128,4>` 467µs×40 层/prefill 步 + 8×116ms 巨怪 + 每层多两趟 elementwise）；换 `_moe_sum_scale_kernel`（fp32 acc ×rsf 单次舍入，一核零中转，BLOCK_D=512/w4） | `sgl_kernel/moe.py`（6ef8b3e） | rsf=0 bitwise；rsf≠0 ≤1ulp（=上游 stock `moe_sum_reduce_triton` 语义）；M=1024 422→74µs（5.7×,848GB/s）、M=8192 3257→563µs（5.8×,895GB/s）；decode 形状（16/80）持平 | p50 975→925ms（四门绿，silu ENGAGED 复验） |

## 归因方法论（可复用）

- in-graph kernel 的 Python 栈不可用（replay correlation 对不上）→ 邻域指纹
  （prev/next kernel 名）+ e2e probe（publish(server_args, role, ranks) 前置，
  真链 `inplace_fused_experts` eager 跑，cpu_op 带形状）双轨定位。
- step 分类：按 graph replay 的 kernel 数分段（255=draft 图 / 4121=verify 图），
  不要按 correlation 全局分组（非图 kernel 污染）。
- 每 40 步 Decode-batch 日志行取间隔 = bench_step p50，不受 profiler 影响。

## 未竟（按优先级）

1. **116ms GPU 停摆（「拍」）**——见 docs/models.md 2026-10-23 行：环境级
   （kernarg 池/管理面候选），`HSA_KERNARG_POOL_SIZE=8388608` 试验进行中。
   停摆不除，一切 kernel 优化只能兑现在剩余 ~36% 占空比内。
2. `fused_k_norm_rope_flashmla` 巨实例（116ms）——已被证实为停摆落点受害者，
   非自身病（同图同 grid 6µs/116ms 两态、期间全 GPU 零并发）。停摆解决后复测。
3. `apply_rotary_emb_triton`/`_rmsnorm` decode 步各 ~5-16ms/步——量级远小于停摆，
   排后。
4. lightop 融合位（norm+rope/silu+quant 已内部完成/moe gate 冗余）——
   docs/vendor-kernels.md §2 机会表，等停摆定性后再评。

## 2026-10-22 das(bw1000) 三路对照（分析，非补丁）
- 报告：tmp_fetch/das_vs_ours_3way.md（cookbook 178 屏 737KB→185KB 去重 + das prof + 本地 das 仓）。
- 台阶：300ms →(消拍 ×2.8)~110-145ms →(vendor 地板+kernel 数砍半)~40-70ms →(deepgemm LL+INT4-W4A8+PD 分离)→17ms 级。
- 关键：das cookbook 零驱动级规避 env ⇒ 拍=nmz18 盒子特有；das 栈=sglang-das0.5.18+lightop0.6.0+deepgemm2.1.0(dtk2604/torch2.11)，P2/P3 押栈代差决策。

## 2026-10-23 拍的差分定位（nmz18 占用期零成本实验）
- 同盒同驱动，6 个外来容器驻留（vllm 0.28/0.29-dtk26041/2610 + sglang 0.5.18-dtk26041，VRAM 86%）→ 探测器 4s/15s×双卡零 >10ms gap、duty 0% 损失 ⇒ **拍非盒级驱动现象，是我方栈特有**。
- 复现电池（card0, 20s）：eager 风暴 98k kernels/s 干净；graph replay 风暴 86k/s 干净；5G 驻留 0.2% 噪音；驻留+风暴 0.1% ⇒ 触发器需真实服务完整构成（~50G/卡 + 8 rank HSA IPC/NCCL + adapter 线程群），kernarg 单因素未证实也未排除。
- 健康容器全部 dtk26041/2610 userspace；我方旧 DTK 为全盒唯一异类 ⇒ 栈升级与消拍可能同解。HSA_KERNARG_POOL_SIZE A/B 仍需 8 卡（nmz18/nmz26 皆满，staged 待空窗）。

## 2026-10-23(续) 差分再修正（用户输入 + 三项核查）
- prof 实证=FP8（channelwise rocBLAS+deepgemm fp32 累加+per_token_quant_fp8，零 int4；形状同我方）⇒ das FP8 即 17ms 级，量化非主差。
- 推翻"我方旧 userspace"：dsv41f 容器=dtk26041/torch2.11 同代；das 亦在 nmz 系调试 ⇒ 残余差分=env+代码构成。
- env 排空：expandable_segments HCU 平台不支持（das 配方该行同为摆设）；烤 env 差=OMPI 惰性+NCCL_TOPO_FILE（已 unset）；ROCSHMEM/NCCL/MC 系仅 engage deepep。可测残余=HSA_FORCE_FINE_GRAIN_PCIE=1 与 HSA_KERNARG_POOL_SIZE=8M。
- staged 四臂 /root/launch_ab1023_{A,B,C,D}.sh（A 基线/B kernarg/C fine-grain/D 双开），空窗即跑；代码构成嫌疑（engram 设备表/ROCM_HOME shim/topk v2）列 strip 梯队后备。

## 2026-10-23(下午) 剪层复现器役：拍的 engram 剂量响应首证（nmz26 卡 2-3，双卡 TP2）
- **卡 4-7 归属判定（用户质询）**：非我方残留——TP4/TP2 两次崩溃均释放干净（0 活进程 0 kfd 占用），4-7 为 1022-1023-NMZ26-dtk 三件套（Up 4h）加载期撞窗。
- **cut8 剪层 boot 配方钉死**：符号链接模型目录+改 config（num_hidden_layers 8 / candidate_source_layer_id 4 / index_source_layer_ids [2,3,4] 保序在 candidate 处收口 / dspark_target_layer_ids [5,6,7] / engram_vocab_size 缩 / engram_num_embeddings 缩 / E384 不可动-形状断言）；launch_cut8_A.sh（PYTHONUNBUFFERED 必须+chmod+x；TP2 卡 2,3，mem-fraction 0.86+max-total-tokens 8192+graph bs8）。boot ~90s，主模型 57.5G/卡+draft 7G。
- **103G→57.5G 之谜解**：engram 大表本体=engram_num_embeddings（每层 384M 项，两表 ~189G@TP8），非 engram_vocab_size。
- **拍的剂量响应（idle，stall_detect 15s×2 卡×多轮）**：engram/卡 1.5G（cut8d）双卡全净 0 gap；23.6G（cut8e，=生产 TP8 每卡尺寸）卡 3 突发 15 gap/3.1% duty、复测 0.3/0.1%（间歇）；47G（cut8f）卡 2 突发 5 gap 10-25ms（与生产 25.3/45.3ms 量子同族）后归零。**结论：engram 设备表大小=拍触发器（onset 剂量 ~23.6G/卡），但稳态 48.6% 级全幅拍还需放大器（8 卡全载/全表/流量）**。
- **剪层模型首请求崩**（vocab 8M×容量 6M 不一致疑越界→rank 死→gloo 断连）：loaded 路径复现待容量配平（每层容量=3 ngram×8 heads×vocab=24×vocab）。
- 遗产（容器 /root）：dsv41f_cut8/、launch_cut8_A.sh、mk_cut8b.py——下轮直接用。

## 2026-10-23(傍晚) engram host 路再役（用户问向：engram 放 CPU）
- **das 对照实锤**：cookbook 全配方 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 + per_rank + CPU_GATHER=0`——das 就是 host 表，与"其不拍+我方剂量响应"三角自洽。
- **注册上限翻案**：单次 cudaHostRegister >12G 死（10-21 结论保留），但**分块 12G×4=48G/进程全过**（60G 才 err=1；ulimit/cgroup 均 unlimited，纯 HSA 层）——生产 TP8 per_rank 23.6G/进程装得下。
- **host 路真死因 = 执行层**：cut8+host 表 boot，表建成功（per_rank 755M 驻留/大页 100%；shared 布局撞 shmem THP=never 慢 10×告警）后：①图捕获内 fused_engram_gate triton 首编死锁（--disable-cuda-graph 绕过）②绕过后 eager 跑 gate → **ROCR-Runtime 7.2 VMFaultHandler 段错误**（GPU 访问注册 host 内存页错误，runtime 处理器自崩）。das 栈同 env 能跑 ⇒ ROCR/驱动处理差异。
- 下步候选：注册 flag 变体（hipHostRegisterMapped/NumaUser）、全量 pre-fault 保 GPU 访问前驻留（本次 23.6G 虚拟仅 755M 驻留疑点）、或移植 das CPU_GATHER 思路（CPU 侧 gather 免 GPU 触 host）。
- 设备路剂量响应（下午役）仍是 P0 主线；host 路=das 同构解法，卡在 ROCR VMFault。

## 2026-10-23(夜) host 路 VMFault 三层根因全破 + 修复实证（nmz26 卡2 TP1 cut8）
- **此前"多表注册失效"复现是我自己的脚本 bug**：multi 单测 mmap 对象是函数局部，返回即 GC → 悬空注册 → 必崩；unit 版模块级存活所以过。机制本身无罪。
- **Bug B（剪层容量不变式）**：`EngramEmbedding.forward` shared 分支 gather **不带 row_lo/row_hi**（默认 2^62 不设界，生产全表覆盖全 id 空间故安全）；id=hash%prime+前缀 offset，上限=每层 24 素数之和。cut8 缩表 6M 但 vocab=4M → id 上限 96M → 越界读 base+idx×256，fault VA 散布表上方 ~20G（与 dmesg 吻合；device 路不崩=越界落相邻 VRAM 读垃圾）。**修复**：engram_vocab_size=250000 + num_embeddings=[Σprimes]=[6003388, 6013872]（容器脚本精确计算），修后 fault VA 全部落回表内。
- **Bug C（host 路真凶）**：本仓 `hcu_adapter/platform_seam.py` 把 sglang `is_hip()` 强制 False（HCU 伪装 CUDA 的平台决策）→ engram.py `_registered_device_ptr` 走 `return host_ptr` 短路——**hipHostGetDevicePointer 从未被调用**，内核拿裸 host VA，GPU 页表无此段 → VMFault。铁证：canary S 组（自己 ctypes 直连 libamdhip64）dev 全部重定位且通过；真表 dev==host 且崩；seam 环境实测 `is_hip()=False from platform_seam.py`。**修复**：指针翻译以 **libamdhip64 是否可加载**为准（find_loaded_library 失败→CDLL("libamdhip64.so") dlopen 兜底；真 CUDA 无此库 OSError→恒等不变），且翻译结果==host 即 raise 拒绝启动。
- **Bug D（潜在，一并修）**：shared 布局注册前零预触（未跟踪风险）+ per_rank 注册后 finish_load MADV_COLLAPSE 重排 PTE（cut8h 首请求崩的另一解释）。**修复**：注册前整段写脏 0xAB（memfd/anon 脏页无写回目标永不回收，免疫加载期页缓存churn 回收）；注册后的表永不 collapse。
- **修复后实证**：boot→load→dummy forward→prefill→DSPARK decode 全通 0 fault；`dev≠host` 重定位日志在案。**cut8 TPOT 对照：host=21.2-21.9s/256tok vs device=21.4-21.8s/256tok——持平**（TP1 无图小剂量代理，绝对值无意义；生产剂量性能与消拍待 8 卡）。
- **补丁位置**：容器 /sglang-v0522/python/sglang/srt/layers/engram.py（4 处：预触写脏/指针翻译 seam 旁路+校验/禁 collapse/地址日志）；seam 本身不动（is_hip=False 是全平台决策，翻转会动 FP8 常量与内核选路）。
- **生产化缺口（未做）**：per_rank 23.6G/rank 单次注册 >12G 必 err → 需**分块注册**（pincap 已证 12G×4=48G/进程过），_HostTable 尚未实现分块+逐块 device 指针；shared 布局全表 94G×2 不适用于生产。8 卡 matched-dose 消拍测试 = 下一个空窗的首项。

## 2026-10-24 nmz18 单测三连（P1.1 转正 + P1.2/1.3 口径翻案，卡 7）
- **P1.1 topk_v2 ragged 臂转正**：96 例 parity 扫描（rem 对齐/任意 × topk{8,64,128} × L{256,4096} × prod/general offsets × 4 reps）**集合语义 0 错**；"非确定性"实为 atomicAppend 发射的到达序（topk_impl.cuh:612/617——与已转正 paged 臂同一套，flashmla_sparse 按集合消费）；微基准 **2.2-2.9×** 快过 legacy 臂（B512/L8k/k2048: 58 vs 156µs；legacy 在 topk=512 直接断言拒绝）；门 `HCU_TOPK_V2_RAGGED` 默认翻 1（dsa_topk_backend.py，注释留凭据；容器已改，回退设 0）。
- **P1.2 rmsnorm 翻案（达标）**：`fused_add_rmsnorm` kernel-only（CUDA graph 重放）**5.88µs** @T=64/H=5120，parity 1-ulp（out maxdiff 0.0625=1bf16ulp，residual 精确 0）；旧 trace decode 段 p50=5.3µs 与 das 5.6µs 同级。**"450µs/80×"=旧 trace 家族和 ÷ das 单核地板的口径混算**；真差距是 eager 段每次调用 ~51µs 的 python/launch 墙钟（图内无感）。
- **P1.3 router 翻案（达标）**：`_router_triton_kernel`（全融合 sigmoid/sqrtsoftplus-open/log1p + 分组 + topk + renorm）kernel-only **10.0µs** @M=64/E=384/K=6（das 28.4µs，我们快 2.8×；验收 ≤85µs 大幅满足）；M=2048→42µs、8192→123µs。三 scoring 变体全同速（libdevice.log1p 无罪）；num_warps 1→4 无改善且大 M 回退（已回滚）；墙钟 ~100µs=wrapper 开销（同 P1.2 结论）。
- **旧 trace 复盘（prof_dsv41f_dec20 TP-0）**：20 步 DRAFT/VERIFY(bs=2)；exp n=860=43 层×20 步（head_gate sigmoid 非路由）；reduce_bf16 438µs p50=1.0a 之前的 emu MoE-sum（已修）；DtoH 仅 3/步（copy_result_to_cpu 常规）零 cudaStreamSynchronize；**>10ms 核间隙 61 处共 3004ms=拍本尊**；1.6 的 4121 核/步基线需新 trace（1.0a/b/c 后）。
- **P1 真残余主题重定位**：kernel 地板全线 ≤das 同级；剩余膨胀=①eager 段 per-call ~50-90µs host 开销（1.5/1.6 主攻：融合+图覆盖）②核数（1.6，需新 trace 基线）③集中 A/B（1.7）。

## 2026-10-25 nmz26 单测役：P1.4 mqa_logits 翻案收口 + P1.5 同步点普查 + P2.1 fp4-MoE 武器普查（卡 3）
- **P1.4 收口（翻案：无需任何修改）**：生产 HCU indexer = **int8_scaled cache**（`mode=int8_scaled`，page 平铺 split 布局 8192 payload+256 scale/8448B 页，`[64,1,132]` 形状仅记帐）+ 调度优先 **sglang 自带 pers-JIT int8 kernel**（`paged_mqa_pers_jit.persistent_int8_paged_mqa_logits`），lightop int8 兜底、torch 链最底。graph-only 计时 @ctx4k：**decode B16 10.1µs / B64 10.4µs（34MB cache @3.3TB/s≈HBM 极限）/ verify B16×R5 9.7µs，全部 maxabs=0 top64=1.000 精确 parity**；lightop int8 14.1µs 备胎；torch 兜底 413µs（41×差距，量化了 lightop/pers-JIT 在位价值）。das 地板族 5-6µs → 达标（大 B 贴带宽）。**旧 trace 的 dequant+rocBLAS bmm 账=过时树**（dec20 trace 无 lightop/pers-JIT 核名自洽）。
- **P1.4 副产物（登记 wheel 主/留档）**：①deepgemm wheel `paged_mqa_logits` 本 gfx metadata 全拒（fp8 R=1/R=5、int8 R=1 皆 "neither backend supports"；int8 prefill 另要 "packed INT8 KV" 异布局）——非阻塞（pers-JIT 更快）但还 wheel 主；②lightop fp8 读核把 cache 字节按 e4m3**fn**（bias 8）解而生产写 **fnuz**（bias 7）→ 系数**恰 2.0×**（lstsq 拟合 1.9985）——topk 选择不变（单调正缩放）、w/2 可精确补偿；该 fp8 臂 HCU 休眠中，激活前须修。
- **bench 方法学三坑（本轮付学费）**：fp8 dtype **view(bf16) 半维** vs to() 转换；page 平铺 split 布局 vs per-token 交织（sglang 两个 torch 参考实现+aiter 包装全按 split 切片——buffer 建成 flat[P,8448] 自证）；多 R 参考构造 torch.cat over R = r-major 而 kernel 输出 b-major（cosine 0.9997 定位）。
- **P1.5 同步点普查（图外消泡，代码层）**：①dsv4 backend decode/verify 路径**干净**（全部命中=prefill 专属/图捕获一次性/CPU 张量条件项；`expand_prefill_casually .item()`、`in_capture seq_lens.cpu()` 均非热路）；②**spec 层（生产 DRAFT/VERIFY 实际在跑）**有真候选：`frozen_kv_mtp_worker_v2.py:471 torch.sum(batch.seq_lens).item()`（GPU 求和同步；:326-328 已示范 cpu-mirror 先行写法）、`dflash_worker_v2.py:752 lengths.max().item()`（gather 定长、:1890 注释自认只需上界）、`:1915 prefix_lens.sum().item()` GPU 兜底分支；③输出路 sampler/output-processor 现树零直接同步；④旧 trace 实证仅 3 DtoH/步（aten::copy_ 3.7µs，copy_result_to_cpu 族）+ 零 cudaStreamSynchronize——**"归零"验收与新 trace 绑定（并 1.6 核数基线同一 boot）**。
- **P2.1 fp4-MoE 武器普查（模型实形：E384/topk6/moe_inter 2304/expert_dtype=fp4，现役=SGLANG_DSV4_FP4_DEQUANT=1 转 fp8 再 w8a8）**：deepgemm 有 **`m_grouped_w4a8_gemm_nt_masked_hipc`**（HIP 专用）+`pack_w4a8_moe_hipc_weight`、`m_grouped_wfp4a16_gemm_nt_contiguous`、`m_grouped_i8_gemm_nt_masked`；aiter 有 moe_c marlin **wfp4a8_channelwise/groupwise/wfp4a16/w4a16/w4a8** 全家——正对 trace 第一大核族（bf16→f32 copy 1104ms/1700 次=fp4 dequant churn）。**坑：moe_c 族无预编 .so（浅路径 glob 0 命中）=冷容器 JIT 编译费**（cache_export/import 基建可解）。下轮首选：moe_c marlin wfp4a8_groupwise 消 dequant 直连 + `moe_c_situ_glu`/`fuse_silu_mul_*` 收编 silu_mul_clamp(636ms)+per_token_group_quant(245ms) 两段。
- 遗产（nmz26 容器 /root）：mqa_bench4.py（fp8/bf16 四臂+parity 三指标）、mqa_bench5.py（int8 生产对：lightop vs pers-JIT）。
- **P2.1 首攻实探（wfp4a8_groupwise 点火，差一步）**：checkpoint 专家张量实探=**原生 MXFP4**（I8 打包 e2m1 [2304,2560]=K5120 nibble + **F8_E8M0 [2304,160]**=K/32 组）——与 aiter wfp4a8_groupwise 消费布局**逐位对口**，现役 FP4_DEQUANT→fp8 迂回纯属无直连 kernel。点火战果：①kernel **预编 .so 在 wheel 内**（aiter/jit/module_moe_c_wfp4a8_kernel.so，零 JIT）且本 gfx 可加载、dtype/shape 检查可达——kernel ABI=**plain [E,N,K/2] uint8 + e8m0 [E,N,K/32]（无需 marlin shuffle）**，activation qgroup=per-token-group fp8；②dispatch map 收结构化 MODE（stage/routing/family/bm/profile 位段，csv 实例 11201102+21101201）且首个合法 MODE（11001101 族）过门；③**卡点=kernel 内段错误**：rowmajor/colmajor 两种 a_scale 布局（cm 布局 helper 已在树 :419）、delta/size_m 两释义组合皆崩——需 wheel 主给 MODE 表+delta/size_m 契约或其内部测试用例；备选=按 invoke_fused_moe_kernel_marlin 全 wrapper 移植（自备 config dict）。
- **wheel 缺陷清单（本轮累计，待集中反馈给 wheel 主/自家仓库）**：deepgemm `paged_mqa_logits` 本 gfx metadata 全拒（fp8/int8 各臂）；lightop fp8 读核 fn 偏置 vs 生产写 fnuz=恰 2×；aiter `utility/fp4_utils` python 面缺 f32_to_e8m0/e8m0_to_f32/f32_to_mxfp4（per_1x32_f4_quant 不可用）；aiter moe_c wfp4a8 无配置表（17 json 全非 wfp4）+MODE 契约不明。

## 2026-10-25(晚) nmz18 复活役：wfp4a16 直连 kernel 全契约破译（deepgemm 路通、布局定性）
- **nmz18 环境恢复**：同居方修了驱动（renderD128-135 回归，8×BW1101 全空）；dsv41f-dspark 容器重启正常。新坑：本机 aiter `moe_align_block_size` 绑定=**出参式 ABI**（`(..., sorted_out, experts_ids_out, ntp_out)`），树内 fused_moe_c 老 wrapper 传参错位→段错误假象（非驱动问题）；直接 torch.ops 调用带预分配出参即通。
- **aiter moe_c wfp4a8 定性=需 wheel 主**：同 commit ga76f3d 两台机器 dispatch 表不同（nmz26 过门 11001101 / nmz18 全拒）——**MODE 表是 build 级编译产物**，与台账"wheel 分叉"风险互证；gfx938/fp4_w4a8_groupwise32 配置目录真实存在（MODE 9525/9523）但本 build 不认。本条挂起，转 deepgemm 路。
- **deepgemm `m_grouped_wfp4a16_gemm_nt_contiguous` 全契约破译（真 checkpoint 权重，E384/K5120/N2304）**：
  ①权重布局=plain `[E,N,K/2]` uint8 但**每 4 字节单元内 nibble 需逆置换**：kernel K 序读 `[0,2,4,6,1,3,5,7]`（lo 全前 hi 全后）→ repack=`[lo0|lo2<<4, hi0|hi2<<4, lo1|lo3<<4, hi1|hi3<<4]`（one-hot 全 K 提取 5120 行解码矩阵，perm 拟合 conf=1.000、逐位 maxabs=0）；
  ②scale=checkpoint e8m0 `[E,N,K/32]` **原样直用**（bias 127）；
  ③m_indices 组行数**必须 128 对齐**（32/64 对齐 parity 崩 rel 1.2-1.7，128=精确）；
  ④**parity：maxabs 0.0010 / rel 0.0027（bf16 级，decode+prefill 双验证）**——checkpoint 原生 MXFP4 零转换直连成立（除 nibble 重排，load 期一次性）。
- **布局定性（对本役关键）**：contiguous+128 对齐对 decode（m≈1.5/专家）= pad 31104 行 **10.8ms/层**、prefill B512×6=16.9ms——**decode 不可用**（省的 dequant ~55ms/步全险被 40 层×10.8ms 吃回）；本 kernel=**prefill/EP 形态**（m/专家自然 ≥64 时 pad 浪费趋零）。**decode 的 wfp4 直连仍缺 masked 变体**（deepgemm 只有 w4a8-INT4 masked hipc；aiter moe_c 有 masked 但卡 MODE 契约）。
- **决策点（还 wheel 主，二选一）**：A. deepgemm 加 `m_grouped_wfp4a16_gemm_nt_masked`（w8a8 masked 同款骨架，自家 wheel）；B. aiter moe_c wfp4a8 各 build 的 MODE 表/调优脚本（或给出 per-build 调优表生成 CLI）。任一到位，decode MoE 消 fp4→fp8 dequant（trace 第一大核族 1104ms）即通。
- 遗产（nmz18 容器 /root）：dgm_wfp4_final.py（全契约终探针）、dgm_perm_extract.py（one-hot 置换提取器）、wfp4_ignite4.py、probe.log。

## 2026-10-26 wave6 nmz18：softmax adapter 洞补齐（生产采样解码解锁）+ 新鲜 decode trace 基线（P1.5 验收/1.6 基线）

- **生产阻塞根因：非 greedy 采样解码必崩**。DSPARK verify accept 链 `SoftmaxTemp.execute → softmax_temp_flashinfer → flashinfer.sampling.softmax`，该入口在我们 hcu-adapter 只有 loud stub——import 成功（Missing 句柄不抛 ImportError）→ dispatch 判 flashinfer 可用 → 调用即 `NotImplementedError` 打死全部 8 scheduler。greedy 走 `AcceptGreedy` 臂不炸，warmup 掩盖了洞；上一轮"16 并发负载完成"实为 server 半路崩（prof 空盘、日志无 throughput 的真因）。
- **修法（零上游改动，纯 adapter 补洞）**：`sgl_kernel/sampling.py` 新增 triton port `_softmax_rows_kernel`（online 两遍行 softmax：pass1 online max/sum、pass2 归一写回；逐行温度，BLOCK_V=8192/num_warps=8；temp≤0 当 1.0 防除零；标量温度广播/奇 vocab/-inf 列/bf16/非连续行全兼容），绑进 `hcu_adapter/flashinfer` allowlist + registry（MECH_SEAM）。部署前 md5 校验本地=共享盘 `hcu_v0522_nmz18`（NFS）两文件同源后才推。
- **单测 11 项全绿**（容器 GPU）：maxabs ≤1.6e-6、行和 1±1e-6、CUDA-graph 捕获安全；serving 形状 N112×V151936 **167.9µs**（torch 链 312.7µs 的 0.54×）；生产 trace 实测 78 次（draft+verify 各一/步）×76.9µs。生产验证：temp=0.7 采样解码 48 token 正常返回、0 异常。
- **新鲜 decode trace 基线**（TP0，39 步×2 图=78 graph launch，bs2/rank，16 并发×600tok @temp0.7；`prof_dsv41f_dec20/dsv41f_tp0_wave6.trace.json.gz`）：
  - 步长 ~950ms（profiler 开销内口径），GPU busy 81%（30.0s/37.05s），>5ms 空泡 121 个共 6.9s——其中 ~60.4ms×39 = **步间 host 泡 ~60ms/步（6%）**，116ms×2 为 warmup 段。
  - **P1.5 运行期验收：DtoH 2.1 次/步、~30µs/步**（166 次共 1.2ms）——残留常数级、无阻塞型图外同步（hipEventSynchronize 40 次=标准步边界等待；普查出的 3 候选均在非生产算法）。**P1.5 判定通过**。HtoD 5.5/步、DtoD 73.6/步 183ms（量级无害）。
  - **1.6 基线：4499 核/步**（175468 事件/39 步；2249.6/图）——与旧账 4121 同量级（口径差异 1.7 统一）。
  - **核族排名（1.6 靶子）**：`vectorized_elementwise` 660/步 4495ms + `elementwise_manual_unroll` 342/步 3930ms = **torch eager elementwise 两族合计 28% GPU 时间=最大融合池**；`apply_rotary_emb_triton` 46.5/步×**1116µs/call（bs2 下病态，疑对全 KV/大 padding 做）**=51.6ms/步；`_silu_clamp_ptgq` 22×1930µs=42.5ms/步；`_hc_mix_stats_partial` 23×1652µs=38ms/步；nccl 23×1123µs=25.8ms/步；`_moe_sum_scale` 22×1138µs=25.1ms/步；aiter `allgather_vec` 21.5×630µs=13.5ms/步。
  - **fp4→fp8 dequant 家族从每步热点消失**："streaming dequant fp8 wo_a"=**载入期一次性**（log 载入段），decode 步内只剩 `_dequantize_k_cache_paged` 41.5/步×122µs（DSA indexer，P1.4 已核最优）——旧账"1104ms dequant 族"确认为载入窗口混入，**decode 步内无此池**。
  - 采样侧首次可测（干净基线，无 profiler，16×600tok @temp0.7）：**246s 完成=系统 ~39 tok/s @bs16、单 rank 稳态 5.1-6.9 tok/s**，accept len 2.23，折算 **~915ms/步**（与 trace 950ms 吻合=步长真实非 profiler 开销；profiler 窗内 0.5-0.7 tok/s 为死锁拖累均值作废）。此为 1.7 四门 A/B 的 A 臂锚点。
  - **环境副作用登记：profiling stop 后 scheduler 死锁**（trace 导出成功后 /health 无响应、进程 0% CPU，重启恢复）——prof 一次即重启规避，根因未查。
- 遗产：本地 `tmp_fetch/fi_softmax_test.py`（adapter 单测）、`tmp_fetch/wave6_trace.py`（trace 普查器）、容器 /root/fi_softmax_test.py、/root/prof_wave.sh、/root/prof_new/（8 rank trace）。

## 2026-10-26(晚) wave6b 拍定案：不是 runtime、不是 kernel 分支——宿主监控栈 DCU 采集冻卡（117ms 量化、8 卡同步、对照实锤）

- **起因**：用户质疑"拍"归因（hipGraphLaunch 已很短、das 无此问题，怀疑 kernel 特殊分支）。逐层排查结果全部翻案：
  ①**怪物事件**：>20ms 的 kernel 记录 324 个、p50=116ms 高度量化，横跨所有核族（rope 49/manual_unroll 45/vectorized 42/hc_mix_stats 35/silu_clamp 30...）= 在飞 kernel 背锅，非 kernel 本身慢——**剔除后每步真实 kernel 时间仅 ~161ms**（旧普查的"rope 平均 1116µs"是怪物拉高的假账，rope p50=6.1µs 完全正常）；
  ②**真实结构**：每步 ~4 次 ~117ms 停摆（gap 直方图 167 个 ~117ms + 126 个 ~20ms + 118 个 ~50ms），**83% GPU 空闲**；干净负载 915ms/步与 161ms kernel + ~750ms 停摆对账吻合；
  ③**停摆位置在图执行中途**（315 个 >50ms gap 仅 1 个贴提交边界），非 host launch 泡；
  ④**8 rank 同刻**（4 rank 抽样 pairwise 重叠 99-100%）——排除单卡电源/时钟独立抖动；
  ⑤排除项逐一验尸：容器/宿主 cgroup 全链无配额（quota=-1、nr_throttled=0）、CPU steal=0、launch 调用跨进程可并行（461 对重叠，无全局锁）、宿主 128 核 2168 线程无超卖；
  ⑥**对照实验实锤**：本地旧 trace（10-23 上午 `fake_64k_c48`，TP3）**gap>50ms=0 全干净**；新 trace（10-23 22:29）315 个——两份之间宿主唯一 GPU 相关变化=**`/opt/monitor/v5_exporter --dcu.enabled=true --hot.enabled=true --collect.interval=30`（曙光监控栈，19:56 部署，29% CPU；同栈 node/process-exporter 不碰 GPU）**。DCU 采集轮询 quiesce 设备、周期/占空比（~230ms 周期 ~117ms 冻结 ≈50% 占空）与停摆谱完全吻合。**das 当时无此问题=彼时该 exporter 未部署**。
- **量化影响**：停摆 ~750ms/步占 82%——消除后步长估 ~250-300ms（~161ms kernel + host/图开销），**吞吐预期 ~3×（39 → ~110+ tok/s）**=全战役迄今最大单杠杆，优先级高于一切 kernel 优化。
- **处置**：监控栈是邻居/运维设施（sugon basic-auth），按铁律不动它——**上报用户走管理渠道**：v5_exporter 关 `--dcu.enabled` 或拉长采集周期/避让压测窗口；生效后重跑 16×600 干净负载复核 ~3×。
- **wave6 普查修正**：核族排名按"剔除怪物后 kept 时间"重排（每步）：silu_clamp_ptgq 27.2ms（17%，p50 598µs 真·均匀慢=1.6 靶）、vectorized_elementwise 22.0ms、manual_unroll 16.0ms、moe_sum_scale 14.5ms、nccl 14.1ms（含对端等待）、hc_mix_stats 13.1ms、rope 9.4ms（p50 6µs，纯量多）。hipGraphLaunch p50 11-20ms/次（2249 节点大图宿主提交成本，独立小池）。
- 遗产：本地 `tmp_fetch/wave6_trace2..6.py`（分布/gap 解剖/跨 rank/launch 串行检验/对照）、`prof_dsv41f_dec20/dsv41f_tp{0,1,4,7}_wave6.trace.json.gz` + 旧对照 trace。

## 2026-10-24(nmz18 深夜) wave7: 停摆归因复核——纯 torch 探针排除"轻负载冻卡"+ das 对比受阻全记录（环境连环坑入账）

- **纯 torch 图回放探针（8 进程×8 卡，零 sglang/零模型，exporter 在役）**：graph 相位 10.9 万次迭代/卡（100 核/图）+ eager 相位 3.0 万次/卡，**全部零 >20ms 停摆**（均值 411-419µs/1510µs）——**exporter 不冻轻负载 GPU**。wave6b 的"监控冻卡"结论收窄为：需要"重负载（大显存占用进程）条件"才成立，或另有真凶；停摆问题保持 OPEN。
- **das 对比四连阻（每个都是硬事实入账）**：
  ① stock 0.5.18（dist-packages）无 deepseek_v41 注册（仅 v4/v4_dspark），dsv41f 只能走 v0522 树；
  ② v0522 树单独起（不挂 adapter shim）死于 `aiter.ops.triton.gemm` 缺失——该缺口由 hcu_v0522 的 aiter shim 补，"纯树 das"在本容器不存在；
  ③ Qwen3.8-Flash-Next-Channel-INT8（qwen4_exp，stock 原生支持）：我们树 w8a8-**INT8** MoE=`NotImplementedError: only for NPU`；
  ④ DSV4.1-Channel-**FP8**（同模型 das 参考量化）：461→~1TB 参数（fp8 ~125G/卡），tp8 载完权重无 KV 空间（OOM 11.45G 分配）；Channel-**INT4**-w4a8（461G）：compressed-tensors 解析 `KeyError: 'Linear'`。换模型路线今晚全灭。
- **环境连环坑（后续 boot 配方都要带）**：
  ①**NCCL 8 卡 init 全盒坏**：`Add XGMI error: could not find GPU 5000`——镜像烘焙 `NCCL_TOPO_FILE=built-in-508-topo` 在驱动重载后失配；2 卡正常（走 shm 兜底）颇具迷惑性；**解法=`NCCL_TOPO_FILE=`（空=自探测，实测 8 卡 allreduce 通）**，且 launch_rt.sh 原作者本来就 `unset NCCL_TOPO_FILE`（老配方早留了后手）；
  ②**custom allreduce 2GB exportable IPC buffer 分配失败（GTT 耗尽）**：崩溃风暴后残留 exportable 分配未释放（设备级、跨容器）；`--disable-custom-all-reduce` 可绕（牺牲 AR 走 nccl）；
  ③冷容器 tvm-ffi 自锁复现一次：`TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 免疫有效（CI 波结论再证）；
  ④**邻居 vllm 三容器突发抢卡**（pzm/laibao/duyl，~107G/卡 短跑型）：00:52-01:5x 每轮 boot 都在赌窗口，OOM 与"spawn 静默挂"（master+spawner 活、8 scheduler 无声消失）皆此因。**生产 server 已停**，待静默窗口用 `/root/launch_rt07.sh`（mem-frac 0.70 版）重起。
- **下一步判决实验（静默窗口优先做，零模型依赖）**：纯 torch 探针加"重负载条件"——每卡先 allocate ~100G 再跑图回放循环；若 117ms 停摆复现=exporter×重负载交互实锤；若仍零=真凶在 sglang 栈内（nccl 逐层屏障 × 某 rank 周期性宿主延迟），再走 das 同形臂对照。
- 遗产：`tmp_fetch/stall_probe.py`（轻负载探针，已验）、`tmp_fetch/nccl_mini.py`（8 卡 allreduce 判具）、`tmp_fetch/launch_qwen_adapter_1026.sh`（das 对比容器配方：tvm-ffi 盾+topo 清空+AR 禁用全套）、nmz18 容器 `das-qwen`（idle sleep infinity，含仓库挂载 /models_qua，停摆实验复用；不需要时可 docker rm）。

## 2026-10-24(凌晨续) wave7b: 三连探针排除法 + 分配风暴主嫌疑 + 共存不可能定量

- **重负载探针**（每卡驻留 128G 到剩 11G + 图回放）：GPU1-7 各 ~9.6 万次迭代**零停摆**；唯独 GPU0（有邻居散占者、仅剩 5G）96 次停摆、慢 12×、**~438ms 量化**（与 117ms 不同现象=显存压力抖动）。
- **IPC 映射探针**（每卡双进程共享 CUDA 张量，custom-AR 同款机制）：GPU1-7 双 rank 零停摆；GPU0 双进程各 300+ 停摆。**IPC 映射不是触发条件；"卡上有外部进程活动"才是**。
- **旧 trace 补刀两刀**：①停摆窗口内 CPU 覆盖率 1.00 但无任何短 op=宿主全程阻塞在长等待调用（eventSynchronize 类）；②324 个怪物事件**零互相重叠**=真实串行占卡窗口，非时间戳扭曲。
- **主嫌疑易主：邻居 vllm 的启动/显存分配风暴 × KFD/MMU 锁竞争冻在飞 kernel**（22:29 抓 trace 时邻居 tp8 正在初始化：8 卡同冻 ✓、117ms 量化=分配锁周期 ✓、das 彼时无此活动 ✓、三次纯 torch 探针时刻无风暴 ✓、exporter 降为从犯或清白——19:56 部署时间与邻居容器上线同窗，相关性需静默窗口复测拆解）。
- **共存不可能的定量**：栈内权重+dequant 后 KV 最小需 0.748×144≈108G/卡；邻居稳态占 45G/卡 → 153G>144G。本轮 0.55 启动死于 `ValueError: minimum viable = 0.7478`（终于是个干净报错）。**boot 只能等邻居退场**。
- **在役自动化**：`/root/boot_watch.sh` 看门狗（8 卡 ≥120G 空闲连续两次→自动点火 launch_rt07.sh），已武装；下一会话只需查 `/root/logs/srv_rt_1025.log` fired up + 跑 `/root/prof_wave.sh` 重抓 trace 判决（预期：零 117ms 停摆+步长 ~250ms=风暴理论闭合）。
- 遗产：`tmp_fetch/heavy_probe.py`、`tmp_fetch/ipc_probe.py`、`tmp_fetch/wave6_trace7.py`（停摆窗 CPU 状态）、`tmp_fetch/wave6_trace8.py`（怪物重叠检验）、`tmp_fetch/boot_watch.sh`；容器 /root/heavy_probe.out、ipc_probe.out、hstall_*.json、ipc_*.json。

## 2026-10-24(晨) wave8: 1.6 首靶 silu_clamp_ptgq 洗清 + wave6 基线定性为"共卡污染"

- **silu_clamp_ptgq（过滤后第一大真慢族 27.2ms/步）洗清**：standalone 复现生产形状（真实 M=576=96 行×topk6，自 AB1023 ENGAGED 日志取证；非 trace 首事件的 960）——CUDA-graph 36.9µs@M960 / ~30µs@M576，**生产 trace 的 598µs=环境膨胀 ~20×，kernel 与 P1.0b 挂点本身健康**。参数扫附带收获：GPB=2/ROWS=2=26.2µs（1.4×白捡，量级 0.5ms/步暂不做）。
- **DVFS 排除**：稀疏发射（duty 100%→0.1%）kernel 仅 36.9→42.7µs（+16%），单进程场景时钟不惩罚突发。
- **SM 并发排除但暴露盲区**：trace 内全家族零跨流并发（纯串行流）——但 **profiler 只记本进程 kernel，邻居进程的 kernel 对 trace 不可见**。
- **wave6 基线定性翻案：共卡污染**。footprint 选择性膨胀模式（rmsnorm/softmax p50 与单测吻合不膨胀；silu/router/mix_stats/moe_sum/nccl 多 MB 族膨胀 8-20×）+ 盲区 + 时间窗（22:29=邻居 tp8 初始化期）→ **22:29 trace 是邻居共卡负载下的污染基线：其 kernel 时长、81% busy、161ms/步"真实 kernel 时间"全部是被邻居 HBM/SM 争抢拖慢的上界值**；4121/4499 核数/步、DtoH 2.1/步、拓扑结论不受影响。干净基线必须独占窗口重抓——看门狗已值守（boot_watch.sh ≥120G×2 自动点火）。
- **全战役教训一条**：共享盒上抓 prof 前必须验"独占"（fuser /dev/dri + mem_get_info 全空），否则 trace 是别人负载的混合账。
- 遗产：`tmp_fetch/silu_ptgq_bench.py`、`tmp_fetch/dvfs_probe.py`、`tmp_fetch/wave6_trace9.py`（并发盲区检验）、`tmp_fetch/wave6_trace10.py`（grid 取样）；容器 /root/silu_ptgq_bench.py、dvfs_probe.py。

## wave9 (10-24 凌晨): 1.6 主靶归因闭环=RAGGED MQA emu 雪崩，triton 兄弟核歼灭 ~2300 kernel/步

- **问题**：wave6 解码 trace 的 elementwise 大头（~2197/步，占全部 kernel 近半）自 wave8 起 unresolved。本轮以"functor 解析+邻域锚定+原始序列解剖"三段式收网：44/步家族（exp/log/abs/softmax_warp 1024 宽/clamp_scalar/Compare×2/BitwiseAnd/Cat×2/bmm×2/LSE 手工链）精确=逐层固定模式，40 主层+4 草稿层=44 ✓（config: index_n_heads=32, index_head_dim=128, index_topk=512, hc_mult=4, sinkhorn=20）。
- **归因排除法**（全部实测/取证，非猜测）：① router 已融合（`_router_triton_kernel` 44/步在 trace）；② mhc 全链已融合（我们 needle 的 `_hc_mix_stats_partial` 46/步+`_hc_mix_reduce_sinkhorn_vec` 88/步在 trace，sinkhorn 在核内）；③ compressor HIP 已融合（hip_compress_* triton）；④ lightop 分臂在 v0522 树**不存在**（本地旧树有 `_is_hcu` 臂是幻觉——本地 checkout 落后！）；⑤ DTK deepgemm wheel 原生无 paged-MQA 三件套（探针 dir() 看到的符号是 shim 注入的）；⑥ paged 口（dsa_paged_mqa→mqa_triton）resolve=triton 且单核良好——但 trace 无 `_paged_mqa_logits_kernel`。
- **真凶（指纹铁证）**：C4 nonpaged 计划（v0522 `dsv4/indexer.py:732` `_nonpaged_mqa_logits → deep_gemm.fp8_mqa_logits`，在 cuda 平台缝隙下对 decode 放行）→ 我们 shim 的 **ragged torch emu**（`mqa_logits_emu._ragged_fp8_mqa_logits`）。nmz18 实测指纹：**56 aten kernels/调用**（38 manual_unroll+31 vec+Cijk_BBH bmm×6+reduce×6+arange+full(-inf)），与 trace 的 Cijk_BBH 47.2/步家族、manual_unroll/vectorized 分布、arange/Compare/And/where 逐项吻合；44 层/步 × ~52 aten ≈ 2300/步=elementwise 普查的总账。emu 装载无 arch 门（gfx936 理由），gfx938 生产一并中招。
- **修复（R-20）**：`mqa_triton.py` 新增 `_ragged_mqa_logits_kernel`（paged 兄弟核同框架：位算术 e4m3fn 解码、fp32 累加、逐行 [ks,ke) 窗、窗外 -inf、窗外 tile 早退只写填充）；`deep_gemm_dtk.py` ragged shim 按 `HCU_MQA_LOGITS_IMPL`（默认 triton，R-13 #4 同开关）路由，emu 留金标准+回退。**gfx936 语义注意**：emu 的存在理由（无 fp8 矩阵单元）只覆盖 gfx936；本核是纯标量/向量 triton，两架构都能跑。
- **门测试（nmz18 gfx938 实测 PASS 全绿）**：`tests/kernels/test_mqa_ragged_gfx938.py`——fp64 真值仲裁：triton median rel **8.87e-08** vs emu 1.66e-03（紧 4 个数量级，emu 的 bf16 bmm 舍入才是误差源）；-inf 语义逐位；16/16 argmax+top8 对真值；shim triton 路由/=loop 回退；**kernel 数 4(1 核) vs 56**。
- **预期收益（待 P0′ 干净窗口 serving 验证）**：解码图 kernel 数 4499→~2200/步（-49%），图节点/capture 内存/回放同步显著缩，elementwise 家族几乎清零；步长应明显下行（与 0.2 图膨胀治理同向）。
- **教训入账**：① trace 里无融合核名+有 eager 链 ≠ 没有融合实现——先查"我们的 shim 把 wheel 符号换成了什么"（dir() 探针会被 shim 自身污染，用 `__module__` 判真身）；② 本地 checkout 可能落后远端真树（本地 sglang 树无 `_is_hcu` 臂/v0522 已有），**分派分析必须对真树 grep**；③ "emu 装载理由"要逐 arch 复核（gfx936 动机不该自动覆盖 gfx938）。
- 遗产：nmz18 部署树已同步（mqa_triton.py/deep_gemm_dtk.py/test 三件）；`tmp_fetch/wave6_trace11..17.py`（归因工具链：functor 解析/邻域锚定/原始序列/全家族普查/python 栈普查）；`tmp_fetch/probe_mqa2..4.py`+`probe_aiter_ragged.py`+`probe_lightop_mqa.py`（指纹探针族）。

## wave10 (10-24 晨): moe_align 小形状单 CTA 融合核——每层 6→1 核（264→44 核/步）+ 窗口链点火

- **问题**：moe_align 四件套（`_moe_align_hist/_scan/_fill/_block_expert` 各 44/步=176 核/步）+ 两个 aten setup（`sorted_token_ids.fill_`+`torch.zeros`，wave6 FillFunctor 88.1/步锚在 hist→hist 之间）= **每层 6 核 × 44 层 = 264 核/步**。四件套在我们自己的 `sgl_kernel/moe.py`（非 sglang 树/DTK wheel）；stock v0522 是 JIT CUDA（HIP 上不能建）。
- **vendor 换核路线否决**：aiter/lightop 的 `moe_align_block_size` 实现的是标准语义（drop/朴素对齐），与我们的 **EP 哨兵桶语义**（-1→专属 LAST 桶→块 expert_id=-1→filter_expert 置零）不兼容——2.1 的"align 出参 ABI 坑"即此类。自己融合更干净。
- **修复（R-21）**：`_moe_align_fused_small_kernel` 单 CTA 一发全做：寄存器直方图（one-hot 分块累加）→块对齐排他前缀（tl.cumsum，与 scan 核同数学）→整缓冲 pad 预填→FIFO 确定性散写（段内序与四件套的原子到达序不同，runner 按行独立无感）→块→专家映射（哨兵桶=-1/尾块=nb）。桶映射/OOB 防护/尾值/cumsum(E+2 extra) 全 bit 级一致。`HCU_MOE_ALIGN_FUSED`（默认 2048=lane 阈值，=0 关闭回归对照）已注册 hcu_env。
- **门测试（nmz26 GPU2 gfx938 全绿 PASS）**：`tests/kernels/test_moe_align_fused_gfx938.py` 15 组 parity（prod96/576、哨兵 30%/全哨兵、numel0/1、单专家偏斜、2048 边界、2088 四件套臂、bs16/E384/E+2、bs128/int64、E8、OOB 道）+ CUDA 图捕获回放追踪 in-place 变更 + **核指纹 fused=1 vs quartet=6**。
- **窗口链实战**：邻居 06:1x 撤退 8 卡全空（torch mem_get_info 实测 140G×8）——但 boot_watch2 恒报 99G 假窗未自动点火。根因：watch2 探针用 `/usr/bin/python3`（torch 2.11 无 HIP），mem_get_info 全异常→min 恒为初值 99，**真窗被假探针掩埋**。手动点火 launch_rt07（06:15:51），window_chain poller 接力（fired up→150s 沉降→prof_wave2 独占门→census→自动重启）。census 已加 W10/FillFunctor 计数行。
- **预期**：decode 图 4499→~1980 核/步（R-20 -2300 + R-21 -220，累计 -56%）；44 层×5 个多余图节点削除。P0′ 干净重 trace 即出真账。
- **教训入账**：① watchdog 探针必须与被保护进程同一 python/环境，且"探测失败"与"资源不足"必须可区分（默认值静默兜底=把真窗探成假窗）；② 小形状下"包装层 setup 核"（fill_/zeros）可与计算核同刀融合——单 CTA 寄存器直方图在 ≤2K lanes 完胜多核 hist+scan+fill 链；③ 语义特殊（哨兵桶）的内核 vendor 替换前先对拍语义，ABI 坑多为语义坑。

## P0′ 判决 (10-24 06:24-06:30 nmz18 独占窗口全自动链): R-21 验收✓ + 三大翻案

- **窗口链实战闭环**：邻居撤退（torch 口径 8×140G）→ boot_watch2 假 99G 掩窗（探针 /usr/bin/python3 无 HIP torch，mem_get_info 全异常恒报初值 99）→ 手动点火；**首战 mem-frac 0.70 死于 `minimum viable=0.7478`**（wave7b 已知账但配方未改）→ 0.80 重点火 3 分钟 boot → poller 06:21:57 捕 fired up → settle 150s → prof_wave2 独占门过（min_free=24G）→ 40 步 trace + census 自动落账 → 自动重启。
- **R-21 serving 验收 ✓**：`_moe_align_fused_small_kernel` 43/步（21.5/半步段×2 图）、四件套残 0、FillFunctor 134→45.8/步（-88 恰为 moe setup 双 fill）。总核 4499→**4160/步**（-339≈R-21 全额）。
- **翻案一（R-20 错账）**：`_ragged_mqa_logits_kernel`=0——生产 decode 走 **paged 计划**（paged_mqa_metadata_tiny 76 次=R-02 vec 路径），Cijk BBH 21.5+BSS 23/半步段=vec 双 bmm 正当实现，**ragged emu 不在生产路径**；wave9 "4499→~2200(-49%)" 预期作废。R-20 保留为 nonpaged 分支修复（正确但非热点）。教训：归因必须先确认"生产实际走的分支"（trace 分支证据>代码路径推理）。
- **翻案二（邻居理论破产）**：独占窗口（8×115G 全为我们 server、无邻居增量）下 **117ms 量化停摆照样全现**（84 次/20 段≈4/步，成对 115.9×2+116.5×2 集中在 MoE/silu 段）+ kernel 时长 95× 膨胀（silu_ptgq 单核 2.85ms vs 单测 30µs）。wave7b "邻居分配风暴"归因错（重负载 probe 满载唤醒态无停摆≠生产低载无停摆的证明力被高估）。
- **翻案三（膨胀=虚胖 dur）**：稳态尾窗 busy 81%/gap 19%、gap 全 <1ms、max 并发 2、时钟 sclk 1350 正常、降频排除——**dur 虚胖=流内串行等待摊入**（NCCL/跨 rank 集合等待核/图依赖），wave6/7 的"kernel 时间账"（161ms~3750ms/步）全部不可作墙钟依据；**唯一可信墙钟=步长**：当前 **1054ms/步**（527ms×verify/draft 两段）vs 1.7 验收线 550ms。
- **下一波主靶重排**：① **116ms 成对停摆**（~232ms/步,MoE 段,独占仍在——需 8 rank trace 时间戳对齐找慢 rank/集合点；嫌疑=NCCL proxy 周期/XMI 门/EP allgather 等待）；② **elementwise 大军真凶**（vectorized 600+manual_unroll 333/半步段≈1900/步,非 MQA——候选=topk 后处理 sbtopk/mbtopk 邻域、quant scale 链、hc torch 段、dp gather/scatter）；③ per_token_group_quant_8bit_v2 88.5/段（177/步,88ms 段时长）。
- 遗产：/root/w10_probe.py、w10_seg.py、w10_cut.py、w10_bi.py（分账工具链）；prof trace 1792794307.*.trace.json.gz×8。

## wave10 续: elementwise 大军真凶图谱（1.6 下一刀的靶单）

- **量**：稳态尾窗（~5 步）aten 核 12919 → **~2584 aten/步**（占 4160 核/步的 62%）；3184 个 aten 连发段锚定到前置自有命名核。
- **图谱（锚点→跟班数）**：① indexer 段=最大簇：`Cijk_Ailk_Bljk_SB_MT32x32x32`（44/步，每锚 10 aten 连发）+`Cijk_BSS`（18 锚×50 连发=900）+SB_MT32x32x16（228 锚带 912）≈ **topk/采样后处理链**（sbtopk gatherTopK/mbtopk 邻域，clamp_scalar 190/add 170/compare 120/masked_fill 92/bitwise_not 136/步）；② `_dequantize_k_cache_paged` **1:1 伴随** 96/步（掩码/索引核，可并入 dequant 核）；③ router/moe_sum_scale 各 ~2/锚、fused_k_norm_rope ~3/锚小尾巴；④ gpu_kernel_impl(+nocast) 800/步=TensorIterator 通配（cat/index/gather 类）。
- **定性**：不是 wave9 推断的单点 ragged emu（那不在生产路径），是**横跨 indexer/MoE/norm 的 torch 胶水层总量**；收编价值=launch/图节点/等待面（62% 核数），墙钟贡献待 116ms 停摆解决后再评（dur 虚胖不可用）。
- **下一刀排序**：116ms 成对停摆（232ms/步=步长 22%，per-rank 8 trace 对齐法）＞ indexer topk 后处理链融合（最大 aten 簇）＞ dequant 伴随核并入。

## emu 审计 (10-24 晨, 用户指令: 日志 emu 即 bug, 优先修)

**服务日志 4 个 emu 装载点的判决表**（日志 + 06:24 干净 trace + 代码/容器实证三方对齐）：

| # | 日志证据 | 判决 | 行动 |
|---|---------|------|------|
| 1 | `[hcu] MQA-logits emu installed (paged→torch gather-GEMM, ragged→torch GEMM)` | **banner 过时**：resolve=triton 下两入口实际路由 triton 核（容器 probe 实证：paged 入口发 `_paged_mqa_logits_kernel`）；但 wrapper 每调用带 **9 个 aten 胶水**（q fp8→fp32 host 转换×N + w/ctx/bt 转换 + out zeros FillFunctor）= 10 核/调用半 emu | 修：胶水并入核内（q 按位解码如 R-20 ragged、out empty 化），目标 10→1-2；banner 改真身 |
| 2 | `HCU_FA_IMPL=auto -- fa3 attention stays on the torch emulation` | **真 emu 在役**（dsa_prefill_backend=fa3 → prefill attention torch 模拟；=triton 有 decode 路线） | 修：prefill 段换 triton/原生路线（排 decode 优化后） |
| 3 | `fp8_utils.bmm_fp8 bridged to torch-native dequant bmm` | **真 emu 在役**（dequant×2+aten bmm/call；trace 44/步层频 hipBLAS 家族嫌疑之一） | 修：融合 fp8 bmm（核内 dequant，gemm_fp8.py 基建在） |
| 4 | `[HCU kprd] read-emu installed` | **死装载**（12 rank 装载 banner，0 调用行——两次日志审计一致） | 不修；防回归：装载时加计数器日志 |

**MQA logits 生产实态（未闭环，下一波第一动作）**：trace 里我们两个 triton 核（`_paged/_ragged_mqa_logits_kernel`）都不在、flydsl 核不在、wheel 无 `fp8_fp4_paged_mqa_logits`（容器实证 exists:False）、三个选择 env 全默认 False、emu paged 版遇 preshuffle=True 必 raise（没崩）——排除法后每层 1×`Cijk_Ailk_SB_MT32x32x32`+10 aten 连发（45.6 锚×10=456 aten/步）的真实 caller 待定。**闭环手段：HCU_DUMP_MQA=1 定向复跑（3 分钟热 boot）拿调用指纹**。注意 `_quantize_fp4_indexer_kernel` 4.1/步（≈draft 4 层）暗示 draft 段走 fp4 indexer 而 target 40 层走别的。
- 关键代码坐标（真树 v0522）：dsv4/indexer.py:818-841 fn 五叉选择（fp8_fp4/aiter_fp4/tilelang/aiter/torch_sm120/else deep_gemm）、:722 _nonpaged_mqa_logits→deep_gemm.fp8_mqa_logits、:1031 run_paged_indexer→fn(...)；dsa_indexer_kpool.py:1029 aiter_paged_mqa_logits(preshuffle=True)；dsa 包符号被 mqa_logits_emu.install() 换成 torch 版（遇 preshuffle raise，故 kpool 路径必崩 → 未走）。
- 教训：装载 banner ≠ 运行实态（install 绑定 vs 运行时 impl 分派两层），emu 审计必须 trace 核名+容器 __module__ 实证双层验证。

## emu 审计续 (10-24 07:00-07:05 定向复跑): MQA 实态再排除三条 + 层频翻案

- **HCU_DUMP_MQA=1 定向复跑**（06:49 重启 + 60tok 请求）：`[HCU mqa]`/`[HCU mqa-emu]` **全程 0 行** = decode 生成路径**一次都不调** deep_gemm 命名空间两个 shim 入口和 emu 入口（三入口全排除）。
- **fn 真身探针**（needle dsv4/indexer.py:848 forward_c4_indexer fn 选择后，HCU_DUMP_MQA 门控）也 **0 行** = **decode 不执行 forward_c4_indexer**（C4Indexer.forward→attn_backend 链在 decode 被短路或走别路）。探针保留在树（env 门控零开销），下轮加三针（forward 入口/方法入口/fn 后）定位断点。
- **层频翻案（关键认知修正）**：重数 trace——indexer 全家族都在 **2-16/步**区间（topk_split_select 8/步、gatherTopK 4.5、mbtopk 4、_quantize_fp4 4.1、_store_fp4 4.1、metadata_tiny 2、_rope_tail_fp4 16.4、_dequantize_k_cache_paged 83≈2/层）而**非 44/步**：**DSV4.1F 的 DSA indexer MQA 不是每层跑**（低频复用/间隔层结构待查）。→ 每层 44/步的 `Cijk_Ailk_SB`+10 aten 连发（456 aten/步）**不是 indexer MQA**，是每层 MLA/投影/别的结构——1.6 下一刀的归因靶要重定位（嫌疑：bmm_fp8 桥的逐层投影调用 或 MHC/compressor 的 torch 段）。
- **修正后的 emu 优先级**：① bmm_fp8 桥（在役、逐层量级嫌疑升至第一）② MQA wrapper 9 aten 胶水（latent——生产未走，修好待路径回归）③ fa3 emu（prefill）④ kprd（死装载）。116ms 停摆仍墙钟第一。

## 116ms 停摆深挖 (10-24 晚, 零成本本地 trace 分析, 四连判决)

1. **116ms ≠ 计算（记账型等待实锤）**：victim `grid=[160,1,1]` 的 w8a8 小核"跑"116.9ms 算力上不可能；所有 victim 前后 gap=+0.00ms（GPU 时间轴被名义占据）、窗内 0 个并发核开始、hipPointerGetAttribute 洗清（10377 次共 8ms p50 0.47µs）。
2. **8 rank 同层同时刻集体等待**：跨 rank 层位指纹 TP-1..7 完全一致（L5/L12/L18/L23/L27/L31，间隔 5-7 层；TP-0 仅受害者换名）——**每 ~5-6 层一次全局冻结**，间隔撞 `speculative_dspark_block_size=5` 的块边界形态。受害者随机（w8a8/nccl/BSS/cross_device_reduce/elementwise/moe_align_fused 谁在跑谁背锅）。
3. **无 DSPARK 的 eager 下 116ms 家族消失（>100ms=0）**，但露出**新量子 20.3±0.2ms**（9 次、受害者随机、python 栈=常态层循环 `_forward_layers_hc_post`）——116 与 20 不同源：116 需 DSPARK（或图），20 是更底层 tick（候选：宿主 exporter 采样周期/cgroup CPU 限流/驱动 tick）。
4. **结构性账**：hipEventSynchronize 39 次×均 917ms=35.7s（≈每步一次，调度等待主体）；hipGraphLaunch 76×均 10.5ms；hipDeviceSynchronize 1×1081ms；hipMemcpyWithStream 66×922ms。
- 已排除累计：邻居（独占照现）、GPU_MAX_HW_QUEUE=4（无效果且端到端 -23%）、hipPointerGetAttribute 慢 ioctl、python GC/阻塞（eager 栈是常态层循环）、硬件时钟（sclk 1350 正常）、hipLaunch 发射饥饿（gap<1ms 无空洞）。
- **下一刀（切分实验）**：图模式+无 DSPARK 一次 boot+短 prof——116ms 消失则 DSPARK 块边界同步实锤（读 spec verify 段 host 路径，注意 `speculative_dspark_sps_table_path` 生产未配置）；仍在则图回放机制。并行只读普查：宿主 exporter/rocm-smi 定时器周期对表 20ms 量子、cgroup cpu.max。
- 遗产：本轮分析全在本地 prof_1024/ 三套 trace（图基线/eager栈/queue对照），零 GPU 成本。

## W11: 116ms 停摆机制判决 (10-25 凌晨, 静态 trace 深挖 + 单卡微基准)

**背景**：切分实验（图+无DSPARK）与用户消融思路（跳算子看停摆）双线被硬件堵死——nmz18 KFD 泄漏 113G×8 待宿主复位、nmz26 仅 4 空卡装不下 dsv41f（476G/4=119G 权重 + 47G engram TP4 分片 > 144G 卡，实测 TP4 boot OOM 于权重段；另 4 卡被同事容器占住）。改走零成本静态深挖本地 prof_1024 trace + nmz26 空卡单元复现。

**五连判决**：
1. **块边界=图边界假说证伪**：图模式 decode 每步仅 **2 次 hipGraphLaunch**（draft+verify 整图，层循环在图内，T3 实测 launches/step=2）；停摆窗口内 0 次 graph launch（对齐率 1/149）——DSPARK 不按 block 分图，"每 5-6 层停摆"与图边界无关。
2. **双家族分型（跨 rank 绝对时刻对齐，8 trace 同窗 40.99s ±2ms 已验证）**：
   - **A 家族（45-60ms×大量）**：8 rank GPU **同时全空窗**（其余 rank 窗内零事件）；本 rank host 整步卡在 `process_batch_result → Event.synchronize`（实测一例 dur=1046ms≈整步）；copy 流 3 个 DtoH（早已提交）不执行；**同一 graph 重放内部横跨停摆**（track108 段结束→48.9ms 空洞→track212 段续跑同 corr）；host 苏醒后下一个 `hipGraphLaunch` 调用本身耗时 **19.46ms**。= **GPU 对已入队工作不执行的驱动级停摆**。
   - **B 家族（116ms×少数）**：单 rank 本地停摆 116ms，**其余 7 rank 在 allgather 核自旋等待**（实测窗内 TP-1..3 各 ~111ms allgather 忙等）——掉队者模型：一个 rank 的本地停摆把全组钉死。上轮"8 rank 同层同时冻结"结论修正：同层位指纹成立，但 A/B 两种成因不同。
3. **量子结构 ~19.4ms**：簇 20/56-58/116-118 = 1×/3×/6× 周期（gap#70 实测 48.9≈2.5×，hipGraphLaunch 19.46 实证周期出现在提交路径上）。
4. **eager 的 20.3ms×75 是另一回事 = host python 每层开销**：99.5% 被 python 帧覆盖（`_call_impl→forward_hc_pre_from_prev/run_attn_hc/run_ffn_hc` 链，~20ms/层×32=650ms/步）——eager 天然 host-bound，与驱动 tick 无关。上轮"eager 20.3ms 量子"归因修正。
5. **健康卡复现不出**：nmz26 card0 单卡微基准（2000 次/项）——sync-warm p50=24µs、idle-2/50ms 后 sync p50=3µs、graph replay p50=16µs；32 流压力仅出 5ms 尾巴（launch p99=10ms）。**20ms 量子在安静卡上不存在** → 触发条件在 nmz18 服务拓扑（8 进程/xGMI/NCCL 多 comm/邻居共用 KFD）或 nmz18 KFD 病态实例本身。warm 路径 1/2000 孤例 25.7ms 提示机制存在但安静态极低频。

**用户消融思路（跳过耗时算子）的判决：证伪**——两家族停摆窗内 GPU 零 kernel（A 家族）或受害者本非慢算子（B 家族掉队者的 116ms 也是排队不执行），"耗时异常算子"不存在，跳算子消不掉驱动级完成/调度 tick，反而更多空闲→更多唤醒。证据链完整，无需再跑消融臂。

**复活后行动清单（一次 reboot 两用：KFD 泄漏+停摆嫌疑同源）**：
1. nmz18 宿主复位后第一件事：launch_rt07.sh 原配方 40 步 trace 复测——A/B 家族消失 → 根因=KFD 劣化，W11 结案归基础设施；
2. 若仍在（驱动版本级）：env 臂 `GPU_MAX_HW_QUEUES=16/32`（只试过下调 4=更差，从未上调）、`GPU_USE_SYNC_OBJECTS=1/0`、`HIP_USE_GRAPH_QUEUE_POOL`；仍不行再软件缓解：batch_result_processor 的 Event.synchronize 改 query-spin 闪避 + DSPARK draft 边界集合通信减负。
- 遗产脚本：tmp_fetch/w11_*.py（boundary/deepdive/host_in_gap/stream_chain/eager_tick_host/eager_span/eager_tile/cross_rank/cross_rank2/resume_align）、bench_sync_tick.py/.2（nmz26 card0 已验证干净）；nmz26 /root/launch_tp4_dspark.sh+nospec.sh（tp4 配方，实测不可用：模型装不下，留作 4 卡小模型臂模板）。

## W12: 停摆收口战 (10-25 01:00-01:30, 静态深挖+合成复现战役)

**图发射 tick 实锤（W11 量子定位）**：TP-0 hipGraphLaunch n=76 = 37×1-5ms + 37×15-25ms（top 20.0-20.9ms），时序**严格交替**（1.4/19.6/1.4/19.6...）——每步第二次图发射确定性吃满一个 ~19.5ms tick；且快发射 1.4ms 也为健康卡（0.097ms）的 14×。两段发射均在 DRAFT 相位内（DSPARK 每步两图：接续队列者快、打醒空队列者吃 tick）。

**TP-0 单点掉队坐实（上轮"8 rank 同层同时冻结"再修正）**：跨 rank 空闲窗普查——真 GPU 空闲窗**只存在于 TP-0**（148 个，即承载真实请求的 DP rank），其余 7 rank 同期以 allgather 长核**自旋**形态陪等（长核≠空闲）；A 家族（全 rank 同时空窗）唯一 1 例 45.7ms，释放 spread=0.0ms。停摆损失主体=TP-0 单点 × 全组陪等。

**合成复现战役（nmz26 健康卡 card0-3，全部打不出 tick）**：
- 单进程全配方（双流图+copy 流 pinned DtoH+Event.synchronize+idle 环）：sync 全 <0.1ms；step wall 7/800 毛刺（17.6/18.6/43.8ms，接近 1×/2× tick 但零星）
- 4 进程 NCCL×4 卡全配方：sync 全 <0.1ms；各 rank 仅 1 次一次性 >130ms（NCCL comm 初始化）
- 内存压力 125G/144G（=nmz18 服务态密度）：replay p50 0.097ms、sync µs，无 tick
- 早前：32 流队列压力仅 5ms 尾巴；安静卡全绿

**判决**：~19.4ms tick 无法在健康硬件上用任何软件拓扑（队列/内存/NCCL/图/拷贝流/同步环）复现 → 根因收敛至 **nmz18 KFD 实例劣化**（与 113G×8 显存泄漏同根：发射路径慢 14-200×、完成检测/队列唤醒吃 1-6 个 20ms tick、每步损失 ≈21ms 发射 tick + A/B 家族停摆）。软件侧已无待试假说；nmz18 宿主复位为唯一解锁动作，复位即终局实验。

**踩坑两枚（入经验）**：① 容器预置 NCCL_TOPO_FILE 为 8 卡拓扑，4 卡进程组必须置空（原配方有 unset 我漏抄，报 `Add XGMI error: could not find GPU 5000`）；② pkill -f 自匹配第三次（bash -c 命令行含目标名→杀己 shell→后续命令静默未执行；铁律=pkill 单独一条 ssh，launch 另起）。

**预置终局脚本**：nmz18 dsv41f-dspark /root/w13_postreboot.sh（一键：probe_cards 体检→launch_rt07→grab_q4 40 步→trace_census→gap/launch-tick 直方图→VERDICT 行）。复位后 `bash /root/w13_postreboot.sh w13` 一条命令出判决：gaps/tick 归零→KFD 根因结案；仍在→驱动版本病，上 GPU_MAX_HW_QUEUES=16/32、GPU_USE_SYNC_OBJECTS 臂。
- 遗产：tmp_fetch/repro_tick.py（全配方复现器，--nccl/--graph/--pinned/--xdst 四刀二分）、bench_pressure.py（灌压基准）、w12_more.py/w12_which_graph.py；nmz26 /root/ 留 repro_tick.py+bench_pressure.py。

## W13: box16 第三盒复现 + 图发射 tick 机制破解 (10-25 08:30-11:10)

**平台就绪**（box16 card0-7 空 8 卡，BW1200=gfx938 同架构、每卡 144G）：装 ssh 公钥免密（askpass 在后台化执行下不可用的根治）；自有容器 stall-repro（团队 sglang:0.5.18-dtk26041-dpskv4 镜像 + /data_add2:/models ro + hyhal 挂载，抄邻居 sby 配方）；模型用本地盘 DeepSeek-V4.1-Flash-Channel-FP8（基础版踩镜像 triton MoE hidden-size 断言 + deepep 踩 forward_deepgemm_masked deprecated，Channel-FP8+a2a none+triton runner 可跑）。

**bring-up 三坑入账**：① JIT 冷编 cuda_fp8.h 缺失→装 `__nv_*→__hip_*` 记号别名垫片（/opt/dtk/include/cuda_fp8.h，注意 AMD 类型带 _fnuz 后缀；nmz26 靠 43 模块种子 cache 从未现场编过、我们 v0522 树源码已移植 __hip_ 记号——镜像 0.5.18 源码还是 __nv_）；② 16 并发首请求触发全 rank 冷 Triton 编译 >300s 被 watchdog 杀→--watchdog-timeout 1500 + 先单请求预热；③ 杀服务要 `pkill -f "sglang::"`（scheduler 子进程改名，单杀 launch_server 留孤儿占端口）。

**判决链（双盒对照 + 线性律）**：
1. **tick 双盒实锤**：box16 健康卡 ARM1（tp8/dp8/ep8 图模式无spec）38/38 次图发射全部 16.5-22.6ms（无一快发射）——~20ms 图发射 tick 不是 nmz18 病态专属，是**驱动族级属性**（W12 "KFD 劣化收敛"结论半推翻）。
2. **A/B 家族（GPU 空转停摆）box16 无**：decode 段 >15ms gap 仅 4 个且全在 prefill 段（eager moe_align 前 host 时间）——116ms/60ms 停摆家族仍是 nmz18 特有（或 DSPARK 依赖，末臂验证中）。
3. **tick 机制破解（单元级线性律）**：图内 NCCL 集合节点数 × ~100µs/节点 = 发射成本——1 AR=0.28ms、30 AR=2.7ms、90 AR=9.5ms（4 进程 tp4，p50，800 次/臂，线性完美）；对照组 3000 纯 kernel 图=0.34ms（kernel 数无关）；eager allreduce=0.08ms（不在图内无惩罚）。服务图 dp-attention 每层 allgather×32 层+边界 ≈200 集合节点 → ~20ms = tick。**hipGraphLaunch 提交路径对图内每个 NCCL 节点做逐节点翻译/打补丁（~100µs/个）**。
4. **env 旋钮两臂无效**：GPU_MAX_HW_QUEUES=16（16/18 tick p50 17.2ms）、GPU_USE_SYNC_OBJECTS=1（15/18 p50 17.1ms）——机制在节点翻译层不在队列调度层。
5. **单元复现边界**：安静单卡/多流压力/125G 内存压力/4 进程 NCCL 全配方/图内 1 AR——全干净；唯一杠杆=图内集合节点数。

**修复方向（依机制）**：① 减图内集合节点数（dp-attention gather 跨层合并/批量化、ncclDevKernel 换 custom-AR 核——普通 kernel 无此惩罚）；② 厂商报告线性律（附 1/30/90 AR 数据，DTK runtime graph-launch 逐节点成本）；③ DSPARK 边界 rs+rsv 若也是 nccl 节点入图，块边界的集合数也是税。**注意：tick 修复收益 ≈20ms×2/步（nmz18 实测每步一快一慢）；A/B 停摆家族（61% 步时主损）仍待 DSPARK 臂/复位 nmz18 判归属。**

- 遗产：box16 /root/{arm.sh（参数化 env+自动 boot+prof+判决）、verdict2.py、bench_graph_nccl.py（线性律复现器）、bench_sync_tick.py、launch_arm1.sh}；本地 tmp_fetch 同名 + cuda_fp8_shim.h（垫片源）；box16 ssh 免密（~/.ssh/config Host box16）。

## W13 续: DSPARK 臂判决——两半谜题各归其主 (10-25 11:20-11:40)

box16（健康盒）+ DSPARK（同生产配方 block5）：图发射 36 次中 18 次 tick（p50 19.29ms max 24.8ms）——**与 nmz18 的 37/76 交替形态完全一致**（draft/verify 两图中集合节点多者吃 tick，符合逐节点线性律）；而 **A/B 停摆家族全 trace 仅 1 例 40-70ms**（nmz18 为每步多次）。

**终局判决（两半各归其主）**：
- **~20ms 图发射 tick** = HIP runtime 对图内 NCCL 集合节点的逐节点提交成本（~100µs×~200 节点）。驱动族级（nmz18/box16、DSPARK/无）。修复=减图内集合节点（gather 跨层合并、custom-AR 核替换 ncclDevKernel 节点）+ 厂商报告（线性律三锚点 1/30/90 AR）。
- **116/60ms 停摆家族（GPU 不执行已入队活，步时主损 61%）** = **nmz18 KFD 实例病态专属**——健康盒同拓扑同 DSPARK 不复现。宿主复位即应消失（w13_postreboot.sh 一键判决，预期修正：复位后 tick 仍在（族级正常）、A/B 家族应归零）。
- eager 20.3ms = 每层 host python（W11），正交。

box16 清场：sglang 服务进程已杀（卡 0-7 释放），容器 stall-repro 保留（复现器/判决脚本在 /root，供厂商报告复演）。

## 2026-10-25/26(box16 钟) W14 box16 DSV4.1-Flash-INT4 役：停摆判决（无）+ 45x 缺口二分钉死（地板=每层计算/host，非通信非拍）+ channel-int4/draft-MoE 配套
- **停摆判决（本役主问）**：adapter(f74faec+本役推针)+树(32 文件 parity)+DSV4.1-Flash-INT4 dp8ep8+DSPARK——三次干净 run（无 profiler）零 watchdog/零调度死/吞吐稳态无间歇塌陷（每 rank 每区间 2.41 tok/s 恒定）⇒ **box16 无 nmz18 式 116/60ms 停摆家族**（与 W13 续"KFD 专属"互证）；驱动族 tick（p50 10.43ms，W13 已定罪=图内 NCCL 集合节点逐节点提交 ~100µs×~200 节点）在但不闸步。
- **二分账（短上下文 8×200 @temp0.7，全部干净无 profiler）**：DSPARK(block5, 模拟 accept 2.93)=**1.86s/步** → 无 spec dp8+dp-attention=**443ms/步**（graph=True，40 层→10.4ms/层）→ 无 spec tp8 dp1 ep8=**305ms/步**（dp-attention 族贡献 ~140ms）→ **NCCL 8 卡微基准地板全健康**（allreduce 81µs/a2a 112µs/allgather 110µs/barrier 133µs，步通信总量 ≤~16ms）⇒ **病灶=每层 kernel/host 计算**，305ms 与 nmz18 账 L130"消拍后地板 250-300ms"分毫不差=同一共享缺口族（L39 阶梯：vendor 地板+核数砍半→40-70ms→deepgemm LL→17ms），das 镜像同形 37.6ms=45x 全程已知、无 box16 新病。
- **DSPARK 4.3x 放大与 nmz18 同型**（915ms vs ~300 地板=3-3.7x）：两嫌疑均已排除——draft MoE E=16 默认配置实测 0.26-0.71ms/call（非 71.8ms 型灾难；调优后 0.25-0.35，收益 ~5ms/步不值重火）、match-expected 模拟机关=CPU 伯努利+小张量（µs 级）；放大本体=draft/verify 机构（~280ms/草稿 token vs 主前向 443ms/40 层）归 P1.6 核数/融合线。
- **channel-int4 全链（d1270e2）**：compressed-tensors strategy=channel group=-1 → 树 gptq 分支门 block_shape[1]>0 不过 → tl.dot(bf16,u8) 编译炸；adapter 改写 [0,maxK] 退化 group + [E,N,1] scale 直配，8 rank 命中、boot/decode/采样全通。E=48 int4 调优（rayless，容器 ray GCS 超时）：UP 8192 71.8→10.2ms/DOWN→5ms，运行期 "Using MoE kernel config" 实证。**调优器欠账：未套 smem 守卫预算（BN=256/stages≥2 条目运行期被钳 _SAFE）——重调时 tuner 须按 _implied_smem≤64K 过滤。** E=16,N=2304 draft 配置亦写（E=128 experts/层 bf16 w13[16,4608,5120]/w2[16,5120,2304]，safetensors 头实证）。
- **最终数（adapter 栈）**：P1 bs16 in64k out1k @accept2.93：wall 1339s/系统 11.9 tok/s/含 prefill TPOT 1339ms（decode 相位步时 1.86s→单 token ~635ms）；P2 短上下文 8×200 spec 126.4s vs 无 spec 89.2s vs tp8 61.0s；warm 64k prefill 203-266s（attention 主导，预期内）。
- **roctracer 双重毒（box16 dtk26041 实锤，方法学）**：①/start_profile+stop 后 ~90s 调度器段错误（RoctracerActivityApi teardown）；②小核时长膨胀 ~160x（silu_and_mul_clamp 0.054ms standalone vs 8.5ms traced）⇒ 本 DTK 栈一切 trace 绝对值作废，真值=客户端墙钟+Decode-batch 节奏。
- **杂项坑**：box16 /models 挂载**间歇 ENOENT**（重载偶发，python open 报 No such file 而 bash 可读；listdir 拼路径即过）；box16 adapter 部署快照缺 nccl_topo_guard/watchdog_cold（f74faec 早于 9b21831，本役脚本手动 unset NCCL_TOPO_FILE 等价兜住，下次部署应刷 HEAD）；容器内 141 个 sglang:: defunct 僵尸（PID1 收养，零资源，不可 kill 只能容器重启清，不影响他用）。
- **遗产**：box16 /root：bench16ns.sh（无 spec 臂）/bench16tp.sh（tp8 臂）/bench16coll.py+sh（集合微基准）/tuned16d.py（draft MoE 探针+调优）/probe_dflt16.py（默认配置成本）；本地 tmp_fetch 同名镜像。box16 已清场：sglang 活体 0，卡 0-7 全空（140/144G），容器 stall-repro 保留。

## 2026-10-26 W14补 trace 异常归因改正（用户历史口供）+ int4 GEMM decode 地板实证
- **归因改正**：W14 的"roctracer 双重毒/小核时长膨胀 160x"改判为**驱动侧空泡吸收**——该问题历史上 prof 表现为 kernel 间空泡（提前下发/graphlaunch 也在），runtime 一度归驱动、一度归编译器，后驱动侧"解了"：trace 里空泡消失但 kernel 耗时变长=空泡时间被折算进 kernel 窗（用户口供，2026-10-26）。本 trace 实证自洽：gptq 核记录 p50 仅 7µs（真值 ms 级）+后继核窗口吸收（act 5056µs≈UP 真值、reduce 2009µs≈DOWN 真值）+主流 94.8% "覆盖"（泡被折进核窗后覆盖虚高）。**nmz18/nmz26 的拍与本问题疑同源**（上层可绕：das 用整 MoE 单核 vendor kernel 后很少复现）。
- **int4 triton GEMM decode 地板实证（事件计时，tracer 无关）**：UP+DOWN 段图内外一致 ~1.2-5.4ms/层对（随时钟/配置波动 3x）；配置全网格扫描（smem≤64K）无 sub-ms 档，最好 UP 3.9/DOWN 1.9ms（BM16/BN256/BK32，smem 34816 合法）。**专家数缩放判别**：M=1 与 M=6 完全同耗（1182/1186µs）、1→48 专家 730→3785µs、单专家有效带宽仅 ~16GB/s=HBM 的 1/180——token 数无关、专家数驱动=权重流延迟绑定+每调用大额固定税（驱动泡吸收进核窗无法从事件计时中剥离）。8.1ms/层×40≈325ms≈无 spec 步 443ms 主体成立。
- **旧调优器 0.246ms 判幻影**（与三轮独立实测矛盾，作废）；E=48 两 JSON decode 段已换 BM16/BN256/BK32（止血，MoE 段 11.7→~2.4-5.8ms/层随时钟态）。**根治=das 方案移植**：fused_experts_impl_aiter（_get_aiter_moe_config_w4a16→moe_c/ASM/CK 整 MoE 单核，gemm1_limit 融 DSV4 clamp，可选 scale 重排）——das decode 零独立 act 核、1229 核/图发射（我方 2067）。
