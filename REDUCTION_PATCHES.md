# REDUCTION_PATCHES — 减核役批量账（emu→triton/融合，单测先行，serving 集中验证）

单批工作流（用户定向）：逐项 parity（含 fp8 ulp 语义门）+ 秒级微基准 → 批量完成后
**一次**集中 serving A/B（nmz18 dsv41f dp8ep8+DSPARK，四门 + bench_step p50）。
不做逐项 boot 对比实验（用户 2026-10-23 指令：prof 发现问题直接解）。

## 批次账（2026-10-22/23，B 臂 = 全融合）

| # | 改动 | 文件 | 单测判决 | serving 效果 |
|---|---|---|---|---|
| 1 | DSV4.1 sparse rope/norm torch fallback → triton 门（`_fused_dsv41_kernels` needle） | `sgl_kernel/_adapters/dsv4_sparse_triton_gates.py`（7508c7d） | bitwise 10/10 + probe 3/3 | B 臂组成 |
| 2 | silu_and_mul_clamp × per-token-group-quant 融合为一核（`silu_and_mul_clamp_quant`，GPB=2 重铺修复大 M 断崖）+ **针 v3 挂点修正**（routed arm 锚 `act_and_mul_triton` else 臂，非 `silu_and_mul_clamp`；MISS/ENGAGED 一次性诊断） | `sgl_kernel/_adapters/dsv4_silu_fused_quant.py`（54d45d0 → 6ef8b3e） | fp8 ulp 门（pair-bytes 逐值 + rtol 0.08）；大 M 0.42×→1×+ | B trace act_and_mul 104ms/步段收回（wall 21.65→19.62s）；B-v2 ENGAGED 四门绿，p50 975ms |
| 3 | MHC `_HC_MIX_DOT_PRECISION`/`_HC_MIX_NUM_STAGES` import-frozen 常量针（tf32x3 崩→ieee fp32 MFMA 路） | `sgl_kernel/_adapters/dsv4_mhc_stats.py`（f74faec） | B 臂 boot 绿（原 tf32x3 compile 崩） | B 臂可跑的前置 |
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
- **Bug C（host 路真凶）**：本仓 `sgl_kernel/_adapters/platform_seam.py` 把 sglang `is_hip()` 强制 False（HCU 伪装 CUDA 的平台决策）→ engram.py `_registered_device_ptr` 走 `return host_ptr` 短路——**hipHostGetDevicePointer 从未被调用**，内核拿裸 host VA，GPU 页表无此段 → VMFault。铁证：canary S 组（自己 ctypes 直连 libamdhip64）dev 全部重定位且通过；真表 dev==host 且崩；seam 环境实测 `is_hip()=False from platform_seam.py`。**修复**：指针翻译以 **libamdhip64 是否可加载**为准（find_loaded_library 失败→CDLL("libamdhip64.so") dlopen 兜底；真 CUDA 无此库 OSError→恒等不变），且翻译结果==host 即 raise 拒绝启动。
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
- **nmz18 环境恢复**：同居方修了驱动（renderD128-135 回归，8×BW1101 全空）；lzg-dsv41f-dspark 容器重启正常。新坑：本机 aiter `moe_align_block_size` 绑定=**出参式 ABI**（`(..., sorted_out, experts_ids_out, ntp_out)`），树内 fused_moe_c 老 wrapper 传参错位→段错误假象（非驱动问题）；直接 torch.ops 调用带预分配出参即通。
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
- **修法（零上游改动，纯 adapter 补洞）**：`sgl_kernel/sampling.py` 新增 triton port `_softmax_rows_kernel`（online 两遍行 softmax：pass1 online max/sum、pass2 归一写回；逐行温度，BLOCK_V=8192/num_warps=8；temp≤0 当 1.0 防除零；标量温度广播/奇 vocab/-inf 列/bf16/非连续行全兼容），绑进 `_adapters/flashinfer` allowlist + registry（MECH_SEAM）。部署前 md5 校验本地=共享盘 `hcu_v0522_nmz18`（NFS）两文件同源后才推。
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
- 遗产：`tmp_fetch/stall_probe.py`（轻负载探针，已验）、`tmp_fetch/nccl_mini.py`（8 卡 allreduce 判具）、`tmp_fetch/launch_qwen_adapter_1026.sh`（das 对比容器配方：tvm-ffi 盾+topo 清空+AR 禁用全套）、nmz18 容器 `lzg-das-qwen`（idle sleep infinity，含仓库挂载 /models_qua，停摆实验复用；不需要时可 docker rm）。
