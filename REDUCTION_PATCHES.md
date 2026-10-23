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
