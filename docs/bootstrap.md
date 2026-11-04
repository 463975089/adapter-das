# 新容器 Bootstrap 手册（镜像 → 可跑 adapter）

> 本文回答：**拿到一个 DTK/sglang 基线镜像的新容器，要让 adapter + 官方
> sglang v0.5.21 跑起来，需要手工做哪些事、为什么、顺序如何。**
> 适用镜像：`10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604x-py3.10-*` 系
> （bw7/nmz26/nmz18/bw6 全部现役镜像）。目标耗时：照做 ≤10 分钟。
> 三类必做事项：**①依赖套件按版本重打 ②tvm-ffi 首次编译手工预热
> ③adapter tree 整体同步 + PYTHONPATH**。
>
> **宿主定位**：本文以 sglang 宿主为主线。依赖按版本重打、tvm-ffi 预热、
> triton/JIT 缓存自检为**双宿主通用**；§8 拉起服务与上游单测（ci.md L1.5）
> 是 sglang 宿主专属流程——vLLM 宿主的组件职责与验证见
> [frameworks/vllm032.md](frameworks/vllm032.md)。

## 1. adapter tree 同步：git pull 优先

仓库为团队内网私仓（`<团队仓库地址>`，凭据走团队流程），
有凭据环境直拉；nmz18 等无凭据节点走本地 bundle + scp 中转。**git 为主、tar/bundle 仅作网络受限时的应急**。

```bash
# 首次（容器内，免凭据）：
cd /work
git clone <团队仓库地址> adapter-tree
export PYTHONPATH=/work/adapter-tree:/work/official-sglang/python

# 日常更新（增量、带版本对账）：
cd /work/adapter-tree && git pull
git status --short    # 必须为空：非空 = 容器树有本地改动/残留，先查明再跑
git log -1 --oneline  # 必须与本地工作区 HEAD 一致——"同步好没好"以 git 可判为准
```

**为什么 git 为主**：自带版本对账（`git log -1` 即答"容器里跑的哪个
commit"，排查第一步）；增量同步；与 CI mirror mirror（L1 也是 clone+check）
同一语义——本地=唯一编辑树、容器=只跑验证、"容器里是什么"永远可考。

**应急：tar+base64 过 ssh**（仅当 git 通道不可达，如隧道断连/内网隔离）：

```bash
# 本地：tar czf adapter_tree.tgz sgl_kernel sitecustomize.py pyproject.toml
# base64 过 ssh 落宿主 → docker cp 容器内解包（scp -P 被节点 sshd 拒）
```

tar 一律解包**落进 git 树**（/work/adapter-tree）而非裸目录——`git status` /
`git diff` 立刻暴露 tar 覆盖了什么、漏了什么；通道恢复后
`git checkout . && git pull`，再按上面两条验收归位。**红线补充：任何方式
同步完，最终态必须能被 git 判读（status 干净 + log 对得上）——不可对账的
同步不算同步完成。**

**红线（两法通用）**：PYTHONPATH 指树根；绝不用逐文件补丁方式同步
（必漏新桥——历史事故：漏 platform_seam 后 spawn 子进程用旧 adapter）。
内部迭代期若用**私有 fork** 推拉，令牌只落 `/root/.git-credentials`，
不进 /work、不进 PYTHONPATH、永不 commit。

## 2. 依赖套件重打（每个新容器各做一次）

镜像自带的包与官方 v0.5.21 的 ABI 期望不齐，必须按以下版本覆盖安装
（版本是踩坑钉死的，勿"顺手升级"）：

```bash
pip install --no-cache-dir \
    numpy==2.2.6 scipy==1.15.2 ml_dtypes==0.5.1 \
    msgspec==0.20.0 \
    pydantic==2.13.1 pydantic-core==2.46.1

# fastapi 特殊处理：pip 不敢删旧目录，必须先手删再装
rm -rf /usr/local/lib/python3.10/dist-packages/fastapi/_compat/
pip install --no-cache-dir --force-reinstall fastapi==0.115.12
```

各钉版本的踩坑理由（为什么不能换）：

| 包 | 钉版本 | 偏离后果 |
|---|---|---|
| numpy | 2.2.6 | 镜像 numpy 1.x 与 v0.5.21 代码冲突 |
| ml_dtypes | **0.5.1** | 0.6 对 numpy1 崩；**0.3.2 对 numpy2 反向 hang**（无报错的死等，最难查） |
| fastapi | 0.115.12 | 必须先 `rm -rf fastapi/_compat/`——pip force-reinstall 不删旧目录 |
| pydantic-core | 2.46.1 | `cls_name` kwarg 是 2.30+ API，旧核 AttributeError |
| msgspec | 0.20.0 | 位置参数契约随版本漂 |

## 0. 容器重建 runbook（2026-10-18 实操定稿）

```bash
# 现役镜像（轮子最新；选镜像先 curl tags/list 找日期最新 tag）：
#   10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-sf_b075-20261018-0048
# 重建三步：备份 /root 自有状态到挂载区 → stop+rm → run（挂载=原 5 项+公共盘只读）
docker exec <旧容器> bash -lc 'pip freeze > /root/g53/pip_freeze_$(date +%m%d).txt && cd /root && tar czf /root/g53/container_root_backup.tgz --exclude=./hcu_t2 --exclude=./g53 --exclude=./.cache --exclude=./.triton --exclude=./.local .'
docker stop g53-tp8 && docker rm g53-tp8
docker run -d --name g53-tp8 --network host --ipc host --privileged \
  --shm-size 64g --security-opt label=disable \
  -v $HOME/models/GLM-5.3-Flash:/models/glm53f:ro \
  -v $HOME/hcu_t2:/root/hcu_t2:rw \
  -v $HOME/g53_tp4:/root/g53 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v $HOME/sglang-v0521:/sglang-v0521:rw \
  -v /public/opandas/DL_DATA/llm-models:/public/opandas/DL_DATA/llm-models:ro \
  10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-sf_b075-20261018-0048 \
  sleep infinity
```

要点：①模型一律公共盘只读挂载（用户 2026-10-18 定规：不在 $HOME 下拷模型，
运维盘压力；opandas 挂载源恢复后需 docker restart 或重建使 bind 生效）；②重建后必做
§3 tvm-ffi 预热 + §5 tilelang pin；③备份 tgz/pip freeze 在 g53_tp4（挂载区）可回灌。

## 3. tvm-ffi 手工预热（首次必做，防 JIT 竞争死锁）

**现象**：首次 import tvm_ffi 会 JIT 编译 torch dlpack 桥（~20-40s，
缓存 `~/.cache/tvm-ffi/` 每容器各编一次）。危险不在慢，在 **TP 多进程
场景**：spawn 的每个子进程都继承 PYTHONPATH → 跑我们的 sitecustomize →
import tvm_ffi → 缓存缺失 → 再派构建进程，**形成 122 个构建进程的父子
链（实测 GLM TP8 卡 50 分钟即此案）**。

**解法**：启动任何服务前，在干净环境手工预编译一次（<1 分钟）：

```bash
env -u PYTHONPATH HCU_ADAPTER_HOST=none _HCU_ADAPTER_ACTIVE_HOST=none python3 \
  $(env -u PYTHONPATH HCU_ADAPTER_HOST=none _HCU_ADAPTER_ACTIVE_HOST=none python3 -c \
  "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'utils', '_build_optional_torch_c_dlpack.py'))") \
  --output-dir /root/.cache/tvm-ffi \
  --libname libtorch_c_dlpack_addon_torch211-rocm.so \
  --build-with-rocm
```

（脚本确切路径以 `dist-packages/tvm_ffi/utils/` 实际为准；`--libname`
的 `torch211` 对应容器 torch 版本。）预热后缓存全容器共享（含 spawn
子进程），此后启动链秒过。**或者**：launch 脚本第一行加
`python3 -c "import tvm_ffi"`（带 PYTHONPATH、单进程）等效。

验证：`ls ~/.cache/tvm-ffi/` 出现 `.so` 即成。冒烟：
`PYTHONPATH=/work:/work/official-sglang/python python3 -c "pass"` 无
Traceback 输出（sitecustomize 静默完成）。

**捷径（2026-10-18 实操）**：从任一健康容器直拷现成 .so（torch 版本一致即兼容，
nmz26 g53-tp8 ← nmz18 nsa-tp8，md5 b1fb7b35 校验一致）：

```bash
# nmz18 侧 b64 出、nmz26 侧解入（或宿主 docker cp 中转），落位后 chmod +x
/root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so
```

30 秒级完成，等效于直构；上方法为无源可拷时的备选。

**自动化（rel/v0.5.22 起，2026-10-26）**：adapter `tvmffi_warmup.py` 已把本节
预热自动化——sitecustomize 前部（先于 dsa_topk_v2 等重桥）探温
`~/.cache/tvm-ffi/**.so`，冷则 flock 排他 + 剥 PYTHONPATH 亲跑 tvm_ffi 自带
构建脚本；加载器仍要自派子进程时窗口内 Popen 包装注入干净 env 物理断自递归
（R-18：桥 eager import 链使每个 python3 重进 tvm_ffi，nmz26 dsv41f 冷容器
2026-10-26 复现 9+ 进程堆积）。`HCU_TVMFFI_WARMUP=0` 可关。
**⚠ 上文「或者：launch 脚本第一行加 `python3 -c "import tvm_ffi"`（带
PYTHONPATH）」的等效变体已失效**——10-24 起 dsa_topk_v2 eager import 使带
PYTHONPATH 的裸 import 在 sitecustomize 阶段即触发自递归；手工预热一律
`env -u PYTHONPATH`（命令替换里查路径的那一步也要）。

**editable 安装补充（2026-11-03）**：安装态 `sitecustomize` 可由 site-packages
找到，仅移除 `PYTHONPATH` 仍会继承宿主并再次进入预热，子进程等待父进程
持有的同一把锁。构建子进程必须同时设置 `HCU_ADAPTER_HOST=none` 和
`_HCU_ADAPTER_ACTIVE_HOST=none`；直接 helper 和兜底 Popen 均使用复制后的
隔离环境，父进程及其他环境键保持原值。环境消费回归见
`tests/adapter/test_tvmffi_warmup.py`；此项构建隔离不改变服务后端选择。

## 4. 运行时库路径补丁

```bash
# nmz26 容器（20261012 镜像）：libomp 不在默认路径
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:$LD_LIBRARY_PATH
```

## 4b. JIT CUDA 语义工具链（cuda-mocker 配方，2026-10-19 实测定稿）

救活 dsv4 JIT 家族（CUB 头）与 causal_conv1d JIT（hipcc uint2 分解缺口）
——零装库，全部用镜像内现成的 DTK cuda-mocker（`/opt/dtk/cuda/cuda-12`，
CUDA 语义虚拟层、底层 HIP）。原理与证据见 [ci.md](ci.md) L1.5g。

```bash
# 一次性：合成 ROCM_HOME 根（镜像固化时建议直接做进镜像）
mkdir -p /opt/hcu-rocm-shim/bin
ln -sfn /opt/dtk/cuda/cuda-12/targets/x86_64-linux/include /opt/hcu-rocm-shim/include
ln -sfn /opt/dtk/lib /opt/hcu-rocm-shim/lib
cp -r /opt/dtk/.info /opt/hcu-rocm-shim/
cat > /opt/hcu-rocm-shim/bin/hipcc <<'EOF'
#!/bin/sh
# DTK cuda-mocker route: strip HIP-mode defines, inject arch (mocker maps
# gfx938 -> __CUDA_ARCH__=750), delegate to dcc -dcccuda12.
out=""
for a in "$@"; do
  case "$a" in
    -DUSE_ROCM|-D__HIP_PLATFORM_AMD__*) ;;
    *) out="$out '$a'" ;;
  esac
done
eval exec env ROCM_PATH=/opt/dtk /opt/dtk/cuda/cuda-12/bin/nvcc -DSGL_CUDA_ARCH=750 $out
EOF
chmod +x /opt/hcu-rocm-shim/bin/hipcc

# 每进程环境（起 JIT 编译的进程都需要；后续可由 hcu_env 收编自动装配）
export ROCM_HOME=/opt/hcu-rocm-shim
export LD_PRELOAD=/opt/dtk/cuda/cuda-12/lib64/libcudart.so
```

已知副作用：`ROCM_HOME` 指合成根时 deepgemm 找 `<ROCM_HOME>/.info/rocm_version`
（上面的 `cp -r .info` 即为此；缺它则 bridge 拒装并连坐 kernels 段 14 测），
长期修复为 `deep_gemm_dtk.py` 改读 DTKROOT。

**容器三件套铁律（2026-10-20 锚定轮改写，替代已废弃的"镜像↔宿主配
对"旧判）**：镜像本身跨机通用，旧"sf_b075 × nmz18 = HIP init 死锁"是
误诊——真凶双因素：①容器缺宿主 `/opt/hyhal` 挂载（`--privileged` 起，
HAL 从宿主来）；②冷 `~/.cache/tvm-ffi` 触发 tvm_ffi 选件构建器自锁
（fork 子进程再 import torch/HSA → 父子互 poll pipe 永挂，指纹=poll+
renderD fd 全开+hsakmt shm deleted——与"镜像错配死锁"完全同构）。
新机新容器标准动作：同款镜像 + privileged + hyhal 挂载 + `ci/cache_import.sh`
播种（种子现含 tvm-ffi/torch_extensions）；或对冷容器先
`export TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 兜底。验活二连：
`python -c "import torch; print(torch.cuda.device_count())"`（60s 内出
8）+ 真张量 `torch.randn(...)@` 一次（枚举走 NVML 会假活，真 init 才算）。

## 4c. aiter JIT cache：机制与收割（2026-10-20 实测定稿）

**结论先行：新环境无需为 aiter 做任何预 warm**——DTK vendor wheel 把全部
算子预编译进包（`aiter/jit/module_<op>.so`，nmz26 实测 51 颗/1.1G；
出厂镜像裸 `import aiter` 仅 3.5s、运行时编译量为零、`jit/build/` 清洁）。
编译只会发生在 wheel 未预编译的新算子/模板上，产物落
`aiter/jit/build/`（`get_user_jit_dir()` 三级：`AITER_JIT_DIR` env →
包目录可写直用 → `~/.aiter` 拷树只读兜底）。

**收割（harvest）配方**——某机 `jit/build/` 长出了产物且值得保留时：

```bash
# 1. 打包运行时编译产物（在源机、源容器内）：
J=/usr/local/lib/python3.10/dist-packages/aiter/jit
tar czf aiter_jit_harvest.tgz -C $J build
# 2. 目标机播种（严格同 wheel 版本；目标容器内解到包目录 jit/ 下）：
tar xzf aiter_jit_harvest.tgz -C $J
```

**版本成对铁律**：模块 .so 按操作名命名（无内容 hash）——跨 wheel 版本
播种旧 .so 会**遮蔽**新 wheel 的重编译（(kernel|bug 门伪装修复)）。播种前
对版本：`pip show aiter` 两端一致才可走（现役两镜像已分叉：sf_b075=
`...2610171925.ga76f3d`、20261016 老=`...2610161749.g722998`）。

**开机观测**：adapter 开机自动打一行
`HCU aiter jit cache: wheel=<ver> prebuilt_so=<N> runtime_built=<M> ...`；
`runtime_built>0` 即发生了新编译（WARNING 级、列产物）——处置二选一：
收割入 wheel 源头（报 vendor），或钉住该 wheel 版本直到收割完成。
`AITER_JIT_DIR` 可把落点重定向到久存卷（同一版本成对前提下）。

## 4d. 新容器 CI 初始化控制（cache 播种 + 段窗缩放，2026-10-20）

冷容器跑 L1 的大头=triton+sglang-JIT 首编风暴（gfx936 实测 kernels 段
45min 窗不够）与 import 冷税（bw47 实测收集 241s、开机 34s/进程）。三板
斧（keys：cache 按 ISA+树 hash 键控，同族卡+同树修订一份种子全网通用）：

```bash
# 1. 热机收割（nmz26 热容器内，~20MB）：
bash ci/cache_export.sh /tmp/hcu_cache_seed.tgz
# 2a. 冷容器导入（tar 通道）：
bash ci/cache_import.sh /tmp/hcu_cache_seed.tgz
# 2b. 或共享盘通道（NMZ 系已挂 /public/home）：热机直落共享区，
#     冷容器软链进来——零搬运：
#   ln -s $HOME/hcu_cache/sglang/jit ~/.cache/sglang/jit
#   ln -s $HOME/hcu_cache/triton     ~/.cache/triton
#     （多容器并发写共享区有踩踏风险；稳妥式=只读软链+本地 overlay 写，
#      或只软链 sglang/jit[build-<hash> 只读命中]、triton 留本地）
# 3. 慢盒段窗缩放（不改脚本）：
HCU_L1_TIMEOUT_SCALE=3 bash ci/l1_gpu_gate.sh <gpu> --up-tree /sglang-v0522
```

适配侧两件已联动：cuda-mocker 配方产物随种子旅行（loader 不匹配会自动
回退重编，坏种无害）；aiter 侧无需播种（wheel 满预编译，见 §4c）。

## 5. 镜像已知差异速查（选 mem-fraction 前看）

| 镜像 | 差异 | 应对 |
|---|---|---|
| bw7 0.5.18-dtk26041-20261010 | **镜像自带 DeepGEMM JIT Pre-Compile 预热会话**（每形状 5.7-7.4GB 挤 KV 预算） | EP 场景 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=false` |
| bw7 0.5.18-dtk26041-20261010 | 显存配置器带 KV 下限校验（不挤爆，直接 ValueError 报 minimum viable） | 按报错值上调 `--mem-fraction-static` 即可，是友好的 |
| nmz26 0.5.18-dtk26041-20261012 | 无 DeepGEMM 预热会话（首个形状仍 JIT，慢一次） | 无需处理 |
| **sf_b075-20261018-0048**（现役） | **tilelang 0.1.15 回归**：mhc tilelang 融合内核（mhc_pre_big_fuse_with_norm）`ThreadSync("shared.dyn")` pass 硬错 not-warp-uniform（0.1.12 同条件仅 Warning hoisting）；aiter/sgl-kernel/torch 轮子更新均健康 | **钉 tilelang 0.1.12**（保融合路径，勿用 SGLANG_OPT_USE_TILELANG_MHC_*=0 回退 torch 链——性能倒退）：`tar xzf /root/g53/vendor_tilelang_0112.tgz -C /usr/local/lib/python3.10/dist-packages/`（先删 0.1.15 的 tilelang/ 与 dist-info）；恢复后 `import tilelang` 应报 0.1.12+dtk2604.torch2110.2610100923 |
| **bw7/bw47（gfx936，BW1000 系）× 0.5.18-dtk26041-py3.10-20261016-2048** | ①镜像内 SMI 库路径缺失（`librocm_smi64.so.2` 只在 `/opt/dtk/.hyhal/rocm_smi/lib`）；②torch 真初始化需**宿主 `/opt/hyhal` 挂载**（漏挂=NVML 枚举 8 卡但 `torch.zeros(device='cuda')` 必挂，枚举走 NVML 提交走 HSA 的"半配对"假象）；③**HSA 栈楔死敏感**：持续 GPU churn（全套 kernels）可在套件中段楔死（poll 等 renderD + `hsakmt_shared_mem (deleted)`），timeout SIGTERM 半途杀 HSA 进程必毒化后续 python；④JIT 段窗需放宽（import 面 ~5-10× nmz，l0 60s/cpu 300s 必超） | ①`export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:...`；②docker run 必带 `-v /opt/hyhal:/opt/hyhal:ro`（win 者配 `LD_LIBRARY_PATH=/opt/hyhal/lib:...` 前插）；③楔死后 `rm -f /dev/shm/hsa* /dev/shm/hsakmt*` 即愈（不省 GPU 复位）；跑测纪律=严格串行+诊断期绝不并行 python；④段窗 ×3（240/900/2700/3600/3600，bw47 实测 import 收集 241s、开机恒定 34s/进程） |
| 全系 | rust TreeCore 无预编译扩展、cargo 半残编不了 | 恒带 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`（cookbook §0） |

## 6. Bootstrap 完成自检（四条全绿再起服务）

sglang 的 AITER 拒用分支必须在 `HCU_DISABLE_OPS=aiter_ar` 下实际检查
dispatch；不能仅检查环境摘要。2026-11-04 修复了自研 AR 退役后遗留的
`_tar_installed` 引用：旧分支抛 NameError 并跳过拒用接线，新分支直接
按当前拒用判定安装 NCCL 回退。回归入口为
`pytest tests/vendor/test_allreduce.py -k gate_disable_list`；该接线检查
不代表 AITER 数学与 Graph 压测已通过。

```bash
cd /work && PYTHONPATH=/work:/work/official-sglang/python python3 - <<'PY'
import numpy, pydantic, fastapi, msgspec   # 依赖层
import torch                                # 容器栈
import tvm_ffi                              # 预热生效（<2s）
import sgl_kernel                           # adapter 面（sitecustomize 已挂）
import sglang                               # 官方面
print("bootstrap OK:", numpy.__version__, pydantic.VERSION)
PY
```

何时需要重做：**每换新容器/新镜像整节重跑**（依赖和预热缓存不跨容器）；
同容器内只更新 adapter 代码 → 只做 §1。

## 7. 手工跑全量测试（L0 + L1，~5-15 分钟）

CI 的 MR watcher 干的就是这两步，手工验证等价于直接跑同款脚本（都在
仓内，容器内执行）：

```bash
cd /work                # adapter tree 根（git clone 或 §1 的 tar 解包处）

# L0 结构门禁（秒级，无 GPU 依赖）：语法/RLF/密钥扫描/catalog 三交叉
python3 ci/l0_gate.py                   # 期望末行: l0 gate: GREEN

# L1 GPU 数值门禁（~5-15min，占一张卡）：
bash ci/l1_gpu_gate.sh                  # 自动挑 VRAM<5% 的空卡
bash ci/l1_gpu_gate.sh 6                # 或指定卡号
# 期望末行: L1: GREEN (gpu=N)
# 内部段: l0 → cpu(test_ops_cpu) → kernels(tests/kernels+adapter)
#         → gates(config/fa3-emu-CP/rope/aiter-MoE/contig 五个 tools 门禁)
#         → vendor(flash_attn wheel 探针, WARN 级不挡门)
# 各段日志在 /tmp/l1_<段名>.log，FAIL 时自动打 tail
```

**单独跑某个域**（细粒度排查时）：

```bash
export PYTHONPATH=/work:/work/official-sglang/python    # 部分门禁需上游源码
export HIP_VISIBLE_DEVICES=$N CUDA_VISIBLE_DEVICES=$N    # 圈一张空卡

python3 -m pytest tests/test_ops_cpu.py -q              # 纯 torch 契约层
python3 -m pytest tests/kernels tests/adapter -q        # triton 数值 vs 参照
python3 tests/tools/_test_hcu_config.py                 # 单个 tools 门禁
python3 tests/tools/_test_hcu_ar2.py                    # custom AR 门禁（占 2 卡）
```

判读约定：`_test_*` 脚本全 assert 无输出=过；pytest 末行 `passed` 数
即绿；vendor 段的 FAIL 是环境事实（坏 wheel）不是回归，见 ci.md 分层。
跑完若要压性能格点，物料在 `tests/tools/_bench_*.py` 与 `_probe_*.py`
（命名规约见 tests/tools/README.md）。

## 8. 手工拉起一个 sglang 服务（最小可用示例）

前提：§1-§6 全绿。以下即 cookbook §1 基线配方（细节/其他模型展开看
[cookbook.md](cookbook.md)）：

```bash
export PYTHONPATH=/work:/work/official-sglang/python      # adapter 优先
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python       # 恒带（rust 无扩展必炸）
export CUDA_VISIBLE_DEVICES=6                              # 圈卡；多卡写 0,1

python3 -m sglang.launch_server \
    --model-path /root/models/Qwen3-0.6B \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.45 2>&1 | tee /tmp/sglang_30000.log
```

**启动正确的signature**（日志里按序出现才算起活）：

1. `HCU adapter config: …`（sitecustomize 挂载成功，逐项后端来源行）；
2. `Loading weights …`→`Capture …`（graph 桶逐个捕完，无 OOM）；
3. `The server is fired up and ready to roll!`（监听就绪）。

**冒烟一发**（另开终端）：

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user",
        "content":"法国的首都是哪里？只答城市名。"}]}' | python3 -m json.tool
# 期望 content 含"巴黎"
```

**收尾**：`pkill -f "sglang.launch_server"`（注意模式别写进同条命令的
其他参数，防自杀）；确认显存已还（`hy-smi`/`rocm-smi` 该卡 VRAM% 回
≈0；kill -9 遗留 defunct 时不还卡，换卡或重建容器）。

更复杂的形态——单测即配方的模板库：TP 多卡/EP/PD/MTP/CP/hicache/AWQ
各有实测 launch 物料 `tests/tools/_launch_*.sh`（容器内改卡号端口即
用），组合特性的门禁与探针是同名 `_test_*.py`/`_probe_*.py`。出问题
先查 cookbook.md §4 平台约束与 models.md 同模型行备注，再怀疑环境。

---
维护注：本文的版本钉与坑清单源自 2026-10-10~14 的多容器实测（models.md
对应行沿链可查）；镜像换代时先在旧表加注再按新实测修订本文件。

## 9. 参考资料

- **DTK 官方文档**(runtime/通信/调试调优,中文):<https://dtk.sourcefind.cn/docs/runtime/zh/latest/index.html>
  ——DTK 26.10 在线文档入口:HIP runtime API、RCCL/DUSHMEM 通信、
  ROCtracer/ROCprofiler/hipprof/hipgdb 调优调试。查 HIP API 语义、profiler
  用法、RCCL env 时以此为准；平台级问题（如大图发射、hipGraphLaunch 开销）
  定量分析时先翻这里。对应本文 §5 镜像差异表中 dtk2604x 的上游文档源。
