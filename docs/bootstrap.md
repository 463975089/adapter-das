# 新容器 Bootstrap 手册（镜像 → 可跑 adapter）

> 本文回答：**拿到一个 DTK/sglang 基线镜像的新容器，要让 adapter + 官方
> sglang v0.5.21 跑起来，需要手工做哪些事、为什么、顺序如何。**
> 适用镜像：`10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604x-py3.10-*` 系
> （bw7/nmz26/nmz18/bw6 全部现役镜像）。目标耗时：照做 ≤10 分钟。
> 三类必做事项：**①依赖套件按版本重打 ②tvm-ffi 首次编译手工预热
> ③adapter tree 整体同步 + PYTHONPATH**。

## 1. adapter tree 同步：git pull 优先（仓库已开源）

仓库已 public 开源：`https://gitee.com/lizhigong/hcu-sglang-adapter`，
容器内免凭据直拉，**git 为主、tar 仅作隧道断连时的应急**。

```bash
# 首次（容器内，免凭据）：
cd /work
git clone https://gitee.com/lizhigong/hcu-sglang-adapter.git adapter-tree
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

## 3. tvm-ffi 手工预热（首次必做，防 JIT 竞争死锁）

**现象**：首次 import tvm_ffi 会 JIT 编译 torch dlpack 桥（~20-40s，
缓存 `~/.cache/tvm-ffi/` 每容器各编一次）。危险不在慢，在 **TP 多进程
场景**：spawn 的每个子进程都继承 PYTHONPATH → 跑我们的 sitecustomize →
import tvm_ffi → 缓存缺失 → 再派构建进程，**形成 122 个构建进程的父子
链（实测 GLM TP8 卡 50 分钟即此案）**。

**解法**：启动任何服务前，在干净环境手工预编译一次（<1 分钟）：

```bash
env -u PYTHONPATH python3 $(python3 -c \
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

## 4. 运行时库路径补丁

```bash
# nmz26 容器（20261012 镜像）：libomp 不在默认路径
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:$LD_LIBRARY_PATH
```

## 5. 镜像已知差异速查（选 mem-fraction 前看）

| 镜像 | 差异 | 应对 |
|---|---|---|
| bw7 0.5.18-dtk26041-20261010 | **镜像自带 DeepGEMM JIT Pre-Compile 预热会话**（每形状 5.7-7.4GB 挤 KV 预算） | EP 场景 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=false` |
| bw7 0.5.18-dtk26041-20261010 | 显存配置器带 KV 下限校验（不挤爆，直接 ValueError 报 minimum viable） | 按报错值上调 `--mem-fraction-static` 即可，是友好的 |
| nmz26 0.5.18-dtk26041-20261012 | 无 DeepGEMM 预热会话（首个形状仍 JIT，慢一次） | 无需处理 |
| 全系 | rust TreeCore 无预编译扩展、cargo 半残编不了 | 恒带 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`（cookbook §0） |

## 6. Bootstrap 完成自检（四条全绿再起服务）

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
