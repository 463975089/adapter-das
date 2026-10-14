# 新容器 Bootstrap 手册（镜像 → 可跑 adapter）

> 本文回答：**拿到一个 DTK/sglang 基线镜像的新容器，要让 adapter + 官方
> sglang v0.5.21 跑起来，需要手工做哪些事、为什么、顺序如何。**
> 适用镜像：`10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604x-py3.10-*` 系
> （bw7/nmz26/nmz18/bw6 全部现役镜像）。目标耗时：照做 ≤10 分钟。
> 三类必做事项：**①依赖套件按版本重打 ②tvm-ffi 首次编译手工预热
> ③adapter tree 整体同步 + PYTHONPATH**。

## 1. adapter tree 同步（整 tar，别逐文件）

```bash
# 本地（Windows，仓库根）：
tar czf adapter_tree.tgz sgl_kernel sitecustomize.py pyproject.toml
base64 adapter_tree.tgz > adapter_tree.b64          # scp 被节点 sshd 拒
# base64-of-ssh 通道落宿主 → docker cp 进容器 /work/，容器内：
mkdir -p /work/adapter-tree && tar xzf /work/adapter_tree.tgz -C /work/adapter-tree
export PYTHONPATH=/work:/work/official-sglang/python   # 指树根（tar 解包处），不是子目录
```

**为什么整 tar**：逐文件补丁必漏新桥（历史事故：漏 platform_seam 后 spawn
子进程用旧 adapter，输出形态诡异）；PYTHONPATH 指 `/work` 树根而非
`/work/adapter-tree`（tar 根解包的对齐关系）。

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

---
维护注：本文的版本钉与坑清单源自 2026-10-10~14 的多容器实测（models.md
对应行沿链可查）；镜像换代时先在旧表加注再按新实测修订本文件。
