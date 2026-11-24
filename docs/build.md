# 构建说明（wheel）

双 dist（governance §7.4）：`sgl-kernel` 纯影子内核 + `hcu-adapter` 适配层。
一次构建出两张 wheel，同戳互锁；构建期定版，上传环节不再改写版本号。

## 1. 前置条件

- DTK 容器：`hipcc` 在 PATH（`source /opt/dtk/env.sh`），且装有 HIP 版
  `torch==2.13.0`（必须来自 DTK nightly 归档；裸 PyPI 的同名 CUDA 构建会被
  `torch.version.hip` 检查拒绝）。
- 编译目标架构取 `AMDGPU_TARGETS` 里的**全部**值，`setup_hip.py` 为每个目标
  生成 `--offload-arch`。DTK 基础镜像/runner 环境通常已导出完整列表
  （实测 `gfx906;gfx926;gfx928;gfx936;gfx938;gfx92a`），直接沿用即可；只有
  想收窄架构、或环境确实没有该变量且无 GPU 可探测时才显式覆盖。
- 首次构建若当前环境缺 `setuptools.build_meta` / `wheel` / `ninja`，
  `build.sh` 会自动补装。
- 装 `hcu-runtime` extra 的机器需要配好 DTK nightly 归档的 pip 源
  （torch/triton/vendor 包从该归档解析）。

## 2. 一条命令构建

```bash
source /opt/dtk/env.sh
bash packaging/build.sh        # 架构沿用镜像自带的 AMDGPU_TARGETS 完整列表
# 需要收窄时才覆盖：AMDGPU_TARGETS="gfx936;gfx938" bash packaging/build.sh
```

内核 wheel 由根目录 `setup_hip.py` 编译（载荷含
`sgl_kernel/native_ops*.so`；`sgl_kernel/_native.py` 运行时探测，缺失时回退
triton/torch 档）。产物（同一个 build 标签）：

```text
dist/sgl_kernel-0.0.0+dtk2610.torch2130.<YYMMDDHHMM>.g<commit6>-cp312-cp312-linux_x86_64.whl
dist/hcu_adapter-0.5.22+dtk2610.torch2130.<YYMMDDHHMM>.g<commit6>-py3-none-any.whl
```

内核 wheel 的平台 tag 随构建 python 版本变化（上例 py3.12）。adapter 的元数据
精确依赖同标签内核，两张必须成对安装：

```text
Requires-Dist: sgl-kernel ==0.0.0+dtk2610.torch2130.<同戳>
```

HiSparse 构建 helper 必须按当前 checkout 的文件路径加载，不能经已缓存的
`hcu_adapter` 包名导入：旧安装的 `sitecustomize` 即使在 HOSTnone 下也可能
预先缓存旧包；只把 checkout 放到 `sys.path` 首位不能改变该包的 `__path__`。
`setup_hip.py` 与版本 pin 一样从自身目录确定 helper；缺失文件直接失败，
不回退到旧安装、不清空或重绑全局包缓存。离线回归覆盖 all-ops 与
`--hisparse-only`、旧包/旧 helper 缓存及缺失 checkout helper；build API doubles
只证明构建入口的来源与参数，不能替代真实 HIP 编译或配套 wheel 验证。

## 3. 版本与戳规则

| 项 | 来源 | main 当前值 |
|---|---|---|
| 内核基线 | `sgl_kernel/version.py` 的 `__version__` | `0.0.0+hcu.dev0` |
| adapter 基线 | `packaging/hcu-adapter/VERSION` | `0.5.22+hcu.1` |
| 平台标签 | `dtk<DTK_VER>.torch<torch 主三段数字>` | `dtk2610.torch2130` |
| 构建戳 | `git rev-parse --short=6 HEAD` + 当前时间 | `<YYMMDDHHMM>.g<commit6>` |

渲染规则：去掉基线里的 local 段，再拼 `+<平台标签>.<戳>`（**不带 `hcu` 段**，
对齐 DTK nightly 里 vendor 包的版本惯例）。DTK 段来源优先级
`DAS_DTK_TAG` > 镜像 ENV `DTK_VERSION` > `DTK_VER`；torch 段来自当前环境的
`torch.__version__`。构建出的扩展与这两者 ABI 绑定，所以**无戳时平台标签仍保留**。

直接 `pip wheel`（测试/开发用的纯 python 变体，不走 `setup.py` 的 native 路径）
没有平台标签，仍渲染 `+hcu.<戳>`。

基线在 main 上不固定（内核 `0.0.0+hcu.dev0`），拉 release 分支时定死；
重配对上游时改对应文件即可。

固定戳（复现某次构建）：

```bash
DAS_BUILD_STAMP="2611111025.ga64ad6" bash packaging/build.sh
```

## 4. 不带戳构建

```bash
bash packaging/build.sh --no-stamp
```

去掉时间/commit 后缀，保留 ABI 平台标签：

```text
dist/sgl_kernel-0.0.0+dtk2610.torch2130-cp312-cp312-linux_x86_64.whl
dist/hcu_adapter-0.5.22+dtk2610.torch2130-py3-none-any.whl
```

此时 adapter 仍钉精确版本 `sgl-kernel==0.0.0+dtk2610.torch2130`（不是公共版本
等号，避免装到不同平台或不同 build 的内核）。

## 5. 安装

先装内核（`--no-deps` 防止它自身的 `torch==2.13.0` / `triton==3.7.0` 与
镜像里已装的 DTK 栈重新解析），再装有运行时依赖的 adapter：

```bash
pip install --no-deps "$(ls -t dist/sgl_kernel-*.whl | head -1)"
pip install "$(ls -t dist/hcu_adapter-*.whl | head -1)[sglang-runtime,sglang-diffusion,hcu-runtime]"
```

- adapter 基础依赖已钉 `packaging>=23` / `torch==2.13.0` / `triton==3.7.0`，
  即使不带 extras 也不会漂到最新 nightly。
- 三个 extra 的分工：`sglang-runtime` 官方 sglang 运行时、`sglang-diffusion`
  扩散入口、`hcu-runtime` DTK vendor 栈（aiter/flash_attn/deep_ep/nixl/vllm 等）。
- `hcu-runtime` 另有三个可独立安装的子 extra：`hcu-runtime-kernels`
  （aiter/flash_attn/deepgemm 等内核）、`hcu-runtime-serve`（vllm/tilelang/
  lmcache/mooncake/nixl）、`hcu-runtime-aux`（torchvision/torchaudio）。
  `hcu-runtime` 通过自引用这三个子 extra 组合，包清单只在子 extra 维护一处
  （L0 校验组合关系）。镜像构建按子 extra 分 `RUN` 层、最后补装
  `hcu-runtime` 收尾，可避免单个 docker layer 超过仓库上限；torch/triton 是
  基础依赖，先单独钉一层效果更好。
- 官方 sglang 本体以 `--no-deps` 安装，依赖由 adapter extra 提供。
- 从旧单 dist 版升级：先 `pip uninstall sgl-kernel`，否则同名残留与
  entry-points 双注册。

## 6. 自检

```bash
python3 ci/l0_gate.py      # 双 dist 结构门禁（版本源、互锁、entry-points）
bash packaging/verify.sh   # A/B/C 三组合安装验证，需要 python venv
```

`verify.sh` 需要目标机 python 带 `venv`/`ensurepip`；基础镜像里若缺 venv，
在镜像内执行或改用 `python3 -m venv --system-site-packages` 验证。

查看 wheel 元数据：

```bash
python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode())" dist/hcu_adapter-*.whl | grep -E "Version|Requires-Dist"
```

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `BackendUnavailable: Cannot import 'setuptools.build_meta'` | 环境缺 setuptools；重跑 `build.sh` 会自动安装，或手动 `pip install "setuptools>=64" wheel` |
| `build.sh: torch is not importable` / `not a HIP/DTK build` | 先装 DTK nightly 归档的 `torch==2.13.0`；裸 PyPI 的 CUDA 构建不可用 |
| `build.sh: hipcc not on PATH` | 在 DTK 容器内 `source /opt/dtk/env.sh` |
| `build.sh: AMDGPU_TARGETS is empty and no HCU device is visible` | 镜像/环境未导出时显式设置，如 `AMDGPU_TARGETS="gfx936;gfx938"` |
| 内核 wheel 里没有 `sgl_kernel/native_ops*.so` | native 编译未生效（回落纯 python）；检查 hipcc/torch 与编译日志 |
| 解析到 `torch 2.15.0` / `triton 3.9.0` | 只装了 kernel 或旧 adapter。装带新基础依赖的 adapter（钉 2.13.0/3.7.0） |
| `No matching distribution found for sgl-kernel==...+dtk....` | 先本地装同标签 kernel wheel；本地文件安装时 pip 不会去 index 找已满足的依赖 |
| `dist/` 多版本混装 | 构建前 `rm -rf dist build`，或按标签显式指定 wheel 文件名 |
| 构建日志的版本与 dist 文件不符 | 检查是否残留旧的 `DAS_BUILD_STAMP` / `DAS_PLATFORM_TAG`；`--no-stamp` 只忽略前者 |

## 8. CI

main 线的 nightly 由
[.github/workflows/nightly-wheel.yml](../.github/workflows/nightly-wheel.yml)
在 push 时构建两张同标签 wheel 并一次性上传 nightly 归档（L0 结构门禁已在 MR
阶段执行，发布流程不重复）；`workflow_dispatch` 只构建并保留 artifact，
不推 nightly。workflow 在 DTK 容器里先从 nightly 归档补装 python 侧唯一缺件
`torch==2.13.0`（其余编译依赖由 build.sh 兜底补装），再跑
`packaging/build.sh`（native 编译，架构取 `AMDGPU_TARGETS`）；release 线同理：
workflow 在构建期写入同一个 `DAS_BUILD_STAMP` 后直接上传，不做上传后版本改写。
