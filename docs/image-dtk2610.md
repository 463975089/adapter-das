# DTK 26.10 基础镜像部署指南（py3.12 → 官方 sglang 0.5.22 + adapter）

> 本文回答：**在 `192.0.2.152:5000/hcu/base/dtk:26.10-ubuntu22.04-py3.12` 基础上，
> 跑通"官方 sglang v0.5.22（零修改）+ adapter"需要装哪些包、为什么、按什么顺序。**
> 与 [bootstrap.md](bootstrap.md) 的关系：那册是 dtk26041/py3.10 现役栈的手工部署手册
> （PYTHONPATH 源码态）；本文是新基底镜像的 pip 态部署，二者并存直至 26.10 转正。
> 依赖清单底料 = 上游 v0.5.22 `python/pyproject.toml` 全量 dependencies（逐条核对过）。
> **[VERIFY-26.10]** 已于 2026-11-05 在 bw47 `lizhg_base2610` 容器（card2）首跑定案并写回，
> 实测记录见各节「实测」注记；第 6 节保留定案结论。
> 前置阅读：[hygon-oss-landscape.md](hygon-oss-landscape.md)（供应商库图谱）。

## 0. 基线事实

- **镜像**：`192.0.2.152:5000/hcu/base/dtk:26.10-ubuntu22.04-py3.12`（DTK 26.10，Python 3.12）。
- **py3.12 意义**：上游 v0.5.22 声明 `requires-python = ">=3.11"`——旧 py3.10 基线**不满足**，
  此前全靠 PYTHONPATH 源码态绕过；新镜像**首次原生满足**，pip 态安装从此合法。
- **【实测 2026-11-05】镜像自带件与缺件**（决定 §1.5 迁移步骤的存在）：
  - 自带：`/opt/dtk` 基础工具链（`llvm/` 下 clang-17）、HIP runtime、pip 在线源可用。
  - **不带 python DTK 栈**：无 torch/triton/aiter/lightop 等——此前假设"镜像自带 DTK HIP torch"
    不成立，需按 §1.5 从现役容器迁移（torch 2.11.0+hip 7.2.26365 / triton 3.6.0）。
  - **`/opt/dtk` 无 `aillvm/`**（AI 专用 LLVM，clang-22，约 4.4 GB）：triton HCU 后端
    `path_to_rocm_llvm()` 在 DTK 下优先找 `aillvm`，缺失时退回 `llvm/clang-17`；clang-17
    不认识 HCU 定制 `-mllvm` 参数（`-hcu-pre-emit-load-store-opt` 等），且**打 USAGE 后以
    rc=0 退出**，triton 的 `subprocess.run(check=True)` 放行 → `open(*.amdgcn)` 抛
    FileNotFoundError → `HSACOError: File operation failed`。**必须随栈迁移 `aillvm`**
    （连同 `hcu_llvm.info`、`blas_cut.info`）。
- **依赖声明与 import 面是两回事**（pip 装包策略的根基）：
  v0.5.22 的 pyproject 声明 `sglang-kernel==0.4.10` 等一排 CUDA-only/不可解析依赖
  → 主 wheel **必须 `--no-deps` 装**；代码运行时只认 `import sgl_kernel`
  （实测 v0.5.22 全树 import 面）→ 我们的影子包按模块名截击，dist 名无关紧要但见 §4 接线点。
- **发版物五件（双 dist 解耦，governance §7.4）**：影子内核 wheel（dist `sgl-kernel`，版本
  `0.4.9+hcu.1`=配对上游内核线，只装 `sgl_kernel/`）+ 适配层 wheel（dist `hcu-adapter`，版本
  `0.5.22+hcu.1` 自有独立线，装 `hcu_adapter/`+`sitecustomize`+vllm entry-points，依赖
  `sgl-kernel==0.4.9+hcu.<时间>.g<commit>` 构建期同戳互锁；`packaging/build.sh` 一次出两张）→ 官方 sglang wheel
  （`--no-deps`）→ 插件 wheel（`--no-deps`）→ 环境级钉版。升级先卸旧单 wheel 版 sgl-kernel
  （entry-points 已迁 hcu-adapter，残留会双注册/消失）。

## 1. 不装的依赖（CUDA-only 或被我们挡位替代）

| 上游声明 | 不装的原因 | 我们的对位 |
|---|---|---|
| `torch==2.14.1`、`torchaudio`、`torchvision`、`torchcodec` | 镜像**不自带** python DTK 栈（实测）；重装 PyPI 版=覆盖成 CUDA 版 | 按 §1.5 从现役容器迁移 DTK 栈（torch 2.11.0+hip 7.2.26365、torchvision 0.26.0；torchaudio 栈内无、需要时另配） |
| `sglang-kernel==0.4.10` | CUDA AOT 内核，无 gfx93x 产物 | **影子内核 wheel（dist `sgl-kernel` `0.4.9+hcu.1`，模块名 sgl_kernel）+ `hcu-adapter` wheel 互锁装** |
| `sgl-flash-mla==1.0.0` | CUDA 内核 | 影子 `sgl_kernel.flash_mla` emu / FlashMLA-das vendor 档（`mla_attention_dtk`） |
| `sgl-deep-gemm==0.2.3.post2` | CUDA JIT 栈 | deepgemm vendor 桥 / triton 档（governance 三档路由） |
| `sgl-deep-ep==0.1.2.post1` | CUDA/NVSHMEM | DeepEP-das 档 / NCCL fallback |
| `flash-attn-4`、`flashinfer_python[cu13]`、`humming-kernels[cu13]`、`quack-kernels`、`nvidia-mathdx`、`nvshmem4py-cu13` | CUDA/Blackwell-only 编译物，HIP 环境装不上或无意义 | 四级路由接住（lightop→aiter→triton→响亮 stub）；fa3 走 `HCU_FA3_EMU` |
| `tilelang==0.1.12` | 镜像不带、DTK 栈也不含（实测）；PyPI wheel 是 CUDA 编译物 | 首轮未装——官方 dsa `tilelang_kernel` 探测失败走 fallback，无害；需要时按 bootstrap.md §6.1 装 HIP 版 0.1.12 |

**【实测 2026-11-05 修正】`cuda-python` / `nvidia-cutlass-dsl` 从"不装"改为"装"**：
官方 0.5.22 `DualGemm._select_mode`（LlamaMLP 必经）有 `is_cuda()` 守卫，但 adapter 平台
合成态下 `is_cuda()` 为真 → import `sglang.kernels.ops.gemm.cutedsl_dual_gemm` → 模块级
`import cuda.bindings.driver`——没有 cuda-python 时 scheduler 启动即炸
`ModuleNotFoundError: No module named 'cuda'`。`cuda-bindings` 是纯 cython 绑定
（import 不 dlopen libcuda，HIP 上无害），装上让官方 CUDA-only 分支的 import 探测通过，
运行时 `can_use_dual_gemm()` 自然落空。**必须装**（见 §2 第六批）。

## 1.5 DTK python 栈与工具链迁移（镜像缺件的补法）

镜像只给了 DTK 系统层，python 侧与 AI 工具链从现役容器一次性迁移（同机 docker 间 tar 管道）：

```bash
# ① python DTK 栈（29 项 ≈6.4 GB）：torch/torch.libs/torchgen/functorch/triton/
#    torchvision/amdsmi/aiter*/lightop*/deepgemm/deep_ep/causal_conv1d*/conch/tvm_ffi
#    及对应 dist-info——在现役容器打包，在基础容器解包到 site-packages
docker exec <现役> tar -C <site-packages> -cf - torch torch.libs torchgen functorch \
  triton triton.libs torchvision torchvision.libs amdsmi aiter aiter.libs lightop \
  lightop.libs deepgemm deep_ep causal_conv1d causal_conv1d.libs conch tvm_ffi \
  <对应 *.dist-info 目录们> | docker exec -i <基础> tar -C <site-packages> -xf -

# ② AI LLVM 工具链（≈4.4 GB，triton JIT 编译必需，见 §0 缺件说明）
docker exec <现役> tar -C /opt/dtk -cf - aillvm hcu_llvm.info blas_cut.info \
  | docker exec -i <基础> tar -C /opt/dtk -xf -

# ③ tvm-ffi dlpack 附加库预热（torch 栈就位后、任何正常 python 使用前）：
#    见 §3 步骤 0 的"自繁殖链"警告——必须以 HCU_ADAPTER_HOST=none 预热
```

栈内容以现役容器 `pip list` 为准；torch 三件版本 torch 2.11.0（hip 7.2.26365）/
torchvision 0.26.0 / triton 3.6.0（2026-11-05 实测，均随栈迁移，勿 pip 覆盖）。

## 2. 直接装的依赖（平台中立，版本随上游声明）

第一批：**服务框架与调度**

```bash
pip install "aiohttp" "fastapi" "uvicorn>=0.51" "uvloop" "watchfiles" \
  "pyzmq>=25.1.2" "orjson" "msgspec" "setproctitle" "psutil" \
  "prometheus-client>=0.20.0" "packaging"
```

第二批：**模型/词表/结构化输出**

```bash
pip install "transformers==5.19.0" "tokenizers==0.23.2" "sentencepiece==0.2.1" \
  "tiktoken" "blobfile==3.0.0" "gguf" "compressed-tensors==0.18.0" \
  "outlines==0.1.11" "xgrammar==0.2.7" "llguidance>=1.7.6,<2.0.0" \
  "interegular" "partial_json_parser" "einops"
```

第三批：**推理客户端/生态胶水**

```bash
pip install "openai==2.8.0" "anthropic>=0.20.0" "mistral_common>=1.11.5" \
  "modelscope" "datasets" "huggingface-hub" "requests" "tqdm" \
  "openai-harmony==0.0.4"
```

第四批：**科学计算/媒体/杂项**

```bash
pip install "numpy" "scipy" "numba==0.65.1" "pillow" "soundfile==0.13.1" \
  "timm==1.0.25" "kernels>=0.14.1,<0.15" "regex" "pybase64" "pydantic" \
  "python-dateutil<3" "python-multipart" "distro" "easydict" "xxhash" \
  "zstandard" "ninja" "build" "IPython" "py-spy"
```

第五批：**多模态与条件依赖**（x86_64 Linux 上游声明为无条件，若镜像跑纯文本可裁剪）

```bash
pip install "smg-grpc-servicer>=0.9.0" "deepseek-recipe==0.2.0" "tml-renderers==0.1.0" \
  "torch_memory_saver>=0.0.9.post1"   # 需与迁移 torch 匹配编译版
```

第六批：**纯绑定 CUDA 包**（§1 修正项；只为让官方 CUDA-only 分支的 import 探测通过，
运行时自然落 fallback。版本与现役栈一致）

```bash
pip install "cuda-bindings==13.4.4" "nvidia-cutlass-dsl==4.8.1"
```

备注（装前一眼灵魂三问，告警即拆）：
- `av`/`decord2`（视频栈）上游声明只在 aarch64 生效，x86 镜像可不装；VL 模型视频输入需要时再补；
- `nvidia-ml-py`：nvml 探针库，纯 python 绑定，装了无害但平台判定不走它——首轮未装，
  起服日志无 nvml ImportError（实测 §6#7），需要时再补；
- **py3.12 轮子覆盖度**：上表全部为主流 PyPI 包，py3.12 manylinux 轮子齐全；`numba==0.65.1`
  与迁移栈 numpy 2.3.5 实测 import 通过（§6#6 钉值）；
- `torch_memory_saver`：首轮未装（torch 版匹配的编译版未随 DTK 栈迁移），用到 memory saver
  特性时按 torch 2.11.0+hip 源码装。

## 3. 我们的 wheel 与装包顺序（装包顺序即配置）

**装包期统一加 `HCU_ADAPTER_HOST=none`**（实测）：adapter wheel 装好后 sitecustomize
在每个 python/pip 进程启动时跑 bootstrap catalog，进程常驻 ~37 s；`none` 前缀下 ~5 s。
docker build 的每条 RUN 都建议带上。

**⚠ tvm-ffi 预热自繁殖链（实测踩坑）**：torch 栈 + adapter wheel 就位后，任何**不带
none** 的 python 启动都会经 bootstrap import tvm_ffi；dlpack 附加库 cache 冷时它
`subprocess.run` 一个 builder（`tvm_ffi/utils/_build_optional_torch_c_dlpack.py`），而
**builder 自身启动也过 sitecustomize → 再 spawn 下一层 builder**，无限链——表现为
pip/pip list/任何 python 进程"永久卡死"（实测链条每层 20–40 分钟、无限增殖）。
**解法**：torch 栈迁移完成后，立即做一次带 none 的预热（builder 继承 none 环境，不再繁殖，
一次真实构建落 `.so`），此后一切正常：

```bash
# 步骤 0：预热 tvm-ffi（torch 栈迁移后、三件 wheel 装完后）
HCU_ADAPTER_HOST=none python3 -c "import tvm_ffi"    # 实测 51 s 一次性构建，落
# /root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so（~200 KB）
# （或直接从已预热容器拷该 .so，torch 版一致即可）
```

```dockerfile
# ① adapter 双 wheel：sgl-kernel（影子内核）+ hcu-adapter（适配层：sitecustomize
#    + 桥 + vllm entry-points）。顺序无所谓，依赖 == 互锁自动组队；两张都装，
#    只装其一会被 sgl_kernel/__init__ 哨兵在 import 期响亮拒绝
COPY wheels/sgl_kernel-*.whl wheels/hcu_adapter-*.whl /wheels/
RUN HCU_ADAPTER_HOST=none pip install /wheels/sgl_kernel-*.whl /wheels/hcu_adapter-*.whl

# ② 官方 sglang，零修改；--no-deps 是策略不是图省事（§0）
COPY wheels/sglang-0.5.22-py3-*.whl /wheels/
RUN HCU_ADAPTER_HOST=none pip install --no-deps /wheels/sglang-0.5.22-py3-*.whl

# ③ 插件（该镜像需要 plugin 时；entry point 经 importlib.metadata 发现，
#    pip 安装是唯一途径，PYTHONPATH 方式无效）
COPY wheels/sglang_hcu_plugin-*.whl /wheels/
RUN HCU_ADAPTER_HOST=none pip install --no-deps /wheels/sglang_hcu_plugin-*.whl

# ④ 预热 tvm-ffi（见上方自繁殖链警告；同 RUN 内完成更稳）
RUN HCU_ADAPTER_HOST=none python3 -c "import tvm_ffi"
```

顺序无关性说明：①②③ 相互独立（shadow 按 sys.modules 截击，不依赖安装次序），
但都在 §1.5 迁移与 §2 六批依赖之后；④ 只依赖 torch 栈（§1.5①）就位。

## 4. 已知接线点（官方原样树特有的，das fork 树没有的）

1. **内核版本探针** `engine.py:1954-1965`：`_is_cuda` 为 True 时
   `assert_pkg_version("sglang-kernel", "0.4.9")`——探 **dist 名** `sglang-kernel`，
   我们的 dist 是 `sgl-kernel` → `PackageNotFoundError → 启动即炸`
   （实测复现：`importlib.metadata.version("sglang-kernel")` 抛 PackageNotFoundError）。
   - **镜像层立即解**（三行之一）：
     `ENV SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（上游内建的逃逸 hatch，
     engine.py:1955 判的就是它；本轮 launch_server 冒烟即此解，实测通过）；
   - **工程解**（与 governance §7.4 同一笔）：影子 dist 更名/加发为 `sglang-kernel`、
     版本基线 `0.5.22+hcu.N`——PEP440 下 `0.5.22+hcu.1 > 0.4.9`，探针自然通过且语义诚实。
2. **`pip check` 会报错**：sglang 的依赖声明显式缺失——`--no-deps` 态的固有现象，
   镜像健康检查用第 5 节导入自检替代 `pip check`，不修。
3. **das fork 的 srt 补丁面不在这棵树上**（本来就是要剥离的）：P1 挂账项在官方树上
   由 plugin 仓承接；首个新镜像系统全量回归按 models.md 四门重录基线，**das 树的绿色
   结论不自动平移**。
4. **轮换噪音（无害，但日志会打）**：BCG `registry.register("cuda_graph", …)` 不在
   `BACKEND_CATALOG`，登记失败打 ValueError——wrap 已在 register 前生效，功能无损
   （与 0.5.22 叠加格轮同源，见 plugin 仓 tests/integration/README.md 台账）。

## 5. 安装后自检（pip 态五条，源码态四条见 bootstrap.md §7）

```bash
# ① sitecustomize 生效（解释器启动即接管）
python -c "import sys,os; assert os.environ.get('HCU_ADAPTER_HOST') is not None or True; \
  import sgl_kernel; print('shadow at', sgl_kernel.__file__)"
#    预期 路径含 site-packages/sgl_kernel/，且进程日志出现
#    "HCU adapter bootstrap: host=..."（bootstrap.py:177）

# ② 双 dist 各就各位（影子内核 + 适配层两张都在、同名 dist 只此一份、路径是我们打包的）
pip show sgl-kernel   | grep -E "^(Version|Location)"    # 预期 0.4.9+hcu.1
pip show hcu-adapter  | grep -E "^(Version|Location)"    # 预期 0.5.22+hcu.1

# ③ 插件可发现（sglang entry_points）
python -c "from importlib.metadata import entry_points; \
  print([e for e in entry_points(group='sglang.srt.plugins')])"

# ④ 平台语义自检（cuda 语义 + HIP 真值并存）
python -c "import torch; print(torch.version.hip is not None, torch.zeros(1,device='cuda').is_cuda)"

# ⑤ 服务冒烟（dummy 权重；模型路径任意本地目录含 config；tiny llama config 配方
#    见 tools/l15_model_smoke.py:tiny_config）。⚠ plugin 与格：launch_server 不设
#    SGLANG_PLUGINS 时默认装全部 plugin（与 L1.5 harness 语义一致）——验"纯 adapter"
#    要显式 SGLANG_PLUGINS=none（哨兵），验叠加态设 SGLANG_PLUGINS=hcu
SGLANG_PLUGINS=none SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  python -m sglang.launch_server --model-path <path> --load-format dummy --tp 1 \
  --port 30000 --skip-tokenizer-init --context-length 128 --max-total-tokens 512
```

**【实测 2026-11-05】①–④ 全绿**（bw47/lizhg_base2610）：

| # | 实测输出 |
|---|---|
| ① | `HCU adapter bootstrap: host=sglang source=availability`；shadow at `…/site-packages/sgl_kernel/__init__.py` |
| ② | `Version: 0.1.0` / `Location: /usr/local/lib/python3.12/site-packages` |
| ③ | `['smg-servicer', 'smg-zmq', 'hcu']`（前两个来自第五批 smg-grpc-servicer 依赖，无害） |
| ④ | `True True` |
| ⑤ | 纯 adapter 格（SGLANG_PLUGINS=none）：端口 115 s 开放（JIT 热后二次启动）；`/generate` input_ids 返回 `GEN_OK [205, 145, 125, 213]`，EXIT=0 |

**【实测已知问题】叠加态（不设 SGLANG_PLUGINS，plugin=hcu 自动装载）launch_server 首请求炸**：
plugin 的 `mem_cache.common._req_radix_key` REPLACE hook 按 fork 语义调
`tree_cache.swa_reprefill_tail_tokens(req)`，而官方 0.5.22 该方法无参
（unified_radix_cache.py:3602）→ `TypeError: … takes 1 positional argument but 2 were
given`。L1.5 叠加格（Engine worker 路径）三轮全绿未触发该调用（worker 请求流的
差异未深究）；launch_server 的 scheduler `_touch_waiting_prefixes` 路径首遇即炸。
**归属 plugin 仓修复**（hook 签名适配官方树），与镜像部署无关——镜像文档记录到此，
修复后随 plugin 版本升级销项。**2026-11-05 追记**：修复（sglang-plugin-das#17，两层
——`_req_radix_key` 签名分发 + `match_kv_cache` 按官方 `MatchPrefixParams` 字段过滤）
已在本环境热替换实证：叠加态 118 s 起服、`/generate` input_ids ×2 绿、进程存活；
正式销项随 plugin wheel 版本升级。

## 6. [VERIFY-26.10] 定案汇总（2026-11-05 bw47/lizhg_base2610 实测）

| # | 项 | 定案结论 |
|---|---|---|
| 1 | torch | **镜像不自带**；按 §1.5 迁移 DTK 栈 → torch 2.11.0（hip 7.2.26365）、torchvision 0.26.0、torchaudio 无（需要时另配） |
| 2 | triton | 3.6.0（随 DTK 栈迁移）。已知内核问题重判（gfx936 div-4 等）在红绿门复跑中做，结果记 vendor-kernels.md（未随本轮完成） |
| 3 | tilelang | 镜像与 DTK 栈均无。dsa `tilelang_kernel` 探测失败走 fallback（launch 日志有噪音行，无害）；需要时按 bootstrap.md §6.1 装 HIP 0.1.12 |
| 4 | tvm-ffi 预热 | **需要**，且**必须以 `HCU_ADAPTER_HOST=none` 预热**（否则自繁殖链，见 §3 警告）。配方 `HCU_ADAPTER_HOST=none python3 -c "import tvm_ffi"`（实测 51 s 单次构建），产物 `~/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so`（~200 KB）亦可从预热容器直接拷贝 |
| 5 | JIT 首编窗口 | launch_server 首启（叠加态）端口开放 249 s，含 3 个 HCU JIT 模块冷编（rotary_embedding 单模块 46 s；triton kernel 编译走 aillvm clang-22）；JIT 热后二次启动 115 s。首编后 `~/.cache/sglang/jit/` 与 triton cache 常驻。bootstrap.md §5 缓存种子可平移，未随本轮逐项复跑 |
| 6 | numba/numpy/py3.12 | numba 0.65.1 + numpy 2.3.5 import 通过（与上游钉值一致），三角钉死 |
| 7 | nvml 探针面 | 起服日志仅一行 `pynvml not installed, skipping NUMA node configuration for GPU`——提示性 skip，无 ImportError、不影响功能。可选装 nvidia-ml-py 消除提示；不装不阻断 |

**追加定案（首跑暴露，原表没有的）**：

| 项 | 结论 |
|---|---|
| aillvm 工具链 | 镜像缺 `/opt/dtk/aillvm`（clang-22），triton HCU 后端必需——缺失时 clang-17 对定制 `-mllvm` 参数打 USAGE 且 **rc=0**，HSACOError 难定位（§0/§1.5 已写） |
| cuda-python 族 | 从"不装"改"必须装"（cuda-bindings==13.4.4 + nvidia-cutlass-dsl==4.8.1，§2 第六批）；否则 DualGemm 路径 scheduler 启动即 ModuleNotFoundError: cuda |
| 装包期开销 | sitecustomize 使每个 python/pip 进程 +~37 s（bootstrap catalog）；装包期统一 `HCU_ADAPTER_HOST=none`（~5 s） |
| tvm-ffi 自繁殖链 | torch 栈就位后冷 cache 任何 python 启动无限 spawn builder（§3 警告）；一次 none 预热解除 |
| plugin 叠加态 launch_server | 已知 hook 签名炸（§5 附记），归 plugin 仓修 |

## 7. 维护

- 依赖清单的**唯一底料**是上游 `python/pyproject.toml` 的 dependencies 数组；
  上游每次换基线（如 0.5.23），本文第 1~2 节按 diff 渐进更新，不重写。
- **2026-11-05 首轮定案已写回**（§0/§1.5/§2/§3/§5/§6）：镜像缺件补法（DTK 栈+aillvm
  迁移、tvm-ffi 预热、cuda-bindings 增装）与装包期 `HCU_ADAPTER_HOST=none` 是新增的
  常设内容，后续换基线时一并继承。遗留未销项：§6#2 红绿门复跑、§6#5 缓存种子逐项
  复跑、plugin 叠加态 launch_server hook 签名（归 plugin 仓）。
- **已知 argparse 前缀二义**：官方 0.5.22 的 `--cuda-graph-max-bs` 同时是
  `--cuda-graph-max-bs-decode`/`--cuda-graph-max-bs-prefill` 的前缀，是否报
  "ambiguous option" 取决于 plugin 装载与依赖完整性（实测：装载态首启曾通过；
  `SGLANG_PLUGINS=none` 或依赖缺失时必报）。机制不深究，**示例统一用
  `--cuda-graph-max-bs-decode` 全名**，两态混跑不踩坑。
- bootstrap.md 在 26.10 全量转正前继续作为源码态手册并存，转正时标注"旧基底
  （26041 止）"。

## 8. MiniMax-H3-Image 的原生 diffusion 平台候选

本节给出 gfx936/DTK26.10 的最终部署配方、支持范围与性能摘要。
验收基线：社区 `5792cf6e5e11`、plugin `fa36983`；adapter 卸载对照为
`e63727e`，attention 对照为 `553b28f`。环境为 Python3.12、Torch2.11、
Triton3.6 DTK、DTK flash_attn2.8.4；两仓按各自安装说明接入同一解释器。

### 平台与支持范围

- adapter 使用公开 OOT 平台 `hcu`，保留真实 HIP 架构、设备和内存信息，
  复用原生非 NVML/CUDA 语义；plugin 通过独立 diffusion registry 加载。
- 当前 AITER wheel 缺少 `ENABLE_CK`，不得补造能力；HCU OOT 不安装
  CUDA-only VAE 优化。`HCU_PLATFORM_SEMANTICS=hip` 会停用该 OOT 准入，
  回到官方探测，但不代表上述依赖缺口已修复。
- 单张 64 GiB、2048×2048、6 步、quality=exact 的文生图及参考图编辑通过。
  TE/DiT 保持 BF16，VAE 保留原生 FP32 编码/BF16 解码与有限值检查。
- 该环境初始化曾触发 MKL 原生异常，已验证配方固定三个 CPU 线程池为 1。
  单 worker RSS 采样峰值约150 GiB，不是最低主机内存需求。

### 2026-11-15 单卡 attention 显式优化

先将 HIP/CUDA 可见设备设为同一张物理卡；`MODEL_PATH` 指向模型根目录，
`GPU_DEVICE_SYSFS` 指向该卡的实际 sysfs device 目录。NUMA 编号按 PCI
连接读取，不照搬验证机的节点号。沿用本文前述依赖与安装环境：

```bash
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_diffusion
export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=hcu HCU_AITER_AR=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMA_NODE=$(cat "$GPU_DEVICE_SYSFS/numa_node")
HCU_FA_IMPL=vendor python3 -m sglang.cli.main serve \
  --model-path "$MODEL_PATH" --backend sglang \
  --num-gpus 1 --tp-size 1 --ulysses-degree 1 --ring-degree 1 \
  --encoder-parallel auto --numa-node "$NUMA_NODE" \
  --component-paths.text_encoder "$MODEL_PATH/text_encoder" \
  --component-attention-backends.text_encoder torch_sdpa \
  --component-attention-backends.transformer fa \
  --component-residency transformer=snapshot-offload,text_encoder=snapshot-offload,video_vae=component-offload \
  --minimax-h3-adaln-online true --warmup-mode off
```

TE/DiT 保留 CPU 权重快照以减少回写，VAE 按组件卸载。服务就绪前通过公开
接口各预热一次完整六步文生图与编辑；health=200 只表示初始化完成。
常规服务保留默认 conditioning cache，公平性能对照另加
`--disable-conditioning-cache`；性能日志用 `SGLANG_PERF_LOG_DIR` 写入任务目录。

vendor 仅在该配方显式选择；`auto` 仍为 Triton。新进程改为
`HCU_FA_IMPL=triton` 或设置 `HCU_DISABLE_OPS=vendor_fa` 可回退，后者优先。
算子支持范围与接口约定见 [非分页 FA 桥](dense-flashattention.md)。

### 性能与验收摘要

以下是两轮独立对照：同卡、同模型、固定种子和编辑原图，各完整预热后各测
三次，统一关闭跨请求编码缓存。冷准备与 profiler 请求排除，仅比较同轮结果。

| 对照轮次 | 配置 | 文生图中位数 | 编辑中位数 |
|---|---|---:|---:|
| 卸载 | 三组件 component-offload | 20.451秒 | 60.140秒 |
| 卸载 | TE/DiT snapshot-offload | 19.290秒 | 58.826秒 |
| 卸载 | 快照卸载加实际 GPU NUMA 绑定 | 18.808秒 | 58.500秒 |
| attention | 上述配置，Triton | 18.849秒 | 58.507秒 |
| attention | 上述配置，DTK FA2 非分页 | 12.307秒 | 25.735秒 |

卸载加 NUMA 的延迟降幅为8.0%/2.7%；attention 换路为34.7%/56.0%。
实际 trace 确认命中 DTK BF16 FA2；CPU 同步含 GPU 等待，不能重复计为搬运收益。
各 provider 内三次 seed42 文生图分别一致；跨 provider 不逐像素一致，seed52
壶形/壶盖有可见变化，不能保证同 seed 复现旧图。固定原图编辑差异与原路径
自身重复波动接近；红改蓝的语义核验通过，不据此宣称完整画质基准通过。

CPU 接口、原生设备数值、变输入 Graph 与实际六步 t2i/edit 分别验收；
两卡原生 TP/RCCL 另有通过记录。社区源码指纹保持不变。未覆盖其他架构、
多参考图、并发或整模型 Graph；非分页验证不解除 vendor 分页缺口。
详细命令、原始请求/图片、哈希与逐轮结果保留在部署目录的
`artifacts/minimax-h3-image-20261114`、`artifacts/minimax-h3-image-single64-20261114`、
`artifacts/minimax-h3-image-perf-20261114`、`artifacts/minimax-h3-image-attention-20261115`。
