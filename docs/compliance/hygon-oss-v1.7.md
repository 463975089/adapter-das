# HYGON 开源合规扫描整改记录（hygon-open-source-v1.7）

- 扫描对象：main @ `66579fdc4bfd`（2026-10-31 扫描，报告
  `adapter-das-main-66579fdc4bfd-合规-20261031-114430.md`）
- 整改分支：`compliance/hygon-oss-v1.7`（2026-11-04，基于整改时点 main）
- 复扫口径：以本分支合入后的 main 新 commit 为准，文件数会多于报告
  （66579fd 之后 main 新增 ~70 个源码文件已一并补头）。

## 整改映射

| 报告阻断项 | 处置 | 落点 |
|---|---|---|
| 363 文件缺 HYGON 头 | `tools/compliance/add_hygon_headers.py` 批量补 H1（py/sh 用 `#`，h 用 `//`，shebang/编码行后插入，幂等）；本分支共补 430 个（363 报告项 + 扫描后新增） | 全仓 |
| §4 17 个来源待确认 | 逐一定性：9 个 fla 第三方 + `utils.py`/`fused_sigmoid_gating_recurrent.py` = 保留原声明、登记不加水头；`tests/vendor/` 3 个 = HYGON 原创，已加 H1；`engram_gate_triton.py`/`slimquant.py` = H2（HYGON Copyright + Modified by + 登记）；`aiter_moe.py` = HYGON 原创配方重实现，已加 H1 + 登记 | [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) |
| THIRD_PARTY_NOTICES.md 缺失 | 新建：FLA(MIT)、vLLM(Apache-2.0)、causal-conv1d(BSD-3-Clause)外部项 + sglang-das(`364fabf`/`7b91060`)、lmslim 内部血缘，全部带固定 commit/许可证/本地路径/HYGON 修改 | 根目录 |
| README 缺许可证说明与第三方入口 | 顶部新增"许可证与第三方声明"段（LICENSE + THIRD_PARTY_NOTICES.md 链接） | [README.md](../../README.md) |
| REDUCTION_PATCHES.md:131 品牌身份 | 两处品牌词替换为中性表述（运维凭据归属方、厂商监控栈；原词见报告 §3），全仓 grep 归零 | REDUCTION_PATCHES.md |
| 防回潮 | governance §5.0 新增"源码文件头义务"条款（新增源码必须带头或登记来源） | [governance.md §5.0](../governance.md) |

## 遗留项（需团队/管理渠道，不在本分支处理）

1. **git 历史邮箱品牌身份（阻断项）**：main 上作者/提交者邮箱使用公司
   禁用域名（见报告 §3）的 commit 共 **21 个**（maxiao1 ×12、liph ×9；
   报告时点 13 个，扫后 L1.5 线又入 8 个）。
   报告要求"对外发布前重写或移除"。改写共享 main 历史影响所有分支/
   fork，需团队统一决策，可选：
   - a) 发布前在导出快照（release 分支/tag）上 `git filter-repo` 重写邮箱，
     仓内 main 不动（推荐：对外归档与内部历史解耦）；
   - b) 全仓历史重写 + force-push + 各分支重建（影响最大，需全员协调）。
2. **commit message 规范（建议项，不阻断）**：报告 §6 建议后续 commit 用
   `type(scope): subject` 并按类型补性能/原因/影响说明；已在新提交中执行
   （如本分支 `chore(compliance): ...`）。
3. **FLA 精确 commit 锚定**：fla_hcu 内 9 个文件的 `Adapted from` 链接指
   上游 main 快照，未带 commit hash；当前以 sglang-das `364fabf` 为传递
   锚点登记。若复扫要求上游原生 commit，需对照引入时点逐文件 diff 锚定。

## 验证

- `python -m compileall hcu_adapter sgl_kernel tools tests ci sitecustomize.py` 0 错；
- 42 个 .sh `bash -n` 全过（shebang 保持在第 1 行）；
- 13 个第三方/移植文件 diff 为零（原声明未动）；文档级 md/JSON/toml 未触碰。
