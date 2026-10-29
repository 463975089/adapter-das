# DeepInfer 宿主接入

依赖公共计算与通信/辅助接口两批代码。参照 DeepInfer
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，通过
`HCU_ADAPTER_HOST=deepinfer` 选择宿主。worker hook 在设备可见性设置后安装 provider。

`hosts.deepinfer` 负责扩展路径、Graph 与后端选择；共享 registry 记录实际提供者，
计算实现不进入 bridge。SGLang 专属 auto 探测仅用于原宿主，通用 auto 规则继续有效。
原有 SGLang 探测函数及 try/except 保留原位置。旧配置/注册导入路径仅作转接。

L0 只扩展对子目录和组合 catalog 的识别，不增加 UI 快照或公共内核的新检查规则。
本批不包含模型 plugin，不改变 channel-FP8 加载或新增缓存执行方案。
节点精度、Graph 与性能验收尚待完成。部署脚本与构建工具由下一批提供。
