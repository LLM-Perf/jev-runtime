# 多推理框架支持：实现状态与接入合同

本页扩展原有 implementation-plan.md；原计划、验收阈值和历史失败记录保持不变。
“同时支持”指多个独立引擎进程/环境同时提供同一套 Jev API，调用者通过不同
base URL 访问。不是在一个 Python 环境里安装所有 CUDA 框架，也不承诺同一 GPU
能无额外容量代价地同时驻留多个模型。

## 目前还缺什么

| 范围 | 已实现 | 尚未完成 |
|---|---|---|
| vLLM | 原生插件、完整标签取分、四种类型、bundle 切换/回退、取消、同机多 worker、受限 LoRA | 模型功能覆盖 12/20，目标至少 18/20；独立数值与性能认证、完整稳定性/故障矩阵 |
| SGLang | 原生插件和 HTTP 接入、同样的类型与生命周期能力 | 模型功能覆盖 12/20，目标至少 18/20；部分数值检查失败；原生退出仍有警告；完整性能和稳定性验收 |
| TokenSpeed | 新增独立插件包、原生 Engine 启动器、HTTP bridge、逐标签 raw readout、完整计费/配额/日志、完成响应驱动的 drain | **实验性，尚无真实 TokenSpeed 模型 GPU 服务验证**；单次 prefill 批量取分优化、广泛模型/并行配置、原生取消确认/崩溃恢复、LoRA |
| 其他框架 | 新增已安装包 entry point 接入，无需修改核心 backend 枚举 | 每个框架仍需实现真实的分数/取消合同；仅兼容 OpenAI chat API 不够 |
| 产品验收 | 版本化注册表、显式离线升级/回退、SDK、配额、校准、探活、证据工具 | 24 小时 soak、144 个受控性能用例、镜像实际构建/运行、两个业务数据验收、兼容副本故障转移、最终版本复验 |

历史 vLLM/SGLang 功能分母仍为 40 个模型/引擎组合。TokenSpeed 单独列在
`profiles/framework-support.json`，20 个 GPU 模型组合全部尚未测试，不能用 CPU
适配器测试替代。原来更广的多节点协调、VLM/高级 readout 属于后续阶段。

## 插件机制

安装在当前服务环境中的第三方包可声明：

```toml
[project.entry-points."jev_runtime.backends"]
my_engine = "my_package.backend:create_backend"
```

工厂接收关键字参数 `settings` 和 `api_key`，返回实现异步
`probe / score / cancel / close` 的 EngineAdapter。配置 `backend: my_engine`。

`jevctl backend list` 仅列出元数据，不导入引擎/CUDA。服务只导入被显式选择的
provider；同名 provider、覆盖内置框架、未安装 provider 会明确报错。
插件是可信的服务端 Python 代码，安装包本身属于管理员控制的发布操作。

`probe` 必须报告实际 precision、上下文/标签限制、取消能力等，原始标签分数
缺失时不允许启动。新增后端也执行 precision 合同检查。
`label_scoring: joint` 是一次调用取多个标签；`single` 会在调度前拆成每标签一条
原生请求，保留同一 prompt 和原来的概率语义。每条请求都有独立持久日志、配额、
取消 ID 与 token 用量；分支及扩展 token 上限按拆分后大小计算。

对于已暴露 `/plugins/jev-runtime/v1/{scoring-capabilities,scores,scores/cancel}`
的引擎，可复用 `ScoringHTTP`。它检查引擎名称、单取消 owner、分数数量/请求 ID、
raw 标志、有限非正 logprob、用量及显式取消确认。普通生成接口不可替代此合同。

## TokenSpeed 为什么需要独立实现

检查的上游提交为
[`7fa8acb1e885389825c077a6aec0326fbbbd7116`](https://github.com/lightseekorg/tokenspeed/tree/7fa8acb1e885389825c077a6aec0326fbbbd7116)。
其 InputProcessor 明确拒绝 `token_ids_logprob`、top-k 和 prompt logprob。
控制 HTTP 层在缺少实际 logprob 时还可能补 0.0 占位值。因此本插件不用该 HTTP
兼容层取分，也不把 TokenSpeed 当成 SGLang 别名。

该提交的 `triton_full.sample` 在施加 penalty/bias 前复制 logits，再返回采样 token
的原始分数。插件给每个目标标签加有限 bias，生成恰好一个 token，并检查实际
返回 token、位置、请求 ID 和完成原因；标签未被选中会失败，绝不重试到“碰巧成功”。
没有 grammar mask、temperature 变换后的伪概率或字符串生成解析。

这会执行 K 次单 token 请求，而不是一次原生 selected-logprob gather。核心会
完整计算其 K 个分支、重复 prompt 和 K 个生成 token。它是功能接入路径，**没有
高性能达标声明**。后续优化应在上游 sampler/scheduler/output 通道增加原生
selected-ID gather，再保持同一合同进行对照验证。

本版本仅允许 `triton_full + enable_output_logprobs + eager + TP/DP/PP=1`，不支持
量化、speculation、PD、在线权重更新或 managed LoRA。启动前核对 13 个上游
关键 Python 文件的 SHA256；这只绑定检查过的代码合同，不是整个运行环境/权重
的完整证明。CUDA kernel、依赖版本和模型 GPU 验证仍是独立门槛。

## TokenSpeed 使用方法（实验性）

先在**单独环境**安装上述提交及其匹配依赖，遵循上游安装方式。当前上游 NVIDIA
Dockerfile 使用 CUDA 13.0 / Torch 2.14.0；上游也提供
[实验性 CUDA 12.9 源码安装路径](https://github.com/lightseekorg/tokenspeed/blob/7fa8acb1e885389825c077a6aec0326fbbbd7116/docs/guides/hopper-cu129.md)，
但明确针对 H100/H200、Python 3.11，构建 sm90a kernel。本项目现有 DSW 是
L20Z、Python 3.12、CUDA 12.9，尚无匹配的 TokenSpeed 安装。适配其他环境需要
单独的依赖/编译/内核验证，不能直接复用历史 SGLang/vLLM 测试结论。

```sh
pip install -e '.[tokenizers]' -e packages/tokenspeed
jevctl backend list
# 通过环境/secret manager 设置 JEV_API_KEY 和 JEV_ADMIN_KEY。
jev-tokenspeed --config examples/tokenspeed.yaml \
  --engine-config examples/tokenspeed-engine.json
```

示例中的路径、revision、显存预算必须对应实际部署。只启动一个 HTTP worker；
该启动器拥有它创建的 Engine 子进程，并使用 Engine 的后台事件循环调度请求。
标准插件前缀下提供 typed/admin/raw API，可直接使用现有 Python/TypeScript SDK。
它没有新增 OpenAI chat 代理；如需与现有 TokenSpeed 服务共进程嵌入，应调用
`TokenSpeedNative` 并在 host 所有的事件循环上提交，另行验证该宿主的退出顺序。

另起 Jev gateway 时，将 `backend: tokenspeed` 和 `engine_url` 指向上述原生插件
服务；gateway 应使用另一个监听端口（如 8795）和独立的 registry_path。
配置 `engine_key_env` 对应其 JEV_API_KEY。bridge 保留同一逐标签能力标志。

上传、prepare、activate、disable、回退复用现有 bundle API。代码安装和底层
框架替换仍需要进程发布；热更新的是配置、模板、策略与校准 bundle。

取消采用“停止等待结果，等待正在执行的一个 token 自然完成”的受限策略。
上游按 ID abort 只移除 frontend 状态，没有 scheduler 完成确认，故这里不把
abort 返回成功视为 drain。保留最多 4,096 个本进程完成凭据，使 HTTP 响应丢失后
仍能确认已完成的请求；凭据保留期间拒绝复用该 ID。4.5 秒内无完成响应会保留
journal/lease；未知 ID、已淘汰凭据或进程重启后的请求也拒绝确认。
不能自动回收这类孤儿记录，不能宣称完整恢复支持。
服务停止遇到不确定 drain 会报错并保留证据。

## 验证分层

CPU 合同覆盖插件发现/冲突、四种类型和两种 readout 的拆分一致性、配额分母、
实际 completion token 统计、错误分数拒绝、取消等待/传输故障保留与配置限制。
这些 tests 使用显式测试替身。源码取分检查可以复核上游采样方法在 bias 前读取
logits 的顺序；它不是完整 TokenSpeed GPU serving 验证。

所有现有引擎仍须按最终源码复验。插件源码、CPU 合同、已构建 wheel、真实 GPU
运行、业务质量和受控性能分别报告，任何一项不能替代另一项。
