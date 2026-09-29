# 多框架接入验证与剩余门槛

本轮实现可扩展 backend provider、显式 raw-score HTTP bridge 和实验性 TokenSpeed
原生 Engine 插件。vLLM、SGLang 的真实 GPU 回归通过；TokenSpeed 仅完成源码合同
检查和 CPU 测试，尚未执行真实模型 GPU 服务。原计划及其验收阈值保持不变。

## 源码与证据边界

| 检查点 | 范围 | 结果 |
|---|---|---|
| `62340feddb4d73274c0902d68cead1012da94624` | 多框架实现，首次四包构建，双引擎 GPU 回归 | 569 项本地测试；每个实际引擎 Python 环境 149 项 CPU 合同；最终 GPU 回归通过 |
| `55674ecbe63850dc9c46da14893ffcad59d2506f` | TokenSpeed 有界完成凭据及并发取消回归 | 571 项本地测试；每个实际引擎 Python 环境 151 项 CPU 合同；四包构建通过 |

后一提交的 `src/`、`packages/vllm/`、`packages/sglang/` 与 GPU 检查点逐文件相同。
其 TokenSpeed 变更没有新增 GPU 认证。上述 DSW 服务运行源码；本地 source-matched
wheel 构建不能替代安装后完整 GPU 回归。TypeScript 未变更，本轮没有重新测试。

本地记录为 [62340fe](../evidence/local-check-62340fe.json) 和
[55674ec](../evidence/local-check-55674ec.json)。最新 Ruff lint/format 通过。

## vLLM 与 SGLang 同时运行

两个引擎在独立 Python 环境、不同 GPU 和端口同时运行。使用已存在的
SmolLM2-1.7B-Instruct，revision `31b70e2e869a7173562077fd711b654946d38674`，
BF16、TP=1、API worker=1、eager、最大上下文 2,048。DSW 的现有其他服务未停止
或修改；这是共享机器上的功能检查，不是独占资源的性能实验。

| 项目 | vLLM | SGLang |
|---|---|---|
| 引擎版本 | 0.30.0+cu129 | 0.5.19 |
| Transformers | 5.17.0 | 5.12.1 |
| GPU / 端口 | L20Z #7 / 18795 | L20Z #6 / 18794 |
| 四种类型、native chat 共存、认证及 bundle 生命周期 | 通过 | 通过 |
| 原生接口与插件标签分数最大误差 | 0 | 0 |
| 流量中的 bundle 切换 | 1,000 次 | 1,000 次 |
| 严格成功的并发流量请求 | 491 | 588 |
| 混合 bundle 版本响应 | 0 | 0 |
| 独立 CPU 单位置参考最大绝对误差 | 0.1074347496 | 0.0866205692 |
| 单位置阈值 | 原有 0.15，通过 | 原有 0.15，通过 |
| Jev 请求清空、所属进程组退出 | 通过 | 通过 |

这两个单位置数值结果不构成整个模型、所有位置或其他 precision/parallelism
配置的数值认证，也不覆盖历史失败的 BF16 配置。

最终记录见 [双引擎 campaign](../evidence/dsw/multi-engine-62340fe/multi-engine-62340fe-r3/campaign.json)。
两次 native 尝试共四个进程组均已退出，四个 registry 的六类待处理工作计数均为零。
GPU #6/#7 的可用显存分别回到本轮启动前的 10,544 / 11,744 MiB。
最终 campaign 的历史进程审计只覆盖 `runs/` 下 178 条记录，不扩大为所有历史目录。

## 失败与退出警告保留

1. 首次 orchestration 在启动引擎之前发生 GPU 显存字符串与整数比较的 TypeError。
   仅修正 harness 的整数转换。原始脚本和部分 campaign 记录保留。
2. 第二次两引擎的四类型与各 1,000 次切换通过，但 postcheck 发现导入源码路径
   未包含声明的完整 commit SHA，整次验证判为失败。重新解包到完整 SHA 路径，
   使用相同产品源码运行第三次；没有放宽检查。失败记录保留。
3. 第二次 vLLM 退出出现一次 EngineDeadError traceback；最终重复没有该 traceback，
   不能据此宣称退出竞态已修复。SGLang 两次均保留 SystemExit/CancelledError
   traceback 与 NCCL destroy_process_group 警告。所有所属进程组均退出，未强杀，
   未发现 leaked semaphore；未捕获原生引擎退出码，不能宣称所有引擎 exit 0。

## TokenSpeed 验证到了哪一层

绑定 upstream `7fa8acb1e885389825c077a6aec0326fbbbd7116` 的 13 个关键 Python
文件。原生 selected-ID logprob 接口在该版本被拒绝；实验性实现使用
`triton_full` 保留的 bias 前 logits，每个标签提交一个生成 token 的请求。
实际请求数量、重复 prompt、completion tokens 均计入配额、持久日志及 usage。

上游采样方法的六个源码级案例覆盖 FP16/BF16/FP32 和 batch=1/3，分数误差为零。
它执行的是取出的 Python 方法体，底层算子使用 Torch CPU 替身，**没有执行真实
TokenSpeed Engine、CUDA kernel 或模型**。
见 [源码取分记录](../evidence/dsw/multi-engine-62340fe/multi-engine-62340fe-r2/tokenspeed-source-readout.json)。

本地及 DSW CPU 合同覆盖插件发现、四种类型、joint/independent readout 拆分、
配额/usage、畸形分数、原生事件循环调度替身、传输故障、取消及生命周期。
最后新增最多 4,096 条的完成凭据和 score/cancel 并发回归，处理 native 完成后
HTTP 响应丢失。凭据淘汰或进程重启后仍保留未知请求日志，**没有实现持久化的
TokenSpeed 崩溃恢复**。最新 [DSW CPU 记录](../evidence/dsw/multi-engine-55674ec/contracts.json)
在 vLLM 与 SGLang 环境分别通过 151 项；它们不是 TokenSpeed 原生环境。

上游默认容器使用 CUDA 13/Torch 2.14，也提供针对 H100/H200、Python 3.11、
sm90a 的实验性 CUDA 12.9 路径。当前 DSW L20Z/Python 3.12/CUDA 12.9 没有
匹配的 TokenSpeed 安装，未修改现有服务依赖或驱动。具体限制和启动示例见
[多框架操作说明](multi-engine.md)。

## 尚不能验收的部分

| 优先级 | 工作 | 当前状态与完成条件 |
|---|---|---|
| P0 | TokenSpeed 原生模型 GPU 验证 | 尚未运行；先建立匹配的隔离环境，再执行 typed/raw parity、1,000 次切换、独立数值检查、drain/重启及资源清理 |
| P0 | TokenSpeed 高效取分 | 当前 K 个标签需要 K 条原生请求；需实现单次 prefill selected-ID gather，跨 sampler/scheduler/output 传递，并做性能与原始分数对照 |
| P0 | TokenSpeed 请求恢复 | 仅本进程有界完成凭据；还缺 native abort 完成确认、进程故障后的可验证完成状态及孤儿日志安全回收 |
| P0 | 广泛模型覆盖 | 原 vLLM/SGLang 各 12/20，目标各至少 18/20；TokenSpeed 新增 20 个组合全部未做 GPU 验证 |
| P0 | 数值与业务质量 | 历史较宽 BF16/量化/并行配置仍有失败；两个已批准业务数据集及错误准则尚未齐备 |
| P0 | 高性能验收 | 144 个受控性能用例未运行；历史短程 matched plugin/native 约 81–86%，未达到 90% 目标；本轮功能回归没有新增性能达标证据 |
| P0 | 生产发布 | 24 小时 soak、实际镜像构建/运行及非 root/PID1 行为、最终发布版本复验尚未完成 |
| P1 | 扩大执行能力 | TokenSpeed LoRA、量化、TP/DP/PP、多 worker、PD、speculation 尚未实现/认证；已有两引擎 LoRA 只在窄配置认证 |
| P1 | 故障与其他引擎 | 兼容副本选择/故障转移及更多故障矩阵待完成；其他引擎可注册 provider，但仍需各自实现真实分数与取消合同 |

“热插拔”已覆盖 bundle 的 prepare/activate/disable/回退；安装新的 Python/CUDA
插件或更换底层引擎仍需发布进程，不是运行中卸载 CUDA 框架。
原 vLLM/SGLang 的 24/40 功能分母保持不变，TokenSpeed 扩展矩阵单独记录。
以上所有未完成事项继续阻止完整产品/高性能发布声明。

## 离线复核

```sh
.venv/bin/python evidence/harnesses/verify_multi_engine_62340fe.py
```

验证器检查 62340fe 的 82 份导出文件 SHA256、153 个 Git 源文件、保留的失败、
最终双引擎结果、四组清理及 TokenSpeed 非 GPU 边界；同时检查 55674ec 的四份
CPU 证据及两检查点核心/原有插件源码一致性。manifest 不包含自身；测试密钥
不在导出包中，原始日志经过实际测试 API/admin 密钥精确值扫描。
