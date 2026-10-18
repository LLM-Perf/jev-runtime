# 按严格成功请求数量验收热切换

此前 hot-switch 测试完成 1,000 次切换后便停止流量，成功请求数只有几百个。
新的默认门槛为每个引擎至少 10,000 个严格成功请求，同时至少完成 1,000 次切换。
控制器持续切换直到两项都满足；任意请求失败都会使整轮失败，不通过重试补足。
最后仍会等待所有已提交请求返回，因此请求数可能略高于最低值。

每个请求包含一个问题，按 choice、boolean、score、rank 循环，输入附带唯一的
case number。它验证合成输入的接口与热切换正确性，不验证业务答案准确率，
不作为模型覆盖、独占 GPU 性能或 24 小时 soak 的替代。

严格成功需要 HTTP 成功、DecisionResponse schema、request ID、引擎身份、
问题类型及候选项、分数/概率语义、计数与 token 用量全部合法。显式 tie=first
测试 bundle 要求 answered。最终按完整 activation 记录核对 generation、bundle
及 digest；缺失或错误的版本映射不能计入严格成功。每次请求均保留，失败或
未核实的请求单独计数。正常通过要求无失败、无混合版本、无未核实请求。

```sh
python tests/integration/run_native_validation.py \
  --run-dir /root/jev-runtime/runs/NEW-UNIQUE-RUN \
  --engine vllm --model-path /root/jev-runtime/models/SmolLM2-1.7B-Instruct \
  --gpus 7 --port 18795 --memory-fraction 0.07 \
  --minimum-requests 10000 --switches 1000 --traffic-concurrency 4 \
  --traffic-timeout 1200 --check-timeout 1500 \
  --source-commit FULL_SHA --runtime-source-commit FULL_SHA
```

先按既有 DSW 流程检查资源、进程归属和端口，并使用对应引擎的独立 Python 环境。
SGLang 与 vLLM 的显存参数含义不同，不直接复用示例中的 fraction。
请求数、并发、时限和切换数都必须为正。外层 check timeout 至少比 traffic timeout
多 180 秒，给准备及已提交请求的 drain 留出时间。

`traffic-responses.jsonl.gz` 保存原始配置、所有请求/响应、activation 和汇总。
`traffic-progress.json` 保存运行进度；运行中的 response_validated_requests 只完成
响应校验，最终 generation 审计后才得到 strict_success_requests。
`contract.json` 引用 gzip 文件的 SHA256。已有四类型、原生取分对照、单位置
参考和按进程身份清理的检查继续执行。

## DSW 扩量结果，2026-09-30

最终源码为 `9da181342326793f6edddbd3f3ada4b0f7dbb320`。两引擎同时运行，
沿用隔离环境：vLLM 0.30.0+cu129 / SGLang 0.5.19，L20Z #7/#6，
SmolLM2-1.7B-Instruct `31b70e2e869a7173562077fd711b654946d38674`，BF16、
TP=1、API worker=1、eager。每端 HTTP 并发为 4，使用默认 30 秒 canary
间隔和 90 秒有效期，未增加有效期或关闭健康检查。

| 最终独立测试轮次 | vLLM | SGLang |
|---|---:|---:|
| 发出的请求 | 10,004 | 10,006 |
| 严格成功请求 | 10,004 | 10,006 |
| 失败 / 混合版本 / 未核实 / 重试 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 热切换次数 | 15,276 | 16,451 |
| choice / boolean | 2,501 / 2,501 | 2,502 / 2,502 |
| score / rank | 2,501 / 2,501 | 2,501 / 2,501 |
| joint / independent bundle 请求 | 4,854 / 5,150 | 4,516 / 5,490 |
| 流量阶段时长 | 132.522 秒 | 145.244 秒 |
| 独立 CPU 单位置最大绝对误差 | 0.107434750 | 0.000161231 |

单位置数值阈值仍为原来的 0.15。每个引擎的 request ID 和带编号输入均唯一；
两个引擎使用同一组输入构造方法，不把它们声称为 20,010 个跨引擎独立语义样本。
每条响应均通过 schema、usage、候选/类型/概率校验，并按全部 activation
记录重新核对 generation/bundle/digest。数量稍高于 10,000 来自停止时在途请求的
自然完成，不含重试补数。上述时间和日志中的延迟仅描述这轮共享机器功能测试。

证据与逐条复核结果：

- [vLLM 逐请求复核](../evidence/dsw/strict-traffic-9da1813/runs/vllm-strict-traffic-9da1813/recount.json)
- [SGLang 逐请求复核](../evidence/dsw/strict-traffic-9da1813/runs/sglang-strict-traffic-9da1813/recount.json)
- [双引擎 campaign](../evidence/dsw/strict-traffic-9da1813/strict-traffic-9da1813/campaign.json)
- [全量证据清单](../evidence/dsw/strict-traffic-9da1813/manifest.json)

## 扩量发现的问题及修复

首轮源码 `4ee78cab97c62d3ae03c5486e628af5bee797a28` 保留为失败，不能与最终
通过轮次混为一个 100% 成功的实验：

| 首轮 | 严格成功 / 尝试 | 失败原因 |
|---|---:|---|
| vLLM | 6,290 / 6,291 | 一次 HTTP 503 `engine_unavailable`，发生在约 90 秒 |
| SGLang | 0 / 4 | 新 harness 错按 vLLM 要求 completion tokens 大于零；四个响应原文保留 |

vLLM 暴露了实际的健康检查缺陷：旧代码只探测某次轮询瞬间的 active version，
热切换可能让另一版本在每次轮询时都处于 inactive，最终 canary 过期。现在会
探测当前 worker 已准备的 READY/ACTIVE/DRAINING 版本，并优先处理 active
版本。监控只能取得 revalidation lease，不能将并发退休或卸载的版本重新准备。
不修改过期阈值，不让业务请求绕过 canary。

SGLang 的原生 selected-ID scoring 使用 `max_new_tokens=0`。harness 现按该
合同要求 completion tokens **恰好为零**；vLLM 仍要求每条 scoring sequence
恰好一个 completion token。增加双引擎回归，非法用量继续判失败。

原始失败尝试的响应、日志和清理记录与最终通过结果一起保存。导出脚本首次因
keys.json 的嵌套租户字段报错，修正为递归扫描后用新导出目录重试；测试结果
没有重写。日志压缩前及 gzip 逐请求记录解压后都扫描实际 API/admin/tenant
密钥，keys.json 不导出。

## 清理与验证范围

四个原生测试进程组全部退出，四个 registry 的六类待处理工作计数全部为零。
GPU #6/#7 可用显存恢复到 10,544 / 11,744 MiB。原有服务的进程身份和命令行
hash 在每轮前后相同。这里确认所属进程组退出；未采集原生引擎退出码。
SGLang 两轮均保留两处 SystemExit/CancelledError traceback 和一处 NCCL
destroy_process_group 警告；没有把退出日志宣称为无警告。vLLM 两轮未出现
traceback，四轮均未发现 leaked semaphore。

本地 590 项 CPU 测试、50 项定向回归、Ruff lint/format 和四个 source-matched
wheel 构建通过。最初受限沙箱中 24 个启动器测试无法绑定回环端口，允许 socket
后完整重跑通过；没有修改断言来跳过这些测试。DSW 运行源码，不将 wheel 构建
等同于安装后的 GPU 验证。
最终源码还在两个实际引擎 Python 环境中各通过 56 项 CPU 回归。

离线核对 87 个证据文件、两个提交的 158+159 个源码文件，以及全部最终请求：

```sh
.venv/bin/python evidence/harnesses/verify_strict_traffic_9da1813.py
```

本轮只扩大请求覆盖并修复相应缺陷。TokenSpeed 真实 GPU 测试、广泛模型覆盖、
业务质量、受控性能和 24 小时 soak 等既有门槛保持原状态。
