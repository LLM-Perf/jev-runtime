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
