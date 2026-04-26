# 增量评估协议

本文固定后续 Data Agent 增量开发的本地评估方式。目标是让每次改动都能和同一批任务、同一组参数、同一套指标比较，避免只凭单个任务 trace 判断收益。

## 1. 固定基线

长期只读锚点：

| 来源 | run_id | 任务数 | 有预测任务 | Primary λ=0.1 | Mean Recall | Mean Redundancy | 主要失败 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `artifacts/standard/baseline` | `20260424T011929Z` | 50 | 40 | 0.500000 | 0.500000 | 0.300000 | 10 个 max_steps 未提交 |

基线运行参数来自该目录的 `summary.json`：

| 参数 | 值 |
| --- | ---: |
| `agent.max_steps` | 32 |
| `agent.temperature` | 0.0 |
| `run.max_workers` | 4 |
| `run.task_timeout_seconds` | 600 |

后续每个增量应先用相同参数跑出新的 `baseline_freeze` 或阶段基线，再和历史锚点及上一阶段结果比较。

## 2. 固定任务切片

版本化模板使用 `configs/eval_*.example.yaml`。本地运行时可以复制为 `configs/eval_*.yaml` 并填入模型参数；`configs/eval_*.yaml` 按仓库约定作为本地配置忽略，不应提交凭据或个人参数。

| 切片 | 配置 | 任务 | 关注问题 |
| --- | --- | --- | --- |
| Smoke | `configs/eval_smoke.example.yaml` | `task_11, task_19, task_26` | 快速确认配置、模型调用、工具、落盘和评分链路可用 |
| Contract | `configs/eval_contract.example.yaml` | `task_25, task_80, task_89, task_163, task_180, task_379` | 题意、字段归属、过滤条件、聚合口径、最终粒度是否漂移 |
| Redundancy | `configs/eval_redundancy.example.yaml` | `task_24, task_38, task_74, task_287, task_292, task_303, task_330` | 最终答案是否夹带冗余列、辅助字段或中间明细 |
| Long | `configs/eval_long.example.yaml` | `task_173, task_344, task_352, task_396, task_418` | 长链路任务是否接近 max_steps、超时或不提交 |
| Full public | `configs/eval_full_public.example.yaml` | 全部 50 个公开任务 | 判断阶段改动是否值得保留 |

## 3. 固定命令

以下命令假设已经根据 `.example.yaml` 创建了对应的本地 `configs/eval_*.yaml`。

```powershell
uv run pytest

uv run dabench run-selected-tasks --config configs/eval_smoke.yaml
uv run dabench score-run <smoke_run_id>

uv run dabench run-selected-tasks --config configs/eval_contract.yaml
uv run dabench score-run <contract_run_id>

uv run dabench run-selected-tasks --config configs/eval_redundancy.yaml
uv run dabench score-run <redundancy_run_id>

uv run dabench run-selected-tasks --config configs/eval_long.yaml
uv run dabench score-run <long_run_id>

uv run dabench run-benchmark --config configs/eval_full_public.yaml
uv run dabench score-run <full_run_id>

uv run dabench compare-runs artifacts/standard/baseline <full_run_id>
```

如果某阶段只改了特定能力，可以先跑对应切片，但合并前必须跑 Full public。

## 4. 记录指标

每次阶段评估记录以下指标：

- `primary_proxy_score`
- `mean_recall`
- `mean_redundancy_rate`
- `prediction_task_count`
- `failure_breakdown`
- `mean_model_step_count`
- `max_model_step_count`
- `p95_e2e_elapsed_seconds`
- `max_e2e_elapsed_seconds`

## 5. 阶段记录模板

```text
增量编号：
代码分支 / commit：
开启的配置：
评估 run_id：

Smoke:
  primary_proxy_score:
  mean_recall:
  mean_redundancy_rate:
  prediction_task_count:
  failure_breakdown:

Contract:
  primary_proxy_score:
  重点任务变化：

Redundancy:
  mean_redundancy_rate:
  额外列任务变化：

Long:
  未提交数:
  timeout 数:
  max_model_step_count:

Full public:
  primary_proxy_score:
  mean_recall:
  mean_redundancy_rate:
  prediction_task_count:
  p95_e2e_elapsed_seconds:

结论：
是否进入下一增量：
需要回退或继续调优的配置：
```
