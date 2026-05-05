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

## 2. 固定配置

`configs/` 只保留三份配置，避免本地评估入口发散。

| 配置 | 任务范围 | 使用场景 |
| --- | --- | --- |
| `configs/selected.yaml` | `run.task_ids` 中列出的任务 | 快速验证、重点任务复跑、调试 inspector / trace |
| `configs/full.yaml` | 全部公开任务 | 阶段验收和主分比较 |
| `configs/docker.yaml` | 平台挂载的 `/input` 全部任务 | Docker 提交镜像默认入口 |

## 3. 固定命令

```powershell
uv run pytest

uv run dabench run-benchmark --config configs/selected.yaml
uv run dabench score-run <selected_run_id>

uv run dabench run-benchmark --config configs/full.yaml
uv run dabench score-run <full_run_id>

uv run dabench compare-runs artifacts/standard/baseline <full_run_id>
```

如果某阶段只改了特定能力，可以先修改 `configs/selected.yaml` 的 `run.task_ids` 做小范围复跑；合并前仍应跑 `configs/full.yaml`。

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

Selected:
  primary_proxy_score:
  mean_recall:
  mean_redundancy_rate:
  prediction_task_count:
  failure_breakdown:
  重点任务变化：

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
