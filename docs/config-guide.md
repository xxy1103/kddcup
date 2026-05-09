# 配置文件说明

配置文件位于 `configs/` 目录，使用 YAML 格式。运行命令时通过 `-c` 指定：

```bash
uv run python -m data_agent_baseline run-task -c configs/easy.yaml
```

---

## 一、`dataset` — 数据集路径

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `root_path` | 路径 | `data/public/input` | 公开数据集的根目录，任务数据从此目录下按 `task_id` 查找 |

---

## 二、`agent` — 主 Agent 配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | 字符串 | `gpt-4.1-mini` | 模型名称。如果同时设置了 `model_env`，则从环境变量读取，忽略此值 |
| `model_env` | 字符串 | 无 | 从环境变量读取模型名的变量名（仅 docker.yaml 使用） |
| `api_base` | URL | `https://api.openai.com/v1` | API 地址 |
| `api_base_env` | 字符串 | 无 | 从环境变量读取 API 地址的变量名（仅 docker.yaml 使用） |
| `api_key` | 字符串 | 空 | API Key。留空则从 `api_key_env` 指定的环境变量读取 |
| `api_key_env` | 字符串 | 无 | 环境变量名。程序会先读环境变量，读不到再读 `.env` 文件 |
| `max_steps` | 整数 | 16 | 主 Agent 最大 ReAct 轮数（model_step -> tool_step 循环次数） |
| `temperature` | 浮点 | 0.0 | 模型温度，0 表示确定性输出 |
| `enable_data_inspector` | 布尔 | `false` | **总开关**。必须设为 `true`，下面的 `data_inspector` 配置才会生效 |
| `model_request_timeout_seconds` | 浮点 | 120 | 单次模型请求超时秒数 |

---

## 三、`data_inspector` — 数据探查配置

### 3.1 `enable_data_inspector: false` 时的流程

`global_data_exploration` 节点直接返回 `{}`（空操作），Agent 的 `messages` 里只有 system prompt + 任务问题，靠自己用工具探索数据。

流程：`START -> init_state -> global_data_exploration(no-op) -> receive_problem -> model_step -> [ReAct循环] -> finalize`

- `global_data_profile` 保持 `None`（不构建 catalog，不调 LLM）
- `inspector` 保持 `None`
- `receive_problem` 不注入 catalog 消息

### 3.2 数据采样参数 `sample_budget`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `catalog_top_distinct_values` | 整数 | 50 | 每个字段展示的去重值数量，取频率最高的前 N 个。`cardinality` 字段会报告真实的去重总数 |
| `max_doc_chars` | 整数 | 2000 | 文档类文件（knowledge.md）在 catalog 中预览的最大字符数 |

---

## 四、`run` — 运行配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `output_dir` | 路径 | `artifacts/runs` | 运行产物输出目录 |
| `log_dir` | 路径 | 无 | 日志目录（仅 docker 使用） |
| `output_layout` | 字符串 | `run_dir` | 产物目录结构：`run_dir`（每 run 一个子目录）或 `flat`（平铺） |
| `run_id` | 字符串 | 空 | 运行 ID，留空自动生成 |
| `max_workers` | 整数 | 4 | 并行执行任务的最大线程数 |
| `task_timeout_seconds` | 整数 | 600 | 单个任务超时秒数 |
| `task_ids` | 列表 | 无 | 要执行的任务 ID 列表。不填则执行全部 |

---

## 五、典型使用场景

### 场景 1：开启 Data Inspector（推荐）
```yaml
agent:
  enable_data_inspector: true
```

### 场景 2：完全不用 data inspector（裸 Agent）
```yaml
agent:
  enable_data_inspector: false
```
