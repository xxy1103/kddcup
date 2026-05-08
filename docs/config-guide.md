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

### 3.1 节点开关

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_global_exploration_llm` | 布尔 | `true` | `true`：让 LLM 把数据目录整理成易读的 markdown profile。`false`：跳过 LLM，直接把 raw catalog JSON 传给下游 |
| `enable_problem_grounding` | 布尔 | `true` | `true`：运行完整的问题锚定流程（4 阶段 guided loop）。`false`：跳过，直接将上一阶段的数据注入为消息给主 Agent |

### 3.2 开关组合效果

| `enable_data_inspector` | `enable_global_exploration_llm` | `enable_problem_grounding` | 实际效果 |
|---|---|---|---|
| `false` | 任意 | 任意 | 两个节点都不执行，Agent 直接看原始数据 |
| `true` | `true` | `true` | **完整流水线**（默认）：LLM 生成 profile -> 问题锚定 -> 主 Agent 收到 handoff |
| `true` | `false` | `true` | 省钱模式：raw JSON -> 问题锚定处理 |
| `true` | `true` | `false` | 快速模式：LLM profile -> 直接给主 Agent |
| `true` | `false` | `false` | 最省钱：raw JSON -> 直接给主 Agent |

### 3.3 `enable_data_inspector: false` 时的流程

图的拓扑不变，但 `global_data_exploration` 和 `problem_grounding` 两个节点都直接返回 `{}`（空操作）：

```
START -> init_state -> global_data_exploration -> receive_problem -> problem_grounding -> model_step -> [ReAct循环] -> finalize
                           |                                    |                        |
                           v                                    v                        v
                       return {}                            return {}                 正常执行
```

- `global_data_profile` 保持 `None`（不构建 catalog，不调 LLM）
- `inspector` 保持 `None`（不运行 DataUnderstandingAgent，不注入 handoff）
- Agent 的 `messages` 里只有 system prompt + 任务问题，靠自己用工具探索数据

### 3.4 问题锚定参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `inject_summary_to_agent` | 布尔 | `true` | 是否将 data understanding 的 handoff 摘要注入到主 Agent 的对话中 |
| `max_agent_steps` | 整数 | 5 | 问题锚定阶段（problem_grounding）最大 LLM 调用步数 |
| `max_phase_retries` | 整数 | 1 | 每个锚定阶段（grounding/fabric/contract）验证失败后的最大重试次数 |
| `profile_guided_fast_path` | 布尔 | `true` | 当 global profile 可用时，跳过 LLM overview 阶段；对单表/单 join 路径的任务直接用规则推导 join，不调 LLM |
| `enable_semantic_tools` | 布尔 | `true` | 是否允许锚定阶段使用语义工具（search_semantic_index、lookup_knowledge 等） |
| `include_inspector_trace` | 布尔 | `true` | 是否在结果中记录 inspector 的详细步骤日志 |
| `context_bundle_limit` | 整数 | 6 | 构建上下文时最多包含的候选字段数 |

### 3.5 数据采样参数 `sample_budget`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `catalog_sample_rows` | 整数 | 5 | 每个表/文件采样的行数（给 LLM 看的数据预览） |
| `catalog_top_distinct_values` | 整数 | 50 | 每个字段展示的去重值数量，取频率最高的前 N 个。`cardinality` 字段会报告真实的去重总数 |
| `max_doc_chars` | 整数 | 2000 | 文档类文件（knowledge.md）在 catalog 中预览的最大字符数（全文仍会传给 LLM） |
| `max_json_chars` | 整数 | 4000 | JSON 文件在 catalog 中预览的最大字符数 |

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

### 场景 1：完整流水线（推荐，数据质量最高）
```yaml
agent:
  enable_data_inspector: true
data_inspector:
  enable_global_exploration_llm: true
  enable_problem_grounding: true
```

### 场景 2：省钱模式（跳过 global profiling 的 LLM 调用）
```yaml
data_inspector:
  enable_global_exploration_llm: false
  enable_problem_grounding: true
```

### 场景 3：快速模式（跳过问题锚定，LLM profile 直接给 Agent）
```yaml
data_inspector:
  enable_global_exploration_llm: true
  enable_problem_grounding: false
```

### 场景 4：最省钱（两个 LLM 阶段都跳过）
```yaml
data_inspector:
  enable_global_exploration_llm: false
  enable_problem_grounding: false
```

### 场景 5：完全不用 data inspector（裸 Agent）
```yaml
agent:
  enable_data_inspector: false
```
