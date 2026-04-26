# Data Agent 软件项目设计书

版本：v0.1
适用阶段：MVP 快速落地版
项目定位：面向数据分析 / 数据科学任务的轻量级 Data Agent 系统
推荐技术栈：Python + LangGraph + DuckDB + pandas / Polars + LiteLLM + Docker

---

# 1. 项目背景

随着大语言模型在语义理解、任务规划、代码生成和工具调用方面能力增强，传统数据分析系统可以从“人工编排流程”逐步演进为“由 Agent 自动理解任务、拆解任务、选择工具、执行分析、检查结果并修复错误”的智能数据系统。

本项目旨在实现一个轻量级 Data Agent 软件系统。系统接收用户的数据相关任务，自动读取输入数据，理解数据结构，将复杂任务拆解为若干子任务，选择合适的执行 Agent 或工具完成每个子任务，并在每一步执行后检查中间结果。当执行失败或结果不满足要求时，系统可以进行局部重试、任务改写、切换 Agent 或局部重规划，最终输出可验证的结果。

本项目不追求一次性完整复刻研究论文中的复杂多 Agent 编排系统，而是优先实现一个工程上可运行、可调试、可扩展的 MVP 版本。

---

# 2. 项目目标

## 2.1 总体目标

构建一个能够自动完成数据任务的 Data Agent 系统，具备以下核心能力：

1. 自动识别输入数据文件和数据结构。
2. 自动将复杂任务拆解为多个可执行子任务。
3. 根据子任务需求选择合适的 Agent 或工具。
4. 顺序执行或后续扩展为依赖图执行。
5. 保存每一步中间结果，形成简化版 Data Catalog。
6. 对中间结果和最终结果进行验证。
7. 在失败时进行局部修复、重试或切换 Agent。
8. 支持 Docker 化部署，便于比赛提交或服务器运行。

## 2.2 MVP 阶段目标

MVP 阶段只实现最关键的主流程：

```text
用户任务
  ↓
数据检查 Data Inspector
  ↓
任务规划 Planner
  ↓
Agent 选择 Selector
  ↓
子任务执行 Executor
  ↓
结果验证 Verifier
  ↓
失败修复 Refiner
  ↓
最终答案 Final Answer
```

MVP 阶段暂不实现复杂功能，例如：

- 不做复杂 embedding 微调。
- 不做完整 Agent benchmark selection。
- 不做复杂 DAG 并行执行。
- 不做完整 A2A / MCP 协议。
- 不做长期强化学习或自动持续训练。
- 不做大型分布式调度。

---

# 3. 需求分析

## 3.1 功能性需求

### FR-1：数据输入识别

系统应支持读取输入目录中的常见数据文件，包括：

- CSV
- Excel
- JSON
- Parquet
- SQLite 数据库
- TXT / Markdown
- PDF，后续扩展

系统需要生成数据摘要，包括：

- 文件名
- 文件类型
- 表结构
- 字段名
- 字段类型
- 样例数据
- 缺失值情况
- 行数和列数
- 可能的主键 / 外键 / 关联字段

### FR-2：任务拆解

系统应将用户原始任务拆解为 3 到 6 个子任务。每个子任务需要包含：

- 子任务 ID
- 子任务目标
- 输入依赖
- 预期输出
- 所需技能
- 推荐工具或 Agent 类型

示例：

```json
{
  "id": "T2",
  "goal": "清洗订单表中的缺失值并标准化日期字段",
  "input": ["orders.csv"],
  "expected_output": "cleaned_orders.csv",
  "required_skills": ["data_cleaning", "date_normalization"]
}
```

### FR-3：Agent 能力画像管理

系统应维护一份 Agent Profile 配置，用于描述每个 Agent 的能力边界。

每个 Agent Profile 至少包括：

- Agent 名称
- 擅长技能
- 可用工具
- 适用场景
- 不适用场景
- 默认优先级

### FR-4：Agent 选择

系统应根据子任务所需技能选择合适的 Agent。MVP 阶段采用规则匹配方式：

```text
data_inspection → DataInspectAgent
sql / join / groupby / aggregation → SQLAgent
python / pandas / statistics / cleaning → PythonAgent
verification / format_check → VerifierAgent
其他 → GeneralDataAgent
```

后续版本可扩展为：

- LLM 选择
- embedding 相似度选择
- benchmark 加权选择
- 小样本实验选择

### FR-5：子任务执行

系统应按照子任务顺序依次执行任务。每个子任务执行时需要：

1. 获取当前上下文。
2. 获取依赖的中间结果。
3. 调用对应 Agent。
4. 调用必要工具。
5. 生成中间结果。
6. 保存 artifact。
7. 更新 catalog。

### FR-6：中间结果保存

系统应将每个子任务的输出保存为 artifact，并在 catalog 中登记。

示例目录：

```text
artifacts/
  T1_data_catalog.json
  T2_cleaned_orders.csv
  T3_analysis_result.csv
  T4_verification.json

catalog.json
```

### FR-7：结果验证

系统应在每个子任务执行后进行验证。验证分为两层：

第一层：硬规则验证。

- 输出文件是否存在。
- 输出是否为空。
- 字段是否符合预期。
- 行数是否异常。
- 是否存在大量 NaN。
- 输出格式是否满足任务要求。

第二层：LLM 验证。

- 子任务目标是否完成。
- 中间结果是否能支撑后续任务。
- 最终答案是否回答了用户问题。
- 是否存在明显逻辑漏洞。

### FR-8：失败修复

当某个子任务失败时，系统应按以下顺序处理：

```text
同 Agent 重试一次
  ↓
LLM 改写子任务描述
  ↓
切换到候选 Agent
  ↓
使用 PythonAgent 兜底
  ↓
局部重规划剩余任务
  ↓
仍失败则输出失败原因
```

MVP 阶段最多重试 2 次，避免无限循环。

### FR-9：最终答案生成

系统应整合所有中间结果，生成最终答案。最终答案需要包含：

- 直接答案
- 使用的数据来源
- 核心计算步骤
- 关键中间结果
- 可能的不确定性
- 输出文件路径，如果有

---

## 3.2 非功能性需求

### NFR-1：可调试性

系统应记录每一步的日志，包括：

- 当前子任务
- 选择的 Agent
- 调用的工具
- 输入摘要
- 输出摘要
- 验证结果
- 错误信息
- 重试次数

### NFR-2：可扩展性

系统应支持后续添加新的 Agent、工具和选择策略，不应把所有逻辑写死在单个文件中。

### NFR-3：稳定性

系统应避免因为单个工具调用失败导致全流程崩溃。每个子任务应具备异常捕获和失败恢复机制。

### NFR-4：可复现性

系统应保存：

- 输入任务
- 规划结果
- 执行日志
- 中间结果
- 最终结果
- 模型调用参数

### NFR-5：部署便利性

系统应支持 Docker 部署，所有依赖通过 requirements.txt 或 pyproject.toml 管理。

---

# 4. 系统总体架构

## 4.1 总体架构图

```text
┌──────────────────────┐
│      User Task        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Data Inspector      │
│  数据读取与摘要生成     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Planner         │
│   任务拆解 / 计划生成   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Selector        │
│   子任务 Agent 选择     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Executor        │
│   调用 Agent / Tools   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Verifier        │
│   中间结果验证          │
└───────┬─────────┬─────┘
        │通过      │失败
        ↓          ↓
┌─────────────┐  ┌────────────────┐
│ Next Task   │  │    Refiner      │
└──────┬──────┘  │ 重试/改写/换Agent │
       │         └───────┬────────┘
       └─────────────────┘
              ↓
┌──────────────────────┐
│     Final Answer      │
└──────────────────────┘
```

## 4.2 分层设计

系统分为五层：

```text
应用层：用户任务入口、最终答案输出
编排层：Planner、Selector、Executor、Verifier、Refiner
Agent 层：DataInspectAgent、SQLAgent、PythonAgent、VerifierAgent
工具层：文件工具、SQL 工具、Python 执行工具、数据分析工具
存储层：artifacts、catalog、logs、memory
```

---

# 5. 核心模块设计

## 5.1 Data Inspector 模块

### 5.1.1 职责

Data Inspector 负责扫描输入目录，识别文件类型，读取样例数据，生成数据摘要。

### 5.1.2 输入

```text
input_dir: str
```

### 5.1.3 输出

```json
{
  "files": [
    {
      "file_name": "orders.csv",
      "file_type": "csv",
      "rows": 10000,
      "columns": 12,
      "schema": {
        "order_id": "string",
        "user_id": "string",
        "amount": "float",
        "created_at": "datetime"
      },
      "sample_rows": [...],
      "missing_values": {
        "amount": 0,
        "created_at": 12
      }
    }
  ]
}
```

### 5.1.4 关键实现

推荐使用：

- pathlib 扫描目录
- pandas / Polars 读取表格
- duckdb 查询大文件
- sqlite3 读取 SQLite
- json 读取 JSON

---

## 5.2 Planner 模块

### 5.2.1 职责

Planner 根据用户任务和数据摘要生成任务计划。

### 5.2.2 输入

```json
{
  "user_task": "请分析销售额最高的地区，并给出原因",
  "data_catalog": {...},
  "agent_profiles": [...]
}
```

### 5.2.3 输出

```json
{
  "subtasks": [
    {
      "id": "T1",
      "goal": "理解销售数据表结构，确定销售额和地区字段",
      "input": ["data_catalog"],
      "expected_output": "字段映射说明",
      "required_skills": ["data_inspection"]
    },
    {
      "id": "T2",
      "goal": "按地区聚合销售额，找出销售额最高地区",
      "input": ["T1"],
      "expected_output": "地区销售额排序表",
      "required_skills": ["aggregation", "groupby", "sql"]
    }
  ]
}
```

### 5.2.4 Prompt 设计要点

Planner 的提示词应强调：

1. 子任务数量控制在 3 到 6 个。
2. 每个子任务必须有明确输入和输出。
3. 每个子任务必须可执行、可验证。
4. 不要生成空泛任务。
5. 不要重复拆解相同目标。
6. 输出必须是合法 JSON。

---

## 5.3 Agent Registry 模块

### 5.3.1 职责

维护所有 Agent 的能力画像。

### 5.3.2 示例配置

```json
[
  {
    "name": "DataInspectAgent",
    "skills": ["data_inspection", "schema_summary", "file_reading"],
    "tools": ["list_files", "read_table_head", "inspect_sqlite"],
    "best_for": "识别数据文件、理解 schema、生成数据摘要",
    "fallback_priority": 1
  },
  {
    "name": "SQLAgent",
    "skills": ["sql", "join", "groupby", "aggregation", "filter"],
    "tools": ["duckdb_query", "sqlite_query"],
    "best_for": "结构化表格分析、多表 join、聚合统计",
    "fallback_priority": 2
  },
  {
    "name": "PythonAgent",
    "skills": ["python", "pandas", "data_cleaning", "statistics", "machine_learning"],
    "tools": ["run_python"],
    "best_for": "复杂数据清洗、统计分析、灵活计算",
    "fallback_priority": 3
  },
  {
    "name": "VerifierAgent",
    "skills": ["verification", "format_check", "logic_check"],
    "tools": ["check_file_exists", "run_python"],
    "best_for": "检查中间结果和最终答案是否满足要求",
    "fallback_priority": 4
  }
]
```

---

## 5.4 Selector 模块

### 5.4.1 职责

根据子任务选择最合适的 Agent。

### 5.4.2 MVP 选择策略

MVP 阶段使用技能匹配：

```python
def select_agent(subtask, agent_profiles):
    required = set(subtask["required_skills"])
    best_agent = None
    best_score = -1

    for agent in agent_profiles:
        skills = set(agent["skills"])
        score = len(required & skills) / max(len(required), 1)
        if score > best_score:
            best_score = score
            best_agent = agent

    return best_agent
```

### 5.4.3 后续升级策略

后续版本可加入：

```text
score = 0.4 × 技能匹配度
      + 0.3 × 相似 benchmark 加权得分
      + 0.2 × 历史成功率
      + 0.1 × LLM judge 分数
      - 0.1 × 成本惩罚
```

---

## 5.5 Executor 模块

### 5.5.1 职责

Executor 负责实际执行子任务，调用对应 Agent 和工具。

### 5.5.2 输入

```json
{
  "subtask": {...},
  "agent": {...},
  "context": {...},
  "artifacts": {...}
}
```

### 5.5.3 输出

```json
{
  "subtask_id": "T2",
  "status": "success",
  "output_path": "artifacts/T2_result.csv",
  "summary": "已按地区计算销售额并排序",
  "error": null
}
```

### 5.5.4 执行原则

1. 所有工具调用必须被日志记录。
2. 所有中间结果必须保存。
3. 不把大表完整塞进 LLM 上下文。
4. 大数据只传 schema、样例、统计摘要。
5. 复杂计算优先使用 Python / SQL 工具，而不是让 LLM 心算。

---

## 5.6 Verifier 模块

### 5.6.1 职责

检查子任务输出是否满足要求。

### 5.6.2 验证类型

#### 硬规则验证

```python
def hard_verify(result):
    checks = []
    checks.append(check_output_exists(result))
    checks.append(check_not_empty(result))
    checks.append(check_schema_if_needed(result))
    return all(checks)
```

#### LLM 验证

LLM 验证输入：

```json
{
  "subtask_goal": "按地区聚合销售额",
  "expected_output": "地区销售额排序表",
  "actual_output_summary": "生成了 region_sales.csv，包含 region 和 total_sales 两列"
}
```

LLM 验证输出：

```json
{
  "passed": true,
  "reason": "输出包含地区字段和销售额聚合字段，满足子任务要求",
  "suggestion": null
}
```

---

## 5.7 Refiner 模块

### 5.7.1 职责

当子任务失败时，对失败进行局部修复。

### 5.7.2 修复策略

```text
Strategy 1：同 Agent 重试
Strategy 2：改写子任务描述
Strategy 3：补充输入上下文
Strategy 4：切换 next-best Agent
Strategy 5：使用 PythonAgent 兜底
Strategy 6：局部重规划剩余任务
```

### 5.7.3 重试限制

```text
每个子任务最多重试 2 次
整个任务最多触发 1 次局部重规划
超过限制后返回失败原因
```

---

## 5.8 Artifact Manager 模块

### 5.8.1 职责

负责保存和索引中间结果。

### 5.8.2 Artifact 类型

```text
表格结果：csv / parquet
结构化结果：json
文本报告：md / txt
图像结果：png / jpg
日志结果：log
```

### 5.8.3 catalog.json 示例

```json
{
  "T1": {
    "type": "json",
    "path": "artifacts/T1_data_catalog.json",
    "description": "输入数据 schema 和样例摘要",
    "created_by": "DataInspectAgent"
  },
  "T2": {
    "type": "csv",
    "path": "artifacts/T2_region_sales.csv",
    "description": "按地区聚合后的销售额表",
    "created_by": "SQLAgent"
  }
}
```

---

# 6. LangGraph 工作流设计

## 6.1 状态定义

```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    user_task: str
    input_dir: str
    data_catalog: Dict[str, Any]
    agent_profiles: List[Dict[str, Any]]
    subtasks: List[Dict[str, Any]]
    current_task_index: int
    current_subtask: Optional[Dict[str, Any]]
    current_agent: Optional[Dict[str, Any]]
    artifacts: Dict[str, Any]
    verification_result: Optional[Dict[str, Any]]
    retry_count: int
    errors: List[Dict[str, Any]]
    final_answer: Optional[str]
```

## 6.2 节点设计

```text
inspect_data
plan_tasks
select_agent
execute_subtask
verify_result
refine_subtask
move_next
generate_final_answer
```

## 6.3 边设计

```text
START → inspect_data
inspect_data → plan_tasks
plan_tasks → select_agent
select_agent → execute_subtask
execute_subtask → verify_result
verify_result → move_next，如果验证通过
verify_result → refine_subtask，如果验证失败且可重试
refine_subtask → execute_subtask
move_next → select_agent，如果还有子任务
move_next → generate_final_answer，如果所有子任务完成
```

---

# 7. 数据结构设计

## 7.1 SubTask 数据结构

```json
{
  "id": "T1",
  "goal": "识别输入数据结构",
  "input": ["input_dir"],
  "expected_output": "data catalog",
  "required_skills": ["data_inspection"],
  "status": "pending",
  "assigned_agent": null,
  "retry_count": 0
}
```

## 7.2 AgentProfile 数据结构

```json
{
  "name": "SQLAgent",
  "skills": ["sql", "join", "groupby"],
  "tools": ["duckdb_query"],
  "best_for": "结构化数据聚合分析",
  "weakness": "复杂非结构化文本分析",
  "priority": 2
}
```

## 7.3 ExecutionResult 数据结构

```json
{
  "subtask_id": "T2",
  "agent": "SQLAgent",
  "status": "success",
  "output_path": "artifacts/T2_result.csv",
  "summary": "成功生成聚合结果",
  "error": null,
  "metadata": {
    "rows": 20,
    "columns": ["region", "total_sales"]
  }
}
```

## 7.4 VerificationResult 数据结构

```json
{
  "subtask_id": "T2",
  "passed": true,
  "hard_checks": {
    "file_exists": true,
    "not_empty": true,
    "schema_valid": true
  },
  "llm_check": {
    "passed": true,
    "reason": "结果满足子任务要求"
  },
  "suggestion": null
}
```

---

# 8. 工具设计

## 8.1 文件工具

```text
list_files(input_dir)
read_text_file(path)
read_json_file(path)
read_csv_sample(path, n=5)
read_excel_sample(path, n=5)
write_json(path, data)
write_csv(path, dataframe)
```

## 8.2 SQL 工具

```text
duckdb_query(sql, tables)
inspect_sqlite_schema(db_path)
sqlite_query(db_path, sql)
```

## 8.3 Python 工具

```text
run_python(code, timeout=60)
execute_pandas_script(script_path)
validate_dataframe(path)
```

## 8.4 验证工具

```text
check_file_exists(path)
check_not_empty(path)
check_columns(path, expected_columns)
check_json_schema(path, schema)
```

---

# 9. 项目目录结构

推荐目录结构：

```text
data-agent/
  README.md
  requirements.txt
  Dockerfile
  main.py

  config/
    agent_profiles.json
    model_config.yaml
    system_prompts.yaml

  src/
    graph/
      state.py
      workflow.py
      nodes.py
      edges.py

    agents/
      base.py
      planner_agent.py
      data_inspect_agent.py
      sql_agent.py
      python_agent.py
      verifier_agent.py
      report_agent.py

    core/
      planner.py
      selector.py
      executor.py
      verifier.py
      refiner.py
      artifact_manager.py
      catalog.py
      logger.py

    tools/
      file_tools.py
      table_tools.py
      sql_tools.py
      python_tools.py
      verify_tools.py

    llm/
      client.py
      prompts.py
      json_parser.py

    utils/
      safe_exec.py
      path_utils.py
      retry.py

  input/
    .gitkeep

  output/
    final_answer.json

  artifacts/
    .gitkeep

  logs/
    run.log
```

---

# 10. 运行流程设计

## 10.1 命令行入口

```bash
python main.py \
  --task "请找出销售额最高的地区并解释原因" \
  --input_dir ./input \
  --output_dir ./output
```

## 10.2 主流程伪代码

```python
def run_data_agent(user_task: str, input_dir: str, output_dir: str):
    state = init_state(user_task, input_dir, output_dir)

    state = inspect_data(state)
    state = plan_tasks(state)

    while has_next_subtask(state):
        state = select_agent(state)
        state = execute_subtask(state)
        state = verify_result(state)

        if not state["verification_result"]["passed"]:
            state = refine_subtask(state)
            if state["retry_count"] > MAX_RETRY:
                break
        else:
            state = move_next(state)

    state = generate_final_answer(state)
    save_final_answer(state)
    return state
```

---

# 11. Prompt 设计

## 11.1 Planner Prompt

```text
你是一个 Data Agent 任务规划器。

你的任务是根据用户问题和数据摘要，将复杂任务拆解为 3 到 6 个可执行子任务。

要求：
1. 每个子任务必须有明确目标。
2. 每个子任务必须有输入和预期输出。
3. 每个子任务必须可被一个数据处理 Agent 执行。
4. 不要生成重复子任务。
5. 不要生成空泛描述。
6. 输出必须是合法 JSON。

用户任务：
{user_task}

数据摘要：
{data_catalog}

可用 Agent：
{agent_profiles}
```

## 11.2 Verifier Prompt

```text
你是一个 Data Agent 结果验证器。

请判断当前子任务的执行结果是否满足子任务目标。

子任务目标：
{subtask_goal}

预期输出：
{expected_output}

实际输出摘要：
{actual_output_summary}

请输出 JSON：
{
  "passed": true 或 false,
  "reason": "原因",
  "suggestion": "如果失败，给出修复建议"
}
```

## 11.3 Refiner Prompt

```text
你是一个 Data Agent 子任务修复器。

当前子任务执行失败，请根据错误信息改写子任务，使其更清晰、更容易执行。

原子任务：
{subtask}

错误信息：
{error}

已有数据和中间结果：
{context}

请输出修复后的子任务 JSON。
```

---

# 12. 错误处理设计

## 12.1 错误类型

```text
DataLoadError：数据读取失败
PlanningError：任务规划失败
AgentSelectionError：找不到合适 Agent
ToolExecutionError：工具执行失败
VerificationError：结果验证失败
LLMOutputParseError：LLM 输出 JSON 解析失败
TimeoutError：执行超时
```

## 12.2 错误处理策略

| 错误类型            | 处理方式                                  |
| ------------------- | ----------------------------------------- |
| DataLoadError       | 尝试其他读取方式，失败则记录文件不可用    |
| PlanningError       | 重新调用 Planner，限制输出格式            |
| AgentSelectionError | 使用 GeneralDataAgent 或 PythonAgent 兜底 |
| ToolExecutionError  | 重试一次，仍失败则切换 Agent              |
| VerificationError   | 进入 Refiner                              |
| LLMOutputParseError | 尝试 JSON 修复，失败则重新调用 LLM        |
| TimeoutError        | 降低数据规模或使用采样数据                |

---

# 13. 日志与可观测性设计

## 13.1 日志内容

每次运行生成一个 run_id：

```text
logs/
  run_20260426_001.log
  run_20260426_001_trace.json
```

trace 中记录：

```json
{
  "run_id": "20260426_001",
  "user_task": "...",
  "steps": [
    {
      "node": "plan_tasks",
      "input_summary": "...",
      "output_summary": "...",
      "status": "success",
      "elapsed_seconds": 3.2
    }
  ]
}
```

## 13.2 调试输出

建议支持 debug 模式：

```bash
python main.py --debug
```

debug 模式下输出：

- Planner 原始输出
- JSON 解析结果
- Agent 选择分数
- Tool 调用参数
- Verifier 判断原因
- Refiner 修改前后对比

---

# 14. 安全设计

## 14.1 Python 执行安全

由于系统可能执行 LLM 生成的 Python 代码，需要进行限制：

- 设置执行超时。
- 限制工作目录。
- 禁止访问系统敏感路径。
- 禁止危险命令。
- 尽量在 Docker 容器内执行。
- 记录所有执行代码。

## 14.2 文件访问安全

所有文件读写应限制在：

```text
input/
output/
artifacts/
logs/
```

禁止写入项目目录外部路径。

---

# 15. 部署设计

## 15.1 Dockerfile 示例

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

## 15.2 requirements.txt 示例

```text
langgraph
langchain
litellm
pandas
polars
duckdb
pyarrow
openpyxl
pydantic
python-dotenv
```

## 15.3 环境变量

```text
MODEL_API_KEY=xxx
MODEL_BASE_URL=xxx
MODEL_NAME=xxx
```

---

# 16. 测试设计

## 16.1 单元测试

需要测试：

```text
test_data_inspector.py
test_planner_json_parse.py
test_selector.py
test_executor.py
test_verifier.py
test_artifact_manager.py
```

## 16.2 集成测试

准备若干小型任务：

```text
任务 1：单表聚合
任务 2：多表 join
任务 3：缺失值清洗
任务 4：JSON 数据抽取
任务 5：结果格式化输出
```

每个任务验证：

- 是否成功生成计划
- 是否正确选择 Agent
- 是否生成 artifact
- 最终答案是否正确
- 失败时是否能重试

---

# 17. 评估指标

## 17.1 任务成功率

```text
成功任务数 / 总任务数
```

## 17.2 子任务成功率

```text
成功子任务数 / 总子任务数
```

## 17.3 一次通过率

```text
无需重试即完成的任务数 / 总任务数
```

## 17.4 平均重试次数

```text
总重试次数 / 总任务数
```

## 17.5 平均耗时

```text
总运行时间 / 总任务数
```

## 17.6 最终得分

如果用于比赛，则以官方评测分数或本地评分脚本为准。

---

# 18. 开发路线图

## v0.1：最小可运行版本

目标：跑通完整链路。

功能：

- 输入数据扫描
- LLM 任务拆解
- 规则 Agent 选择
- Python / SQL 工具调用
- 中间结果保存
- 简单验证
- 最终答案生成

## v0.2：增强稳定性

目标：提升任务成功率。

功能：

- Verifier 强化
- Refiner 重试机制
- JSON 输出修复
- 更好的日志系统
- PythonAgent 兜底

## v0.3：增强 Agent 选择

目标：更接近论文中的 heterogeneous agent selection。

功能：

- Agent 历史成功率统计
- task skill 标签
- 通用 embedding 检索相似任务
- benchmark 加权 Agent 选择

## v0.4：引入局部重规划

目标：失败后不直接终止。

功能：

- 当前子任务之后的 partial replan
- 中间结果复用
- catalog 驱动重规划

## v0.5：DAG 与并行执行

目标：提高效率。

功能：

- 子任务依赖图
- 拓扑排序执行
- 无依赖子任务并行执行
- 多 Agent 并发调度

---

# 19. 风险分析

## 19.1 LLM 规划不稳定

风险：Planner 输出任务过粗、过细或 JSON 格式错误。

缓解：

- 强制 JSON schema。
- 使用 pydantic 校验。
- 失败时自动修复 JSON。
- 限制子任务数量。

## 19.2 Agent 选择错误

风险：错误 Agent 执行子任务，导致失败。

缓解：

- Verifier 检查。
- next-best Agent 切换。
- PythonAgent 兜底。
- 后续引入 benchmark 选择。

## 19.3 工具执行失败

风险：Python 代码报错，SQL 查询失败。

缓解：

- 捕获异常。
- 让 LLM 根据错误日志修复代码。
- 设置重试次数。
- 保存失败代码便于复盘。

## 19.4 上下文过长

风险：大数据或多轮中间结果导致 prompt 超长。

缓解：

- 不传完整数据，只传 schema 和样例。
- 中间结果存文件，只传路径和摘要。
- 使用 catalog 管理数据资产。

## 19.5 结果不可验证

风险：最终答案看似合理但实际错误。

缓解：

- 多用程序化验证。
- 对关键计算保存代码和结果。
- 输出核心证据。
- 对最终答案进行反向检查。

---

# 20. 总结

本项目设计的 Data Agent 系统采用轻量级多模块架构，核心思想是：

```text
任务拆解 → Agent 选择 → 工具执行 → 中间验证 → 局部修复 → 最终输出
```

与完整研究型 Data Agent 系统相比，本设计有意降低复杂度，优先实现可运行、可调试、可扩展的工程版本。MVP 阶段只保留论文式 Data Agent 的主干思想：子任务化、能力匹配、中间结果检查、失败修复和 artifact 复用。

后续系统可以逐步升级为更完整的多 Agent pipeline orchestration：引入 benchmark 加权 Agent 选择、embedding 相似任务检索、局部重规划、DAG 并行执行、长期记忆和持续学习。

本设计适合作为 Data Agent 比赛项目、数据分析自动化项目或科研原型系统的起点。
