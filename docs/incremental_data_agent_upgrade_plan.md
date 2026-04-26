# 当前项目增量升级为 Data Agent 的开发计划

本文面向当前仓库 `kddcup2026-data-agents-starter-kit`，目标是用增量开发方式，把现有 ReAct-style / LangGraph baseline 逐步升级为 `data_agent_software_design_doc.md` 中描述的 Data Agent。设计书里的代码和目录只是概念演示，本文按当前项目已有结构做适配，不建议重建一个全新的 `data-agent/` 目录。

---

## 1. 当前项目基线判断

当前项目已经具备一条比赛可用的主链路：

```text
dabench CLI
  -> DABenchPublicDataset 读取 task.json + context/
  -> LangGraphAgent 单 Agent 工具调用循环
  -> ToolRegistry 分发 list/read/sql/python/answer 工具
  -> runner 写 trace.json / prediction.csv / summary.json
  -> score-run 用公开 gold 做本地代理评分
  -> submit / Docker 路径面向评测环境
```

当前最值得保留的资产：

- `src/data_agent_baseline/agents/langgraph_runtime.py`：已有 LangGraph 主循环、模型/tool 路由、trace 记录。
- `src/data_agent_baseline/tools/registry.py`：已有统一工具注册与 `answer` 终止工具。
- `src/data_agent_baseline/run/runner.py`：已有单任务、批任务、超时、并发、产物落盘。
- `src/data_agent_baseline/scoring.py`：已有公开 demo 的本地代理评分与错误诊断。
- `configs/submission.yaml`、`Dockerfile`、`run/submission.py`：已有提交态路径。

当前与目标 Data Agent 的主要差距：

- 没有显式 Data Inspector / Data Catalog，模型要靠工具调用临时发现数据。
- 没有前置 Perception / 语义合同，字段归属、过滤条件、聚合口径容易漂移。
- 没有显式 Planner / SubTask 状态，复杂任务靠单 Agent 即兴推进。
- 没有真正的 Agent Profile / Selector，目前只有一个通用工具型 Agent。
- `answer` 一旦调用就终止，缺少提交前验证与修复。
- 工具失败后的恢复主要靠模型自发重试，缺少语义纠偏和局部重规划。
- 中间结果主要存在临时 Python 工作区和 trace 里，缺少可复用 artifact catalog。

因此升级路线应是：先补观测、合同和验证，再引入规划、选择和修复，最后再考虑更复杂的多 Agent / DAG / 记忆。

---

## 2. 增量开发总原则

1. 保持提交接口稳定不改变 `dabench submit`、`prediction.csv`、`/input`、`/output`、`/logs` 约定。
2. 保持现有 baseline 可回退新能力尽量通过配置开关接入，例如 `agent.enable_data_inspector`、`agent.enable_answer_validation`，避免一次性替换整条运行图。
3. 每个增量都要能单独评分每完成一个阶段，就固定运行同一组公开任务，比较 `primary_proxy_score`、`mean_recall`、`mean_redundancy_rate`、未提交数、超时数、模型轮数和耗时。
4. 不把大表塞进 prompt数据目录、计划、中间结果只传 schema、样例、统计摘要和 artifact 路径。
5. 先做硬规则，再做 LLM 判断验证、工具路由、文件类型检查、列宽检查、空输出检查都应优先用确定性代码完成。
6. 计划服务于比赛任务
   设计书建议 3 到 6 个子任务，但当前 DABench 公开任务有不少是单表或短链路问题；实际实现应允许 `1 到 4` 个子任务，复杂任务才拆得更细。

---

## 3. 固定评估协议

建议先在增量 0 中冻结一套固定评估方法，后续每个增量都按同样方式验证。

### 3.1 参考基线

仓库已有三份可参考批次，其中 `artifacts/standard/baseline` 建议作为固定的标准对比基线：

| 基线来源                            | run_id               | 任务数 | 有预测任务 | Primary λ=0.1 | Mean Recall | Mean Redundancy | 主要失败                     |
| ----------------------------------- | -------------------- | -----: | ---------: | -------------: | ----------: | --------------: | ---------------------------- |
| `artifacts/standard/baseline`     | `20260424T011929Z` |     50 |         40 |       0.500000 |    0.500000 |        0.300000 | 10 个 max_steps 未提交       |
| `artifacts/runs/20260412T035736Z` | `20260412T035736Z` |     50 |         46 |       0.698805 |    0.710000 |        0.311952 | 4 个 max_steps 未提交        |
| `artifacts/runs/20260424T014353Z` | `20260424T014353Z` |     50 |         47 |       0.677000 |    0.690000 |        0.370000 | 2 个 max_steps，1 个 timeout |

`artifacts/standard/baseline` 的运行参数来自其 `summary.json`：`max_steps=32`、`temperature=0.0`、`max_workers=4`、`task_timeout_seconds=600`。该目录应尽量保持只读，用作长期对比锚点；新实验可以继续写到 `artifacts/runs/<run_id>/`。

后续建议以一次新跑出的 `baseline_freeze_*` 作为主比较基线，因为模型、API 服务和参数会影响结果。

### 3.2 固定任务切片

建议新增或维护几份本地评估配置，均复用 `run.task_ids`：

- Smoke：`task_11, task_19, task_26`
- 语义合同风险：`task_25, task_80, task_89, task_163, task_180, task_379`
- 答案冗余风险：`task_24, task_38, task_74, task_287, task_292, task_303, task_330`
- 长链路/不提交风险：`task_173, task_344, task_352, task_396, task_418`
- Full public：全部 50 个公开 demo 任务

### 3.3 每阶段固定命令

```powershell
uv run pytest
uv run dabench run-selected-tasks --config configs/eval_smoke.yaml
uv run dabench score-run <run_id>
uv run dabench run-selected-tasks --config configs/eval_contract.yaml
uv run dabench score-run <run_id>
uv run dabench run-benchmark --config configs/react_baseline.example.yaml
uv run dabench score-run <run_id>
```

提交态验证在接近可提交版本时执行：

```powershell
uv run dabench submit
```

或用 README 中的 Docker 模拟命令挂载 `/input`、`/output`、`/logs`。

### 3.4 每阶段关注指标

- `primary_proxy_score`：主比较指标。
- `mean_recall`：是否真的覆盖了 gold 列。
- `mean_redundancy_rate`：是否减少多余列。
- `prediction_task_count`：是否减少未提交。
- `failure_breakdown`：max_steps、timeout、模型请求失败、验证失败等原因。
- `mean_model_step_count` / `max_model_step_count`：是否减少空转。
- `p95_e2e_elapsed_seconds` / `max_e2e_elapsed_seconds`：是否引入明显耗时风险。
- 重点任务的 trace：是否出现字段归属、聚合口径和最终提交形状改善。

---

## 4. 增量 0：冻结基线与调参试验框架

### 目标

在改代码前先建立稳定对照组，让之后每个阶段的收益和回退都可见。

### 建议改动

- 新增 `configs/eval_smoke.yaml`、`configs/eval_contract.yaml`、`configs/eval_redundancy.yaml`、`configs/eval_long.yaml`。
- 新增一份 `docs/eval_protocol.md` 或在本文后续维护每次增量的评分记录。
- 在 `summary.json` 中继续保留当前已有的 `max_steps`、`temperature`、`task_timeout_seconds`、`max_workers`。
- 可选：新增一个轻量脚本或 CLI 子命令，用来汇总多个 run 的 `score.json`，形成横向对比表。

### 可调优项

- `agent.max_steps`：例如 16、24、32。
- `agent.temperature`：建议主线保持 0.0，少量试验可测 0.1。
- `run.task_timeout_seconds`：长任务阶段可单独测 600、900。
- `run.max_workers`：批量效率参数，不作为质量提升项混入对比。

### 验证方式

- `uv run pytest` 通过。
- 固定切片都能生成 `summary.json` 并可被 `score-run` 评分。
- 得到一份新的 baseline freeze 报告，记录主分、冗余率、未提交数、超时数。

### 验收标准

- 后续任何增量都能和同一批任务、同一组参数比较。
- 文档中记录 baseline 指标，避免凭印象判断效果。

---

## 5. 增量 1：Data Inspector 模块

### 目标

在主解题 Agent 开始前，先完成“感知 + 数据理解/探索”，把用户问题的语义和底层数据的语义对齐。这个增量不只是列文件或生成 schema，而是建立 Data Agent 的前置认知层：

- 先理解 `data/query`、environment、optimization goal。
- 再组织和理解数据，让后续 Agent 更容易发现、访问和使用数据。
- 使用统一语义目录、轻量 data fabric、semantic data organization 和 semantic indexes 提高探索效率。
- 将原本单独的 Answer Contract 能力并入 Perception，作为“答案形状、过滤条件、指标口径、风险术语”的前置感知。

### 阶段 1：Perception

Perception 负责理解当前任务的外部语境和执行目标，输出结构化 `PerceptionResult`。建议至少包含：

A. Source Scanner

负责确定性扫描：

* 有哪些文件/表/文档
* 各自规模多大
* 哪些是结构化，哪些是非结构化
* 哪些对象名字和问题关键词更相关

这是最底层、最稳定的一层。

C. Task Classifier

负责判断题型和所需能力：

* 这是计算题还是抽取题
* 是结构化优先还是文档优先
* 是否必须多源融合
* 是否需要长上下文阅读

Perception 可以先用离线 prompt 模板实现，模板要显式对齐 Data Agent 的职责：理解 environment、data、tasks、agents、，而不是直接求答案。

### 阶段 2：Data Understanding and Exploration Agent

这一阶段设计一个单独的 Data Understanding and Exploration Agent。它只负责探索和组织数据，不提交最终答案，不替代主解题 Agent。

职责：

- 组织和理解数据资产，包括 CSV、JSON、SQLite/DB、Markdown/TXT。
- 让后续 Agent 更容易发现和访问数据，给出每类资产的推荐工具。
- 生成 unified semantic catalog，统一记录数据资产、字段语义、样例、关系和不确定性。
- 建立轻量 data fabric：把context 中分散的文件、表、文档和工具访问方式组织成可查询的数据访问层。
- 做 semantic data organization：识别疑似实体、事实表、维表、主键、外键、连接字段、指标字段。
- 做 semantic indexes：第一版采用轻量关键词/字段/术语倒排索引，不引入 embedding。

建议输出 `UnifiedSemanticCatalog`：

- `assets`：文件路径、类型、大小、推荐访问工具。
- `schemas`：表/文件字段、类型、样例、缺失情况、行数估计。
- `semantic_entities`：疑似实体、实体字段、实体来源。
- `field_meanings`：字段名、字段别名、从文档或样例推断出的含义。
- `relationships`：疑似主键、外键、join 候选、跨文件关系。
- `query_relevance`：与当前问题最相关的文件、字段、关系、文档片段。
- `semantic_uncertainties`：仍需主 Agent 核实的风险点。

建议输出 `semantic_index.json`：

- 文件名、表名、字段名 token index。
- 字段别名 index。
- query 关键词到字段/文件候选的倒排 index。
- 高风险术语到字段/公式候选的映射。
- 文件类型到推荐工具的路由 index。

### 当前项目适配

不要照设计书新建独立 `main.py` 或全新 `src/core/`。建议在当前包内新增：

```text
src/data_agent_baseline/inspectors/
  __init__.py
  perception.py
  data_understanding.py
  semantic_catalog.py
  semantic_index.py
  prompts.py
```

LangGraph 中建议先以主 Agent 前置阶段接入：

```text
init_state
  -> perceive_task
  -> understand_and_explore_data
  -> model_step / 后续主循环
```

同时可以保留一个普通工具：

```text
inspect_context
```

该工具供主 Agent 或后续调试复用，但增量 1 的核心产物应来自前置 Data Inspector，而不是完全依赖模型主动调用工具。

运行时建议写入：

```text
artifacts/runs/<run_id>/<task_id>/
  perception.json
  semantic_catalog.json
  semantic_index.json
```

如果短期内不想把 `run_output_dir` 传进 Agent，可先把三类产物写入 `trace.json` 的独立节点记录；后续 Artifact Manager 增量再统一落盘。

### 可调优项

- `agent.enable_data_inspector`：是否启用前置 Data Inspector。
- `data_inspector.max_exploration_steps`：Data Understanding Agent 最多探索步数。
- `data_inspector.catalog_sample_rows`：默认 5 或 10。
- `data_inspector.max_doc_chars`：默认 2000。
- `data_inspector.max_json_chars`：默认 4000。
- `data_inspector.null_check_rows`：大文件只抽样统计。
- `data_inspector.index_mode`：第一版固定为 `keyword`。
- `data_inspector.inject_summary_to_agent`：是否把压缩后的 perception + catalog 摘要注入主 Agent prompt。

### 验证方式

- 单元测试覆盖 Perception JSON 解析、字段完整性和高风险术语识别。
- 单元测试覆盖 CSV、JSON、SQLite、Markdown、空文件、坏 JSON 的 semantic catalog 生成。
- 单元测试覆盖 keyword semantic index：query 关键词能命中相关字段、文件和别名。
- 对 `task_11` 验证 catalog 能识别 `knowledge.md`、`json/Patient.json`、`json/Examination.json`，并发现 `ID` 关系。
- 对 `task_89` 验证 `ranked second` 被标记为高风险，并提示区分 `rank` / `position`。
- 对 `task_180` 验证 `per unit` 被标记为高风险，并提示可能是比值条件。
- 对 `task_80` 验证最终号码字段需要确认主档来源。
- 跑 Smoke、语义合同风险切片和 Full public，比较模型平均步数、主分和未提交数。

### 验收标准

- Data Inspector 失败不会导致整题直接失败，最多退回原 baseline 主循环。
- `perception.json`、`semantic_catalog.json`、`semantic_index.json` 或等价 trace 节点可审计。
- 主 Agent 初始上下文能看到压缩后的任务感知和数据语义目录。
- Smoke 切片不下降；语义风险切片至少在 trace 中体现更准确的字段/条件候选。
- Full public 的 `primary_proxy_score` 不明显下降，`mean_model_step_count` 持平或下降。

---

## 6. 增量 2：轻量 Planner 与 SubTask 计划

### 目标

把复杂任务从“单 Agent 即兴推进”改为“有短计划、有可验证里程碑”的执行方式。

### 当前项目适配

新增：

```text
src/data_agent_baseline/agents/planner.py
src/data_agent_baseline/agents/schemas.py
```

`SubTask` 建议字段：

- `id`
- `goal`
- `depends_on`
- `expected_output`
- `required_skills`
- `verification_hint`

与设计书不同，当前项目先不要拆出多个真实 Agent。第一版 Planner 只生成执行计划，并把计划注入当前 `LangGraphAgent` 的 system/task prompt，作为模型行动边界。

计划数量建议：

- 简单任务允许 1 到 2 步。
- 多表、多源、半结构化任务允许 3 到 5 步。
- 禁止空泛任务，例如“分析数据”“生成答案”这种不可验证描述。

### 可调优项

- `agent.enable_planner`。
- `agent.max_subtasks`：默认 4。
- `agent.allow_single_subtask`：默认 true。
- `agent.planner_temperature`：默认沿用主模型温度 0.0。
- `agent.plan_in_prompt_max_chars`：避免计划挤占上下文。

### 验证方式

- 单元测试 Planner JSON 解析、Pydantic 校验、非法 JSON 修复。
- 在 trace 中记录 `plan_tasks` step。
- 对长链路切片检查计划是否包含最终收敛步骤，而不是无限读取。
- 跑 Smoke，确认简单任务没有被过度规划导致步数上升明显。

### 验收标准

- 简单任务平均步数不明显增加。
- 长链路任务 trace 中能看到明确的最终输出步骤。
- `Agent did not submit an answer within max_steps` 数量不增加。

---

## 7. 增量 3：工具路由守卫与 DuckDB 表格工具

### 目标

降低“用错工具但继续沿着错误语义跑”的概率，并增强 CSV/JSON 表格分析能力。

### 当前项目适配

当前 `execute_context_sql` 只适用于 SQLite / DB，但历史 trace 中模型常把 CSV 当 SQL 数据库查。建议：

1. 给现有工具增加文件类型守卫对 CSV 调 `inspect_sqlite_schema` 或 `execute_context_sql` 时，返回明确建议：应使用 `read_csv`、`execute_python` 或新增 DuckDB 工具。
2. 新增 DuckDB 工具

```text
execute_table_sql
```

能力范围：

- 允许对 `context/` 内 CSV / Parquet / JSON 做只读 SQL。
- 由工具负责注册路径到 DuckDB view，模型只需传表路径和 SQL。
- 返回列、行、row_count、truncated。

3. 给工具结果增加 `suggested_next_actions`
   工具错误不只返回异常，也返回可执行替代方案。

### 可调优项

- `tools.enable_duckdb_table_sql`。
- `tools.table_sql_timeout_seconds`。
- `tools.table_sql_row_limit`。
- `tools.csv_autodetect_sample_size`。
- `tools.guard_invalid_file_type`：默认 true。

### 验证方式

- 单元测试 CSV 不能走 SQLite 工具，错误中包含替代建议。
- 单元测试 DuckDB 可查询 CSV、Parquet、JSON。
- 针对 `task_80, task_89, task_249, task_344, task_379` 跑工具风险切片。
- 统计 trace 中工具错误次数、同类错误重复次数。

### 验收标准

- 工具错误次数减少，或错误后恢复路径更短。
- 不引入写文件、越界读路径或非只读 SQL 风险。
- 至少在部分 CSV-heavy 任务上减少 Python 代码量或模型轮数。

---

## 8. 增量 4：Artifact Manager 与中间结果 Catalog

### 目标

把 Data Inspector 产物、计划、关键脚本和候选结果持久化，形成可复盘的任务级 artifact catalog，提升后续验证能力。

### 当前项目适配

新增：

```text
src/data_agent_baseline/artifacts/
  __init__.py
  manager.py
  catalog.py
```

让 `runner` 在创建任务输出目录后，把 `task_output_dir` 传入 Agent / ToolRuntimeContext：

```text
artifacts/runs/<run_id>/<task_id>/
  trace.json
  prediction.csv
  perception.json
  semantic_catalog.json
  semantic_index.json
  plan.json
  intermediate_catalog.json
  tool_artifacts/
```

建议先保存小而关键的内容：

- Data Inspector 输出：`perception.json`、`semantic_catalog.json`、`semantic_index.json`。
- Planner 输出。
- `execute_python` 的代码、stdout、stderr、成功状态。
- 被提交前的候选答案表摘要。
- Answer Validator 报告。

注意：当前 `TaskContextWorkspace` 会清理临时目录，不应依赖临时目录保存长期结果。

### 可调优项

- `artifacts.persist_tool_outputs`。
- `artifacts.persist_python_code`。
- `artifacts.max_artifact_bytes`。
- `artifacts.save_intermediate_tables`：默认只保存摘要，避免大量文件。
- `artifacts.redact_model_messages`：若未来涉及敏感数据，可控制 trace 内容。

### 验证方式

- 单元测试 artifact 写入是原子、UTF-8、路径不越界。
- 运行 Smoke，检查每个任务目录中新增文件存在且 JSON 合法。
- 故意制造 Python 失败，确认失败代码和 stderr 可复盘。

### 验收标准

- 每个任务的关键中间状态可从 artifact 目录重建。
- 不显著增加 Docker 提交输出体积。
- 不改变 `prediction.csv` 格式。

---

## 9. 增量 5：提交前 Answer Validator

### 目标

解决当前 Top2 问题：模型把中间结果、冗余列、错误粒度或候选集直接提交。

### 当前项目适配

当前 `answer` 工具在 `registry.py` 中校验基本结构后立即终止。建议分阶段改造：

1. 增量 5A：在 `_answer` 内增加硬规则校验，但只记录 warning，不拦截。
2. 增量 5B：当启用严格模式时，`answer` 不立即终止；若验证失败，返回非终止工具结果和修复建议，让模型再改一次。
3. 增量 5C：把验证从工具内抽成 LangGraph 节点 `validate_answer`，形成真正的提交前门禁。

硬规则建议：

- 列数和行宽合法。
- 答案列不应明显多于 Perception 中的 `expected_answer_shape`。
- 单值题不应提交大量行。
- 不应提交全量明细表作为最终答案。
- 预测列中不应出现明显辅助 ID、debug 字段，除非题目要求。
- 空答案必须和题意兼容。
- 数值数量级异常时给出 warning。
- 重复列、全空列、完全相同列应提示删除。

LLM 验证可作为第二层，但不应替代硬规则。

### 可调优项

- `agent.enable_answer_validation`。
- `agent.answer_validation_mode`：`warn`、`repair_once`、`strict`。
- `agent.max_answer_repair_attempts`：默认 1。
- `validator.max_extra_columns`。
- `validator.single_value_max_rows`。
- `validator.use_llm_check`：默认先 false，稳定后再开。

### 验证方式

- 单元测试：
  - 结构合法答案通过。
  - 多余列触发 warning。
  - 单值题多行触发 repair 建议。
  - 空答案在无证据时不直接通过。
- 运行答案冗余风险切片：
  - `task_24, task_38, task_74, task_287, task_292, task_303, task_330`
- 观察 `mean_redundancy_rate` 是否下降。

### 验收标准

- Full public 的 `mean_redundancy_rate` 下降。
- `primary_proxy_score` 不下降，理想情况下上升。
- 未提交数不因过严验证明显增加。

---

## 10. 增量 6：失败 Refiner 与局部重试

### 目标

解决当前 Top3 问题：工具失败或验证失败后只修执行形式，不重审语义；复杂任务接近 max_steps 仍不收敛。

### 当前项目适配

在 `AgentGraphState` 中新增有限状态：

- `retry_count_by_reason`
- `last_validation_result`
- `last_tool_error`
- `semantic_alignment_revision`
- `near_step_limit`

新增或改造节点：

```text
refine_after_tool_error
refine_after_validation_failure
force_converge_near_limit
```

处理顺序建议：

```text
工具失败
  -> 判断是否文件类型/路径问题
  -> 给确定性替代建议
  -> 如果重复失败，要求模型重述 Perception 中的答案形状和语义约束

验证失败
  -> 返回 validator 的具体失败项
  -> 最多修复 1 到 2 次

接近 max_steps
  -> 强制进入最小可答模式
  -> 禁止继续泛读，必须基于已有证据提交或说明失败
```

### 可调优项

- `agent.max_refine_attempts_per_task`：默认 2。
- `agent.max_validation_repair_attempts`：默认 1。
- `agent.force_converge_at_step_ratio`：例如达到 `max_steps * 0.8`。
- `agent.fallback_to_python_after_tool_errors`：默认 true。
- `agent.repeat_tool_call_threshold`。

### 验证方式

- 单元测试 route 条件：工具错误、validator 失败、接近 max_steps 都进入正确节点。
- 长链路切片：
  - `task_173, task_344, task_352, task_396, task_418`
- 对比：
  - 未提交数。
  - 平均模型轮数。
  - timeout 数。
  - 是否产生低质量过早提交。

### 验收标准

- `Agent did not submit an answer within max_steps` 和 timeout 数下降。
- 不因为强制收敛导致简单任务主分下降。
- trace 中能看出 refiner 不是只修语法，而是重审合同或答案形状。

---

## 11. 增量 7：Agent Profiles 与规则 Selector

### 目标

引入设计书中的 Agent Profile / Selector 思想，但先以“同一模型 + 不同角色 prompt + 工具子集”的轻量方式落地。

### 当前项目适配

不要一开始实现多个复杂 Agent 类。建议新增配置：

```text
configs/agent_profiles.yaml
```

Profile 示例：

- `DataInspectAgent`：偏数据目录、schema、样例读取。
- `SQLAgent`：偏 SQLite / DuckDB 查询、join、groupby、aggregation。
- `PythonAgent`：偏 pandas / polars、复杂清洗、半结构化解析。
- `VerifierAgent`：偏答案合同和提交形状检查。
- `GeneralDataAgent`：兜底。

实现方式：

- Selector 根据 `SubTask.required_skills`、数据文件类型和历史错误选择 profile。
- Profile 控制 system prompt 的重点和可见工具列表。
- 第一版仍串行执行，不做并行多 Agent。

### 可调优项

- `agent.enable_profiles`。
- `selector.skill_match_weight`。
- `selector.file_type_bonus`。
- `selector.failure_penalty`。
- `selector.default_profile`。
- `profiles.<name>.tool_allowlist`。

### 验证方式

- 单元测试 selector 分数：
  - SQL / join / groupby 选 SQLAgent。
  - pandas / cleaning / parsing 选 PythonAgent。
  - verification 选 VerifierAgent。
- trace 中记录每个子任务的 `assigned_profile`。
- 对 SQL-heavy、Python-heavy、doc-heavy 任务分别抽样评估。

### 验收标准

- Profile 选择可审计。
- 不降低 Smoke 和简单题性能。
- 工具误用率下降，尤其是 CSV/DB 工具混用。

---

## 12. 增量 8：显式 LangGraph 工作流重构

### 目标

把前面已经稳定的能力整理成真正的 Data Agent 工作流，而不是继续塞进单个模型循环。

### 当前项目适配

在当前 `langgraph_runtime.py` 基础上演进，不重写 runner 和 CLI。目标图：

```text
START
  -> init_state
  -> perceive_task
  -> understand_and_explore_data
  -> plan_tasks
  -> select_agent
  -> execute_subtask
  -> verify_result
  -> refine_subtask / move_next
  -> generate_final_answer
  -> finalize
  -> END
```

这个阶段才真正把设计书中的 `AgentState` 字段映射进当前 `AgentGraphState`：

- `perception`
- `semantic_catalog`
- `semantic_index`
- `subtasks`
- `current_task_index`
- `current_agent`
- `artifacts`
- `verification_result`
- `retry_count`
- `errors`

需要保持兼容：

- `runner.execute_task()` 仍返回 `AgentRunResult.to_dict()` 形态。
- `trace.json` 仍能被 `score-run` 读取。
- `answer` 仍能被写出为 `prediction.csv`。

### 可调优项

- `agent.workflow_mode`：`react_baseline`、`inspector_planned`、`full_data_agent`。
- `agent.max_steps_per_subtask`。
- `agent.max_subtasks`。
- `agent.max_replans`。
- `agent.verifier_mode`：hard-only / hard+llm。

### 验证方式

- 用 scripted model 测试每条 route。
- 对同一任务分别运行旧模式和新模式，确认输出目录结构兼容。
- Full public benchmark 比较主分和耗时。
- Docker submit dry-run 确认不破坏提交路径。

### 验收标准

- 新工作流可通过配置开启，旧 baseline 可配置回退。
- `trace.json` 能清晰呈现 perceive、understand、plan、select、execute、verify、refine。
- Full public 主分持平或提升，未提交数和冗余率至少一项改善。

---

## 13. 增量 9：历史表现记忆与 Benchmark 辅助选择

### 目标

在不使用 hidden gold 的前提下，利用公开 demo 的 trace 和 score 报告改进 Selector 与策略选择。

### 当前项目适配

新增离线分析产物：

```text
artifacts/diagnostics/
  task_signatures.json
  profile_success_stats.json
  failure_clusters.json
```

可从已有 `score.json`、`trace.json` 提取：

- 任务难度。
- 文件类型组合。
- 问题关键词。
- 使用工具序列。
- 是否超步、是否冗余、是否满覆盖。
- 哪类 profile / 工具路径更有效。

线上运行时只使用“规则和统计摘要”，不要读取 gold。

### 可调优项

- `selector.history_success_weight`。
- `selector.similar_task_top_k`。
- `selector.difficulty_weight`。
- `selector.disable_history_for_submission`：必要时可关闭历史记忆。

### 验证方式

- 对公开任务做交叉验证式评估：不要直接用同一任务的 gold 结果给同一任务加特权规则。
- 比较开启/关闭历史选择的 profile 分布和得分。
- 检查是否出现对公开任务过拟合的硬编码答案风险。

### 验收标准

- 选择策略更稳定，长链路和工具误用任务改善。
- 没有引入针对 task_id 的答案硬编码。
- hidden 提交路径不依赖 public gold。

---

## 14. 增量 10：局部重规划与 DAG/并行执行

### 目标

接近设计书后期能力：失败后局部重规划，并为可独立子任务预留并行执行空间。

### 当前项目适配

这不是早期重点。只有在显式工作流稳定后再做：

- `plan_tasks` 输出依赖关系。
- `execute_subtask` 支持跳过已完成依赖。
- `refine_subtask` 可只重写当前子任务之后的计划。
- 无依赖子任务可以并行，但最终 `answer` 仍单点提交。

### 可调优项

- `agent.enable_partial_replan`。
- `agent.max_replans`：默认 1。
- `agent.enable_parallel_subtasks`：默认 false。
- `agent.parallel_subtask_workers`。

### 验证方式

- 构造小型单元测试任务，验证拓扑排序和依赖失败传播。
- 在公开任务中只对复杂多源任务开启，避免简单任务额外开销。
- 对比长链路切片的未提交数和耗时。

### 验收标准

- 局部重规划不会丢失已有 catalog 和 artifacts。
- 并行模式不破坏 trace 顺序和最终答案。
- 复杂任务有收益后再考虑默认开启。

---

## 15. 推荐实施顺序

优先级从高到低：

```text
0. 冻结评估协议
1. Data Inspector（Perception + Data Understanding）
3. 工具路由守卫 + DuckDB 工具
5. Answer Validator
4. Artifact Manager
6. Refiner / 强制收敛
2. Planner
7. Agent Profiles / Selector
8. 显式 LangGraph 工作流
9. 历史表现记忆
10. 局部重规划 / DAG
```

说明：

- Data Inspector 替代原 Data Catalog + Answer Contract，是后续 Planner、Validator、Selector 的共同输入，应优先完成。
- Planner 在概念上很重要，但要消费 Data Inspector 输出；如果前置语义目录不稳定，过早规划会放大错误假设。
- Artifact Manager 可以和多个阶段交叉推进，但不要让它阻塞早期质量提升。
- 多 Agent / Selector 不应过早复杂化；先用 profile prompt 和工具子集模拟即可。

---

## 16. 每个增量的记录模板

建议每次完成一个增量后，在本文末尾或单独文档记录：

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

Contract slice:
  primary_proxy_score:
  重点任务变化：

Redundancy slice:
  mean_redundancy_rate:
  额外列任务变化：

Long slice:
  未提交数:
  timeout 数:
  max_model_step_count:

结论：
是否进入下一增量：
需要回退或继续调优的配置：
```

---

## 17. 近期最小可行路线

如果只做第一轮高性价比升级，建议按下面 5 步：

1. 增量 0：冻结评估切片和 baseline。
2. 增量 1A：完成 Perception，生成 `perception.json`，覆盖 query、environment、optimization goal、答案形状和风险术语。
3. 增量 1B：完成 Data Understanding and Exploration Agent，生成 `semantic_catalog.json` 和 keyword `semantic_index.json`。
4. 增量 3：新增工具路由守卫和 DuckDB 表格 SQL 工具，并利用 semantic index 给出推荐访问路径。
5. 增量 5：在 `answer` 前加硬规则 validator，先 warning，再 repair_once，并读取 Perception 的 `expected_answer_shape`。

这 5 步直接对应当前复盘中的三个高频短板：

- 题意语义锚定缺失。
- 工具误用后恢复不够聪明。
- 最终答案未收敛、冗余列伤分。

等这 5 步稳定后，再把 Planner、Selector、Refiner 正式纳入 LangGraph 状态，项目就会从“单 Agent 工具循环 baseline”自然演进成文档目标中的轻量级 Data Agent。
