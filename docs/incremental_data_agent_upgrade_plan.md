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
- 没有题意语义合同，字段归属、过滤条件、聚合口径容易漂移。
- 没有显式 Planner / SubTask 状态，复杂任务靠单 Agent 即兴推进。
- 没有真正的 Agent Profile / Selector，目前只有一个通用工具型 Agent。
- `answer` 一旦调用就终止，缺少提交前验证与修复。
- 工具失败后的恢复主要靠模型自发重试，缺少语义纠偏和局部重规划。
- 中间结果主要存在临时 Python 工作区和 trace 里，缺少可复用 artifact catalog。

因此升级路线应是：先补观测、合同和验证，再引入规划、选择和修复，最后再考虑更复杂的多 Agent / DAG / 记忆。

---

## 2. 增量开发总原则

1. 保持提交接口稳定  
   不改变 `dabench submit`、`prediction.csv`、`/input`、`/output`、`/logs` 约定。

2. 保持现有 baseline 可回退  
   新能力尽量通过配置开关接入，例如 `agent.enable_data_catalog`、`agent.enable_answer_validation`，避免一次性替换整条运行图。

3. 每个增量都要能单独评分  
   每完成一个阶段，就固定运行同一组公开任务，比较 `primary_proxy_score`、`mean_recall`、`mean_redundancy_rate`、未提交数、超时数、模型轮数和耗时。

4. 不把大表塞进 prompt  
   数据目录、计划、中间结果只传 schema、样例、统计摘要和 artifact 路径。

5. 先做硬规则，再做 LLM 判断  
   验证、工具路由、文件类型检查、列宽检查、空输出检查都应优先用确定性代码完成。

6. 计划服务于比赛任务  
   设计书建议 3 到 6 个子任务，但当前 DABench 公开任务有不少是单表或短链路问题；实际实现应允许 `1 到 4` 个子任务，复杂任务才拆得更细。

---

## 3. 固定评估协议

建议先在增量 0 中冻结一套固定评估方法，后续每个增量都按同样方式验证。

### 3.1 参考基线

仓库已有三份可参考批次，其中 `artifacts/standard/baseline` 建议作为固定的标准对比基线：

| 基线来源 | run_id | 任务数 | 有预测任务 | Primary λ=0.1 | Mean Recall | Mean Redundancy | 主要失败 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `artifacts/standard/baseline` | `20260424T011929Z` | 50 | 40 | 0.500000 | 0.500000 | 0.300000 | 10 个 max_steps 未提交 |
| `artifacts/runs/20260412T035736Z` | `20260412T035736Z` | 50 | 46 | 0.698805 | 0.710000 | 0.311952 | 4 个 max_steps 未提交 |
| `artifacts/runs/20260424T014353Z` | `20260424T014353Z` | 50 | 47 | 0.677000 | 0.690000 | 0.370000 | 2 个 max_steps，1 个 timeout |

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

## 5. 增量 1：Data Inspector 与 Data Catalog

### 目标

把“先列文件、再人工预览”的隐式过程，沉淀成一个确定性数据目录，降低模型早期摸索成本。

### 当前项目适配

不要照设计书新建独立 `main.py` 或全新 `src/core/`。建议在当前包内新增：

```text
src/data_agent_baseline/inspectors/
  __init__.py
  catalog.py
  context_inspector.py
```

同时在 `ToolRegistry` 中新增一个工具：

```text
inspect_context
```

该工具读取当前 `PublicTask.context_dir`，返回轻量摘要：

- 文件路径、类型、大小。
- CSV：列名、行数、样例行、粗略类型、缺失值数。
- JSON：顶层结构、records 数、样例 key、样例记录。
- SQLite / DB：表名、建表 SQL、列信息、样例行。
- Markdown / txt：前若干字符、标题行。

运行时还应把摘要写入任务输出目录：

```text
artifacts/runs/<run_id>/<task_id>/data_catalog.json
```

如果暂时不想把 `run_output_dir` 传进 Agent，可先只把 catalog 放进 trace，等增量 5 再统一 artifact 化。

### 可调优项

- `agent.catalog_sample_rows`：默认 5 或 10。
- `agent.catalog_max_doc_chars`：默认 2000。
- `agent.catalog_max_json_chars`：默认 4000。
- `agent.catalog_null_check_rows`：大文件只抽样统计。
- `agent.auto_inspect_context`：是否在初始 prompt 中自动注入 catalog 摘要。

### 验证方式

- 单元测试覆盖 CSV、JSON、SQLite、Markdown、空文件、坏 JSON。
- 对 `task_11` 验证 catalog 能识别 `knowledge.md`、`json/Patient.json`、`json/Examination.json`。
- 对 `task_89`、`task_180` 这类多 CSV / DB 任务，检查 catalog 是否包含关键列。
- 跑 Smoke 和语义合同风险切片，比较模型平均步数是否下降。

### 验收标准

- 不降低公开 full benchmark 主分。
- Smoke 切片平均模型轮数下降或持平。
- trace 中可以直接看到数据目录，人工复盘不需要重新打开每个文件。

---

## 6. 增量 2：题意语义合同 Answer Contract

### 目标

解决当前 Top1 问题：题意被近似字段、错误条件或错误聚合口径替代。

### 当前项目适配

新增一个轻量结构 `AnswerContract`，不要一开始就实现完整 Planner：

```text
src/data_agent_baseline/agents/contracts.py
```

建议字段：

- `target_columns`：题目最终要求的列，不是中间辅助列。
- `row_grain`：一行代表什么，例如 patient、driver、event、aggregate value。
- `expected_cardinality`：单值、少量多行、未知。
- `filters`：题面明确过滤条件。
- `metrics`：聚合、排名、计数、平均、比例等公式口径。
- `source_priority`：字段优先来自主档表、事实表、知识文档等。
- `high_risk_terms`：`per unit`、`ranked second`、`tally`、`type`、`contains` 等。
- `open_questions`：需要通过工具确认的字段映射。

实现方式建议分两步：

1. 增量 2A：只通过 prompt 强制模型在第一轮行动前生成简短合同说明，并记录到 trace。
2. 增量 2B：新增独立 LangGraph 节点 `build_answer_contract`，使用结构化 JSON / Pydantic 校验，失败时重试一次。

不要把合同写成“必须 3 到 6 子任务”。DABench 很多题最终只是一个表格答案，合同比过度拆解更重要。

### 可调优项

- `agent.enable_answer_contract`。
- `agent.contract_mode`：`prompt_only`、`structured_node`。
- `agent.contract_max_retries`：默认 1。
- `agent.high_risk_terms_path`：术语规则表路径。
- `agent.contract_in_prompt`：是否把合同摘要注入后续模型上下文。

### 验证方式

- 单元测试：给定题目文本，合同 JSON 能被解析并通过最小 schema。
- 人工检查重点任务：
  - `task_180`：`more than 29.00 per unit` 应进入高风险项，不能直接等价为 `Price > 29`。
  - `task_89`：`ranked second` 应提醒区分 `rank` 与 `position`。
  - `task_80`：最终号码字段应提醒优先确认主档来源。
  - `task_163`：`type` 应确认是事件层字段而不是预算 category。
- 跑语义合同风险切片，比较 `task_80/89/163/180/379` 的 trace 是否减少错误字段落点。

### 验收标准

- 合同节点失败不会导致整题直接失败，最多退回原 baseline。
- 合同内容进入 trace，人工可审计。
- 语义风险切片主分不下降，至少 1 到 2 个历史错误任务出现可解释改善。

---

## 7. 增量 3：轻量 Planner 与 SubTask 计划

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

## 8. 增量 4：工具路由守卫与 DuckDB 表格工具

### 目标

降低“用错工具但继续沿着错误语义跑”的概率，并增强 CSV/JSON 表格分析能力。

### 当前项目适配

当前 `execute_context_sql` 只适用于 SQLite / DB，但历史 trace 中模型常把 CSV 当 SQL 数据库查。建议：

1. 给现有工具增加文件类型守卫  
   对 CSV 调 `inspect_sqlite_schema` 或 `execute_context_sql` 时，返回明确建议：应使用 `read_csv`、`execute_python` 或新增 DuckDB 工具。

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

## 9. 增量 5：Artifact Manager 与中间结果 Catalog

### 目标

把数据目录、合同、计划、关键脚本和候选结果持久化，形成简化版 Data Catalog，提升复盘和后续验证能力。

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
  data_catalog.json
  answer_contract.json
  plan.json
  intermediate_catalog.json
  tool_artifacts/
```

建议先保存小而关键的内容：

- Data Inspector 输出。
- Answer Contract。
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

## 10. 增量 6：提交前 Answer Validator

### 目标

解决当前 Top2 问题：模型把中间结果、冗余列、错误粒度或候选集直接提交。

### 当前项目适配

当前 `answer` 工具在 `registry.py` 中校验基本结构后立即终止。建议分阶段改造：

1. 增量 6A：在 `_answer` 内增加硬规则校验，但只记录 warning，不拦截。
2. 增量 6B：当启用严格模式时，`answer` 不立即终止；若验证失败，返回非终止工具结果和修复建议，让模型再改一次。
3. 增量 6C：把验证从工具内抽成 LangGraph 节点 `validate_answer`，形成真正的提交前门禁。

硬规则建议：

- 列数和行宽合法。
- 答案列不应明显多于合同中的 `target_columns`。
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

## 11. 增量 7：失败 Refiner 与局部重试

### 目标

解决当前 Top3 问题：工具失败或验证失败后只修执行形式，不重审语义；复杂任务接近 max_steps 仍不收敛。

### 当前项目适配

在 `AgentGraphState` 中新增有限状态：

- `retry_count_by_reason`
- `last_validation_result`
- `last_tool_error`
- `semantic_contract_revision`
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
  -> 如果重复失败，要求模型重述语义合同

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

## 12. 增量 8：Agent Profiles 与规则 Selector

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

## 13. 增量 9：显式 LangGraph 工作流重构

### 目标

把前面已经稳定的能力整理成真正的 Data Agent 工作流，而不是继续塞进单个模型循环。

### 当前项目适配

在当前 `langgraph_runtime.py` 基础上演进，不重写 runner 和 CLI。目标图：

```text
START
  -> init_state
  -> inspect_data
  -> build_answer_contract
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

- `data_catalog`
- `answer_contract`
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

- `agent.workflow_mode`：`react_baseline`、`contract_planned`、`full_data_agent`。
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
- `trace.json` 能清晰呈现 inspect、contract、plan、select、execute、verify、refine。
- Full public 主分持平或提升，未提交数和冗余率至少一项改善。

---

## 14. 增量 10：历史表现记忆与 Benchmark 辅助选择

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

## 15. 增量 11：局部重规划与 DAG/并行执行

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

## 16. 推荐实施顺序

优先级从高到低：

```text
0. 冻结评估协议
1. Data Catalog
2. Answer Contract
4. 工具路由守卫 + DuckDB 工具
6. Answer Validator
5. Artifact Manager
7. Refiner / 强制收敛
3. Planner
8. Agent Profiles / Selector
9. 显式 LangGraph 工作流
10. 历史表现记忆
11. 局部重规划 / DAG
```

说明：

- Planner 在概念上很重要，但当前失败更集中在题意合同、工具误用和答案收敛；因此可以先实现合同和验证，再做完整计划。
- Artifact Manager 可以和多个阶段交叉推进，但不要让它阻塞早期质量提升。
- 多 Agent / Selector 不应过早复杂化；先用 profile prompt 和工具子集模拟即可。

---

## 17. 每个增量的记录模板

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

## 18. 近期最小可行路线

如果只做第一轮高性价比升级，建议按下面 5 步：

1. 增量 0：冻结评估切片和 baseline。
2. 增量 1：新增 `inspect_context` 和 `data_catalog`。
3. 增量 2：新增 `AnswerContract`，先用 prompt-only，再改成结构化节点。
4. 增量 4：新增工具路由守卫和 DuckDB 表格 SQL 工具。
5. 增量 6：在 `answer` 前加硬规则 validator，先 warning，再 repair_once。

这 5 步直接对应当前复盘中的三个高频短板：

- 题意语义锚定缺失。
- 工具误用后恢复不够聪明。
- 最终答案未收敛、冗余列伤分。

等这 5 步稳定后，再把 Planner、Selector、Refiner 正式纳入 LangGraph 状态，项目就会从“单 Agent 工具循环 baseline”自然演进成文档目标中的轻量级 Data Agent。
