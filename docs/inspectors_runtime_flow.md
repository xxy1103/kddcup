# Inspectors 层完整运行流程

本文档说明当前 `src/data_agent_baseline/inspectors` 层在一次任务运行中的完整流程。读完后，应能够理解 inspectors 如何介入主 Agent、每个阶段产出什么、配置项如何影响行为，以及如何通过 artifacts 调试。

## 1. 总览

当前 inspectors 层不是单个检查器，而是主解题 Agent 前置的一条数据理解流水线。它的目标不是计算最终答案，而是在主 Agent 真正调用工具解题前，先把题目语义、数据资产、字段候选、join 路径、输出契约和风险整理成结构化交接文档。

整体流程如下：

```text
runner
  -> LangGraphAgent
    -> init_state
    -> perceive_task
       -> PerceptionAgent: 只读题目，抽取实体、指标、过滤条件、答案形状和风险
    -> understand_and_explore_data
       -> DataUnderstandingAgent
          -> build_semantic_catalog: 扫描数据目录
          -> build_semantic_index: 建立字段/文件/别名/风险索引
          -> SemanticQueryTools.build_context_bundle: 汇总字段候选、schema、知识片段、join 路径
          -> deterministic_handoff: 规则版 handoff
          -> GuidedDataUnderstandingLoop: hybrid 模式下的分阶段 LLM 数据理解
          -> semantic_context_envelope: 封装交给主 Agent 的语义上下文
    -> model_step / tool_step
       -> 主 Agent 根据 handoff 真正计算答案
    -> finalize
```

如果 `agent.enable_data_inspector=false`，`perceive_task` 和 `understand_and_explore_data` 会直接跳过，主 Agent 按普通 ReAct 流程运行。

## 2. 运行入口

单任务入口在 `src/data_agent_baseline/run/runner.py` 的 `_run_single_task_core`。

核心逻辑是：

```python
agent = LangGraphAgent(
    model=model or build_chat_model(config),
    tools=tools or create_default_tool_registry(),
    config=LangGraphAgentConfig(
        max_steps=config.agent.max_steps,
        enable_data_inspector=config.agent.enable_data_inspector,
        data_inspector=config.data_inspector,
    ),
)
run_result = agent.run(task)
```

也就是说，inspectors 层是否启用由 `agent.enable_data_inspector` 控制，inspectors 内部行为由 `data_inspector` 配置控制。

当前 `configs/eval_contract.yaml` 中 relevant 配置为：

```yaml
agent:
  enable_data_inspector: true

data_inspector:
  mode: hybrid
  inject_summary_to_agent: true
  max_agent_steps: 5
  max_phase_retries: 1
  enable_semantic_tools: true
  include_inspector_trace: true
  context_bundle_limit: 6
  max_join_hops: 3
  sample_budget:
    catalog_sample_rows: 6
    max_doc_chars: 2000
    max_json_chars: 4000
```

因此当前运行会启用完整 inspectors 流程，并在 `hybrid` 模式下执行规则扫描加 LLM 分阶段理解。

## 3. LangGraph 中 inspectors 的位置

inspectors 挂在 `src/data_agent_baseline/agents/langgraph_runtime.py` 中。

图节点顺序是：

```text
START
  -> init_state
  -> perceive_task
  -> understand_and_explore_data
  -> model_step
  -> tool_step / react_step / repair_step
  -> finalize
  -> END
```

其中：

- `perceive_task` 调用 `invoke_perception_agent`。
- `understand_and_explore_data` 调用 `DataUnderstandingAgent.run`。
- `model_step` 之后才进入主 Agent 的工具调用和答案生成循环。

这意味着 inspectors 的产物会在主 Agent 第一次思考前注入，而不是在主 Agent 运行中途补充。

## 4. 第一阶段：PerceptionAgent

代码位置：`src/data_agent_baseline/inspectors/perception.py`

`perceive_task` 会调用：

```python
perception_result = invoke_perception_agent(task, self.model)
perception_envelope = perception_result.envelope
```

### 4.1 输入

PerceptionAgent 只读取题目级信息：

```json
{
  "task_id": "...",
  "difficulty": "...",
  "question": "..."
}
```

它不会扫描文件，不会选字段，不会计算答案。

### 4.2 输出 JSON

模型必须返回一个 JSON 对象，对应 `PerceptionDraft`：

```json
{
  "question": "exact original question string",
  "difficulty": "task difficulty string",
  "entities": [],
  "metrics": [],
  "filter_phrases": [],
  "expected_answer_shape": {
    "row_shape": "single_row|multiple_rows|unknown",
    "column_hint": "short description",
    "only_requested_columns": true
  },
  "high_risk_terms": []
}
```

字段含义：

- `entities`: 题目涉及或要求输出的实体，例如 event、country、school。
- `metrics`: 题目显式要求的指标或操作，例如 cost、average、count。
- `filter_phrases`: 完整过滤短语，例如日期、状态、命名实体、范围限定。
- `expected_answer_shape`: 预期答案行列形态。
- `high_risk_terms`: 可能改变答案的风险点，例如 `aggregation_grain`、`join_key_ambiguity`、`tie_handling_ambiguity`。

### 4.3 重试机制

`invoke_perception_agent` 内部最多尝试两次：

1. 第一次使用正常 perception prompt。
2. 如果输出为空、不是合法 JSON、schema 校验失败，则构造 retry prompt。

第二次仍失败时抛出 `PerceptionBuildError`。

LangGraph 捕获该错误后，会把 `inspector` 设置为：

```json
{
  "error": "Perception failed: ..."
}
```

后续 `understand_and_explore_data` 会跳过，但主 Agent 仍会继续执行。

### 4.4 AgentEnvelope

成功后，PerceptionAgent 输出一个 `AgentEnvelope`：

```json
{
  "protocol_version": "agent-exchange/v1",
  "task_id": "...",
  "sender": "PerceptionAgent",
  "recipient": "DataUnderstandingAgent",
  "message_type": "perception_result",
  "content": {
    "summary": "...",
    "semantic_claims": [],
    "uncertainties": [],
    "payload": {}
  }
}
```

其中 `payload` 就是标准化后的 perception JSON。

## 5. 第二阶段：DataUnderstandingAgent 总流程

代码位置：`src/data_agent_baseline/inspectors/data_understanding_agent.py`

入口：

```python
result = understanding_agent.run(task, perception_envelope)
```

`run()` 的主要流程为：

```text
1. perception_payload = perception_envelope.content.payload
2. catalog = build_semantic_catalog(task)
3. semantic_index = build_semantic_index(question, catalog, perception_payload)
4. query_tools = SemanticQueryTools(...)
5. context_bundle = query_tools.build_context_bundle(question)
6. deterministic_handoff = _build_handoff(...)
7. 如果 mode=hybrid，执行 GuidedDataUnderstandingLoop
8. 构造 semantic_context_envelope
9. 返回 DataUnderstandingResult
```

最终返回的 `DataUnderstandingResult` 包含：

```python
perception
semantic_catalog
semantic_index
perception_envelope
semantic_context_envelope
data_understanding_handoff
summary
synthesis
synthesis_error
inspector_steps
handoff_status
validation_errors
```

## 6. Semantic Catalog：扫描数据目录

代码位置：`src/data_agent_baseline/inspectors/semantic_catalog.py`

调用：

```python
catalog = build_semantic_catalog(task, budget=self.config.sample_budget)
```

它遍历 `task.context_dir` 下的所有文件，根据后缀识别资产类型：

```text
.csv                  -> csv
.json                 -> json
.db/.sqlite/.sqlite3  -> sqlite
.md/.txt/.rst         -> document
其他                  -> file
```

### 6.1 CSV

CSV 扫描会读取：

- header
- row_count
- sample_rows
- 每列 sample_values
- 每列 missing_count
- 简单类型推断：`integer`、`number`、`string`、`unknown`

### 6.2 JSON

JSON 扫描会：

- 读取并解析 JSON。
- 将嵌套结构扁平化成字段路径，例如 `records.event_id`。
- 推断字段类型。
- 保存 preview，长度受 `sample_budget.max_json_chars` 控制。

### 6.3 SQLite

SQLite 扫描会：

- 以只读方式连接数据库。
- 枚举表。
- 读取 `PRAGMA table_info`。
- 采样若干行。

### 6.4 Document

文档扫描会：

- 读取文本内容。
- 抽取 Markdown heading。
- 保存全文和 preview，preview 长度受 `sample_budget.max_doc_chars` 控制。

### 6.5 Catalog 输出结构

catalog 主要结构：

```json
{
  "task_id": "...",
  "assets": [],
  "schemas": [],
  "semantic_entities": [],
  "field_meanings": [],
  "relationships": [],
  "query_relevance": {},
  "semantic_uncertainties": []
}
```

其中：

- `assets`: 文件级资产列表，包含 path、kind、size、recommended_tools。
- `schemas`: 各资产 schema 和样例。
- `relationships`: 基于字段名规则发现的 join 候选。
- `query_relevance`: 题目 token 与文件/字段 token 的关键词相关性。
- `semantic_uncertainties`: 扫描失败或无法解析的风险。

## 7. Semantic Index：建立倒排索引

代码位置：`src/data_agent_baseline/inspectors/semantic_index.py`

调用：

```python
semantic_index = build_semantic_index(
    question=task.question,
    catalog=catalog,
    perception_payload=perception_payload,
)
```

索引包含：

```json
{
  "file_token_index": {},
  "field_token_index": {},
  "field_alias_index": {},
  "query_index": {},
  "risk_index": {},
  "tool_routing_index": {}
}
```

### 7.1 file_token_index

把文件路径 tokenize，建立：

```text
token -> 文件资产
```

### 7.2 field_token_index

把字段名 tokenize，建立：

```text
token -> 字段引用
```

字段引用大致形态：

```text
csv/example.csv.column
json/event.json.records.event_id
db/example.db.table.column
```

### 7.3 field_alias_index

根据字段名规则添加语义别名。例如：

- 字段名含 `id`: 添加 `id`、`identifier`、`key`、`join`
- 字段名含 `time`: 添加 `time`、`finish`、`duration`
- 字段名含 `rank`: 添加 `rank`、`ranked`、`placing`
- 字段名含 `city/state/region/county`: 添加 `geography`、`scope`
- 字段名含 `average/avg/mean`: 添加 `metric`
- 字段名含 `sum/total/count/score/amount/cost`: 添加 `metric`、`operation`

### 7.4 risk_index

基于 PerceptionAgent 给出的 `high_risk_terms`，为风险建立候选字段集合。

例如：

- `join_key_ambiguity`: 返回 id/key/join 相关字段。
- `geographic_scope_ambiguity`: 返回 county/city/state/region 等字段。
- `aggregation_grain`: 返回 metric/operation/average/count/total 等字段。

## 8. SemanticQueryTools：构造 context bundle

代码位置：`src/data_agent_baseline/inspectors/semantic_query.py`

DataUnderstandingAgent 会构造：

```python
query_tools = SemanticQueryTools(
    catalog=catalog,
    semantic_index=semantic_index,
    limit=self.config.context_bundle_limit,
    max_join_hops=self.config.max_join_hops,
)
```

当前配置中：

```yaml
context_bundle_limit: 6
max_join_hops: 3
```

### 8.1 可用工具

`SemanticQueryTools` 支持：

```text
search_semantic_index(query)
lookup_knowledge(term)
get_asset_schema(asset_path, include_samples=False)
find_join_paths(source, target)
build_context_bundle(query)
known_field_refs()
```

这些是 DataUnderstandingAgent 内部工具，不是主 Agent 直接调用的工具。

### 8.2 context bundle

当 `enable_semantic_tools=true` 时，会调用：

```python
context_bundle = query_tools.build_context_bundle(task.question)
```

输出：

```json
{
  "query": "...",
  "field_candidates": [],
  "asset_schemas": [],
  "knowledge_hits": [],
  "join_paths": [],
  "risk_candidates": {},
  "rejected_field_hints": []
}
```

字段含义：

- `field_candidates`: 题目命中的字段候选，来自字段索引和 alias 索引。
- `asset_schemas`: 相关资产的简化 schema。
- `knowledge_hits`: 文档资产中的相关片段。
- `join_paths`: 根据字段关系图搜索出的 join 路径。
- `risk_candidates`: 高风险项对应的候选字段。
- `rejected_field_hints`: 明确不建议使用的字段提示。

### 8.3 Join Path 搜索

`find_join_paths` 会先构造字段关系图，再用 BFS 查找 source 到 target 的路径。

关系图中包含两类边：

1. 跨资产同名或近似同名 id 字段。
2. 同一资产中的 relationship 字段桥接，例如 `link_to_xxx` 或 `id` 字段。

搜索最大跳数由 `max_join_hops` 控制。

## 9. deterministic_handoff：规则版交接文档

DataUnderstandingAgent 一定会先构造规则版 handoff：

```python
deterministic_handoff = _build_handoff(...)
```

这一步不调用模型，主要由以下函数组成：

```text
_build_grounded_concepts
_build_join_paths
_build_answer_contract
_build_handoff_uncertainties
```

### 9.1 grounded concepts

当前规则较朴素，主要处理一些常见 token：

- 题目中有 `event`：建立 `event` 概念，角色为 `answer_entity`。
- 题目中有 `cost`：建立 `cost` 概念，角色为 `metric`。
- 题目中有 `lowest/minimum/min`：建立 `lowest` 操作概念。
- 题目中有 `highest/maximum/max`：建立 `highest` 操作概念。

字段 grounding 来自 `context_bundle.field_candidates`。

### 9.2 answer contract

规则版 contract 会推断：

- `metric_operation`: `min`、`max` 或 `unknown`
- `row_policy`: 最值题默认 `preserve_all_ties`
- `columns`: 如果题目包含 `event`，默认输出 `event_name`
- `metric_field`: 从 metric concept 的 grounded fields 中选择第一个

### 9.3 uncertainties

uncertainties 来自：

- PerceptionAgent 的 `high_risk_terms`
- 缺少 join path 时添加 `missing_join_path`
- 可选 LLM notes 中的不确定性

### 9.4 inspector trace

规则阶段会写入一条 inspector step：

```json
{
  "phase": "deterministic_context",
  "prompt_type": "rules",
  "accepted_draft": {
    "asset_count": 4,
    "schema_count": 4,
    "field_candidate_count": 6,
    "join_path_count": 6
  }
}
```

## 10. hybrid 模式：GuidedDataUnderstandingLoop

当前配置为：

```yaml
mode: hybrid
max_agent_steps: 5
max_phase_retries: 1
```

因此会启动 `GuidedDataUnderstandingLoop`。

### 10.1 阶段顺序

正常阶段为：

```text
overview
grounding
fabric
contract
```

如果 handoff 质量校验失败且仍有模型调用预算，则额外执行：

```text
repair_or_critique
```

### 10.2 max_agent_steps

`max_agent_steps` 控制 inspectors 内部 LLM 调用总预算。

当前为 5，所以理想情况下：

```text
overview   1 次
grounding  1 次
fabric     1 次
contract   1 次
repair     1 次
```

如果某个阶段第一次输出失败，并触发 retry，就会额外消耗一次预算，可能导致后续 repair 没有预算。

### 10.3 max_phase_retries

`max_phase_retries=1` 表示每个阶段最多允许一次重试。

每个阶段总尝试次数是：

```text
max_phase_retries + 1 = 2
```

### 10.4 字段白名单

hybrid loop 会生成字段白名单：

```python
field_whitelist = sorted(query_tools.known_field_refs())
```

模型在任意阶段返回的：

```text
field_ref
from_field
to_field
```

都必须严格来自该白名单，否则 `_validate_draft_field_refs` 会拒绝输出。

这是避免 LLM 编造字段的关键约束。

### 10.5 overview 阶段

目标：

- 判断任务意图。
- 抽取概念。
- 标记歧义目标。
- 请求必要的 semantic tools。

输出 schema：

```json
{
  "task_intent": "string",
  "concepts": [
    {
      "term": "string",
      "role": "answer_entity|metric|filter|time|operation|unknown"
    }
  ],
  "ambiguity_targets": [],
  "tool_requests": []
}
```

### 10.6 grounding 阶段

目标：

- 把题目概念落到具体字段。
- 给出 accepted fields。
- 给出 rejected fields 和原因。

输出 schema：

```json
{
  "grounded_concepts": [
    {
      "term": "string",
      "role": "answer_entity|metric|filter|time|operation|unknown",
      "accepted_fields": [
        {
          "field_ref": "exact allowed_field_refs item",
          "confidence": "high|medium|low",
          "reason": "string"
        }
      ],
      "rejected_fields": [
        {
          "field_ref": "exact allowed_field_refs item",
          "reason": "string"
        }
      ]
    }
  ],
  "remaining_uncertainties": [],
  "tool_requests": []
}
```

校验规则：

- 所有字段必须在白名单中。
- 同一字段不能同时出现在 accepted 和 rejected。

### 10.7 fabric 阶段

目标：

- 描述 join path。
- 描述数据粒度。
- 描述关系风险。

输出 schema：

```json
{
  "join_paths": [
    {
      "purpose": "string",
      "path": [
        {
          "from_field": "exact allowed_field_refs item",
          "to_field": "exact allowed_field_refs item"
        }
      ],
      "confidence": "high|medium|low"
    }
  ],
  "data_grain": "string",
  "relationship_risks": [],
  "tool_requests": []
}
```

### 10.8 contract 阶段

目标：

生成主 Agent 最需要的答案契约。

输出 schema：

```json
{
  "columns": [],
  "filters": [],
  "group_by": [],
  "metric_operation": "min|max|sum|count|average|lookup|unknown",
  "metric_fields": [],
  "row_policy": "single|multiple|preserve_all_ties|unknown",
  "distinct_policy": "preserve|deduplicate|unknown",
  "output_grain": "string",
  "remaining_uncertainties": []
}
```

校验规则：

- `metric_fields`、`group_by` 中字段必须来自白名单。
- contract 不能使用 grounding 阶段明确 rejected 的字段。

### 10.9 repair_or_critique 阶段

当 `_validate_handoff_quality` 发现 handoff 质量问题，并且仍有模型调用预算时，会进入 repair 阶段。

它会收到：

- 当前 working memory。
- 当前 handoff。
- validation errors。
- 最近 tool observations。

然后只修补失败或缺失部分。

## 11. Guided handoff 组装

LLM 阶段输出的是 draft，不直接作为最终 handoff。代码会调用：

```python
handoff = _build_guided_handoff(...)
```

将 draft 转成正式 `DataUnderstandingHandoff`。

结构定义在 `src/data_agent_baseline/inspectors/handoff.py`：

```python
class DataUnderstandingHandoff(BaseModel):
    brief_markdown: str
    question_grounding: QuestionGrounding
    data_fabric: DataFabric
    answer_contract: AnswerContract
    uncertainties: list[HandoffUncertainty]
    llm_notes: list[str]
    llm_notes_error: str | None
    handoff_status: Literal["complete", "partial", "fallback"]
    validation_errors: list[str]
```

核心子结构：

```text
QuestionGrounding
  -> concepts
     -> term
     -> role
     -> grounded_fields
     -> rejected_fields

DataFabric
  -> join_paths
     -> source
     -> target
     -> path

AnswerContract
  -> columns
  -> row_policy
  -> metric_operation
  -> metric_field
  -> filters
  -> group_by
  -> metric_fields
  -> output_grain
  -> distinct_policy
```

## 12. handoff 质量校验和状态

生成 handoff 后，会调用：

```python
validation_errors = _validate_handoff_quality(task.question, handoff, context_bundle)
```

这不是答案正确性校验，而是结构完整性和明显风险检查。

当前检查包括：

- 有字段候选但 `question_grounding.concepts` 为空。
- 题目明显要求输出列，但 `answer_contract.columns` 为空。
- 题目有 aggregate/extreme 词，但 `metric_operation` 仍为 `unknown`。
- 题目是 metric 问题，但 `metric_fields` 为空。
- 题目看起来有 filter，但没有 filters，也没有 filter/time grounding。

状态含义：

```text
complete:
  handoff 通过质量校验。

partial:
  handoff 可用，但仍有 validation warnings。

fallback:
  guided loop 失败，退回 deterministic_handoff。
```

主 Agent 收到不同状态时，注入提示也不同：

- `complete`: 把 handoff 当作可信指导，默认不重新验证。
- `partial/fallback`: 把 handoff 当作候选路线，最终答案前需要解决 warnings。

## 13. Semantic Context Envelope

DataUnderstandingAgent 最后会构造：

```python
semantic_context_envelope = self._build_semantic_context_envelope(...)
```

这是交给主 Agent 的 envelope：

```json
{
  "sender": "DataUnderstandingAgent",
  "recipient": "MainSolvingAgent",
  "message_type": "semantic_context",
  "content": {
    "summary": "...",
    "resources": [],
    "artifacts": [],
    "semantic_claims": [],
    "uncertainties": [],
    "payload": {
      "catalog_asset_count": 0,
      "catalog_schema_count": 0,
      "answer_contract": {},
      "handoff": {}
    }
  }
}
```

其中 `summary` 就是 `handoff.brief_markdown`。

`artifacts` 固定包含：

```text
perception.json
semantic_catalog.json
semantic_index.json
data_understanding_handoff.json
```

## 14. 注入主 Agent

如果配置：

```yaml
inject_summary_to_agent: true
```

则 `understand_and_explore_data` 会向主 Agent 的 messages 追加一个 `HumanMessage`。

注入内容包含：

```text
1. Data Understanding Brief
2. Full data_understanding_handoff.json
3. 主 Agent 使用说明
```

如果 `handoff_status == complete`，注入提示大意是：

```text
Treat this handoff as trusted guidance from a separate data understanding agent.
Use the full JSON for structured fields, join paths, answer contract, rejected fields, and uncertainties.
Do not re-verify it by default; call tools only to compute the requested result, resolve uncertainty, or investigate a clear conflict.
```

如果 handoff 是 `partial` 或 `fallback`，提示主 Agent：

```text
Treat this partial/fallback handoff as a candidate route.
Use the JSON to focus exploration, but resolve validation warnings before finalizing the answer.
```

之后主 Agent 进入正常 `model_step`，开始调用实际工具计算答案。

## 15. 输出 artifacts

runner 会把 inspectors 结果写到每个 task 的输出目录。

代码位置：`src/data_agent_baseline/run/runner.py` 的 `_write_task_outputs`。

如果 `run_result["inspector"]` 是 dict，会写：

```text
perception.json
semantic_catalog.json
semantic_index.json
data_understanding_handoff.json
data_understanding_trace.json
```

另外完整运行结果在：

```text
trace.json
```

各文件用途：

| 文件 | 用途 |
| --- | --- |
| `perception.json` | 题目语义初步理解 |
| `semantic_catalog.json` | 数据资产扫描结果 |
| `semantic_index.json` | 字段、文件、别名、风险索引 |
| `data_understanding_handoff.json` | 给主 Agent 的核心交接文档 |
| `data_understanding_trace.json` | inspectors 内部分阶段 trace |
| `trace.json` | 完整 LangGraph 运行记录 |

## 16. inspector_steps 怎么读

`data_understanding_trace.json` 里的 `inspector_steps` 是理解 inspectors 行为的关键。

常见阶段：

```text
deterministic_context
overview
overview_tools
grounding
grounding_tools
fabric
fabric_tools
contract
repair_or_critique
fallback
```

每个 step 结构：

```json
{
  "phase": "overview",
  "prompt_type": "phase|retry|tool|rules|error",
  "tool_requests": [],
  "tool_results": [],
  "validation_error": null,
  "accepted_draft": {}
}
```

排查建议：

- 如果 `validation_error` 不为空，说明该阶段模型输出 JSON 或字段引用不合格。
- 如果出现 `*_tools`，说明上一阶段请求了内部 semantic tools。
- 如果最终 `handoff_status=fallback`，看 `fallback` 阶段和前面的 validation error。
- 如果主 Agent 答案错了，优先看 `answer_contract` 和 `rejected_fields` 是否误导。

## 17. 配置项影响总结

### mode

```yaml
mode: rules
```

只使用 deterministic handoff，不跑 guided LLM loop。

```yaml
mode: hybrid
```

先跑 deterministic handoff，再跑 guided LLM loop 增强。

### inject_summary_to_agent

控制是否把 `brief_markdown` 和完整 `data_understanding_handoff.json` 注入主 Agent。

关闭后 inspectors 仍会运行并写 artifacts，但主 Agent 不会看到这份 handoff。

### max_agent_steps

控制 DataUnderstandingAgent 内部 guided loop 的模型调用总数。

正常四阶段需要 4 次。若希望 repair 更稳定，至少需要 5 次。

### max_phase_retries

控制每个 guided phase 的 JSON/校验失败重试次数。

### enable_semantic_tools

开启时使用 `SemanticQueryTools.build_context_bundle`，并允许 guided loop 请求内部 semantic tools。

关闭时只使用 catalog 中的粗略 query relevance 和 relationships。

### include_inspector_trace

控制 LangGraph step 里是否包含完整 `inspector_steps`。

注意：runner 写出的 `data_understanding_trace.json` 当前仍会使用 `inspector.get("inspector_steps", [])`。

### context_bundle_limit

控制 context bundle 的候选数量，例如字段候选、schema、knowledge hits、join paths。

### max_join_hops

控制内部 join path 搜索最大跳数。

### sample_budget

控制扫描资产时保留多少样例：

```yaml
sample_budget:
  catalog_sample_rows: 6
  max_doc_chars: 2000
  max_json_chars: 4000
```

## 18. 一句话理解

inspectors 层可以概括为：

```text
PerceptionAgent 先读懂题目；
SemanticCatalog/Index 读懂数据目录；
SemanticQueryTools 把题目和数据连接起来；
DataUnderstandingAgent 把这些信息压缩成结构化 handoff；
主 Agent 带着 handoff 去真正执行工具计算并提交答案。
```

