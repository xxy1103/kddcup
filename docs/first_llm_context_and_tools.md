# 首次 LLM 调用上下文与新增工具说明

本文记录当前分支中主 Agent 第一次调用 LLM 时的上下文结构，以及本轮新增/调整的工具能力。

## 一、首次 LLM 调用前的执行流程

主 Agent 入口仍然是 `LangGraphAgent.run()`。在第一次进入 `model_step` 前，图执行顺序为：

1. `init_state`
   - 创建 `SystemMessage`
   - 使用 `prompt.py` 中的 system prompt

2. `global_data_exploration`
   - 如果 `enable_data_inspector=true`，构建完整 `semantic_catalog`
   - 完整 catalog 会保存到 trace/artifacts 的 `semantic_catalog.json`
   - 同时生成一个轻量版 `global_data_profile`

3. `analyze_ambiguity`
   - 仅当 `enable_ambiguity_analysis=true` 时运行
   - 默认配置一般关闭

4. `receive_problem`
   - 创建首次 `HumanMessage`
   - 注入用户问题、轻量 catalog、可选 ambiguity analysis、视频 timeline 文本和 stable frame 路径

5. `model_step`
   - 将当前 messages 与工具 schema 一起传给模型

## 二、首次 LLM 调用时的上下文组成

第一次 LLM 请求主要由三部分组成：

1. `SystemMessage`
2. `HumanMessage`
3. Tool schemas

### 1. SystemMessage

来自 `src/data_agent_baseline/agents/prompt.py`。

主要内容包括：

- Agent 必须使用工具，不能臆造工具输出
- 结构化数据统一视为逻辑 SQL 表
- 文档和图片路径仍然相对 task context
- `knowledge.md` 是重要语义依据
- 轻量 catalog 只用于定位，不等于完整证据
- 字段画像、Top values、min/max、关系推断需要用 semantic catalog 工具按需查询
- 结构化数据验证使用 `execute_probe_query`
- 图片证据使用 `read_context_image` 按需读取

### 2. HumanMessage

由 `src/data_agent_baseline/agents/langgraph_runtime.py` 中的 `receive_problem()` 构造。

结构大致如下：

```xml
<user_query>
User Question: ...
</user_query>

<context_injection>
...preamble...

<lightweight_catalog>
{
  "task_id": "...",
  "structured_tables": [...],
  "documents": [...],
  "media": [...],
  "knowledge_documents": [...],
  "semantic_uncertainties": [...]
}
</lightweight_catalog>

<ambiguity_analysis>
...仅 enable_ambiguity_analysis=true 时存在...
</ambiguity_analysis>
</context_injection>

<action_trigger>
Based on ...
</action_trigger>

<video_context>
...仅任务含视频预处理结果时存在...
</video_context>
```

### 3. Tool schemas

首次模型调用通过 `model.bind_tools(...)` 绑定工具，因此工具名称、描述、参数 schema 也会进入模型输入。

当前默认工具包括：

- `answer`
- `execute_probe_query`
- `execute_python`
- `get_column_distinct_values`
- `search_semantic_catalog`
- `get_table_profile`
- `get_field_profile`
- `get_table_relationships`
- `list_context`
- `lookup_doc_outline`
- `read_doc`
- `search_doc`
- `read_context_image`

`execute_context_sql` 不再默认暴露给主 Agent。SQLite 也通过逻辑表走 `execute_probe_query`。

## 三、轻量 catalog 的内容

轻量 catalog 由完整 `semantic_catalog` 投影生成，不再包含完整字段画像。

### structured_tables

面向 Agent 的结构化数据目录。

每个结构化数据源都被抽象成逻辑表：

```json
{
  "table": "lc_freefloat",
  "row_count": 3119,
  "columns": [
    {"name": "id", "type": "integer"},
    {"name": "CompanyCode", "type": "integer"},
    {"name": "AFloats", "type": "number"}
  ]
}
```

注意：

- Agent 首轮看不到 CSV/JSON/SQLite 底层路径
- Agent 使用 `table` 字段作为 SQL 表名
- 不包含 `distinct_values`
- 不包含 `cardinality`
- 不包含 `min_value/max_value`
- 不包含 inferred relationships

### documents

普通文档列表，包含路径和 headings。

文档路径仍然暴露，因为 `read_doc/search_doc/lookup_doc_outline` 需要路径。

### media

图片等媒体路径列表。

视频 stable frames 只在首轮列出路径，不直接附图片内容。

### knowledge_documents

`knowledge.md` 会全文注入。

这是当前首轮上下文中仍然可能较大的部分，但它是语义规则、字段含义、单位和口径的关键来源。

### semantic_uncertainties

保留轻量扫描中的解析失败或风险提示，方便模型知道某些资产可能需要额外检查。

## 四、视频上下文

视频预处理后，首轮上下文会包含：

- timeline markdown 全文
- stable frame 图片路径列表

首轮不会再附加图片 base64。

模型如果需要查看某张稳定帧，应调用：

```json
{
  "path": "video/briefing_stable_frames/stable_005_t0034.00s.jpg",
  "detail": "high"
}
```

工具 `read_context_image` 会把图片作为多模态内容追加到下一轮模型请求中。trace 中只保存图片路径、mime type、size 和状态，不保存 base64。

## 五、新增工具说明

### search_semantic_catalog

用途：在完整 semantic catalog 中搜索候选表、字段、文档、关系或不确定性。

参数：

```json
{
  "query": "AFloats",
  "scope": "all",
  "limit": 20
}
```

`scope` 可选：

- `all`
- `tables`
- `fields`
- `documents`
- `relationships`
- `uncertainties`

适合场景：

- 不确定某个业务词对应哪个字段
- 想快速搜索同名/近似字段
- 想查某个实体是否出现在关系推断中

### get_table_profile

用途：查看某个逻辑表的完整语义画像。

参数：

```json
{
  "table": "lc_freefloat"
}
```

返回内容包括：

- table
- kind
- row_count
- fields
- 每个字段的 type
- missing_count
- cardinality
- distinct_values
- min_value/max_value

适合场景：

- 首轮轻量 catalog 只看到列名，需要确认字段类型和取值分布
- 需要判断某个字段是否适合过滤、join、聚合

### get_field_profile

用途：查看某个逻辑表中单个字段的完整画像。

参数：

```json
{
  "table": "lc_freefloat",
  "column": "AFloats"
}
```

适合场景：

- 只关心一个字段，不想加载整张表的字段画像
- 需要查看 Top values、cardinality、min/max
- 需要确认字段单位或数值范围

### get_table_relationships

用途：查看某张逻辑表相关的 inferred/explicit relationships。

参数：

```json
{
  "table": "lc_freefloat"
}
```

返回内容包括：

- source_table
- source_fields
- target_table
- target_fields
- relationship_type
- cardinality
- confidence
- evidence

适合场景：

- 需要 join 但不确定 join key
- 需要确认关系是显式外键还是推断关系
- 需要比较多个候选 join 路径

### read_context_image

用途：按需读取 context 中的图片，并把图片作为多模态输入传给下一轮模型。

参数：

```json
{
  "path": "video/briefing_stable_frames/stable_005_t0034.00s.jpg",
  "detail": "high"
}
```

支持格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

trace 中返回：

```json
{
  "path": "...",
  "mime_type": "image/jpeg",
  "size": 49188,
  "status": "image attached to next model request"
}
```

实际图片内容不会写入 trace。

## 六、调整后的既有工具

### execute_probe_query

现在面向逻辑表，而不是底层文件。

示例：

```json
{
  "queries": [
    "SELECT COUNT(*) FROM lc_freefloat",
    "SELECT MAX(AFloats) FROM lc_freefloat"
  ],
  "limit": 10
}
```

新增行为：

- 会自动跳过空 query
- 会自动移除 query 开头的 `-- ...` 注释行
- 仍然只允许 `SELECT` / `WITH`

这修复了模型生成带中文 SQL 注释时被只读校验误拒绝的问题。

### get_column_distinct_values

现在也使用逻辑表名。

示例：

```json
{
  "table": "lc_freefloat",
  "column": "ChangeDate",
  "top_n": 20
}
```

它执行实时计算，不依赖首轮轻量 catalog 中的字段列表。

### execute_python

当前仍是旧机制：在临时 context workspace 中执行 Python。

注意：它还没有完全统一到逻辑表抽象。也就是说，当前版本中：

- SQL 工具已经统一为逻辑表
- semantic catalog 工具已经统一为逻辑表
- Python 工具仍可能让模型回到底层文件路径

如果后续要彻底隐藏结构化文件路径，需要给 Python 环境注入类似：

```python
query("SELECT ... FROM logical_table")
read_table("logical_table")
tables()
schema("logical_table")
```

## 七、当前 trace 中可观察到的效果

以 `artifacts/runs/20260604T132944Z/task_1/trace.json` 为例：

- 完整 `semantic_catalog.json` 约 373k 字符
- 首轮 `global_data_profile.json` 约 44.8k 字符
- 首轮不包含 `distinct_values`
- 首轮不包含 `relationships`
- 首轮不包含结构化数据的 `asset_path`
- 首轮模型 input tokens 约 21.7k
- 模型按需调用了 `read_context_image`
- 模型按需调用了 `get_table_profile`
- 结构化查询使用了逻辑表名 `lc_freefloat`、`lc_exgindustry`

这说明“完整 catalog 保留，但首轮只注入轻量视图”的目标已经生效。
