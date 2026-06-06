## submit_tool_result 设计方案

### 设计目标

使用 `submit_tool_result` 作为唯一的最终提交工具。模型在提交答案时指定一个数据工具及其参数，由系统实时执行该工具并将完整结果转化为答案。

### 典型使用场景

模型经过一系列探索后，已经构造出一条精确的 SQL 查询（或一段 Python 脚本），其输出恰好就是最终答案。此时模型直接调用：

```json
// 场景 1：用 SQL 查询结果直接提交
{
  "tool_name": "execute_probe_query",
  "tool_args": {
    "queries": ["SELECT name, amount FROM orders WHERE status = 'pending'"],
    "limit": 200
  }
}

// 场景 2：用 Python 计算结果提交（约定 print JSON）
{
  "tool_name": "execute_python",
  "tool_args": {
    "code": "import json\n# ... 计算逻辑 ...\nprint(json.dumps({'columns': ['name', 'total'], 'rows': [...]}))"
  }
}

// 场景 3：提交时重命名或筛选列
{
  "tool_name": "execute_probe_query",
  "tool_args": { "queries": ["SELECT first_name, last_name, balance FROM accounts"], "limit": 200 },
  "columns": ["first_name", "last_name", "balance"]
}
```

### 整体架构变更

```
                    ┌──────────────┐
                    │   model_step  │
                    └──────┬───────┘
                           │ tool_calls
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         submit_tool_result        其他工具
                 │                    │
                 ▼                    ▼
             执行源工具              返回数据
             提取结果
             AnswerTable
                 │
                 ▼
              validate_answer_step       │
                       │                │
                       ▼                │
                    finalize            │
```

### 详细变更清单

#### 1. 新增 Schema（langgraph_tools.py）

```python
class SubmitToolResultArgs(BaseModel):
    tool_name: str = Field(
        description=(
            "The data tool to execute for generating the answer. "
            "Supported: 'execute_probe_query', 'execute_python', 'execute_context_sql'."
        )
    )
    tool_args: dict[str, Any] = Field(
        description=(
            "The arguments to pass to the specified tool. "
            "Use the same argument format as calling the tool directly."
        )
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional: override or reorder the answer columns. "
            "If omitted, columns are extracted from the tool's output automatically."
        ),
    )
```

#### 2. 新增结果提取器（registry.py）

为每种支持的源工具实现一个结果提取函数，将工具输出统一转换为 `(columns, rows)` 格式：

| 源工具 | 提取逻辑 |
|--------|----------|
| `execute_probe_query` | 返回结果是 `{ok, results: [{ok, columns, rows}, ...]}` 的批量结构。默认选取 **最后一个 ok=True 的子查询**结果（模型通常把最终查询放在最后）。 |
| `execute_python` | 返回结果是 `{success, output, stderr}`。约定模型在 Python 代码中 `print(json.dumps({"columns": [...], "rows": [...]}))`。系统从 stdout 中解析 JSON 提取。 |
| `execute_context_sql` | 返回结果是 `{ok, content: {columns, rows}}` 结构，直接提取。 |

```python
# 每种源工具的结果提取器
def _extract_answer_from_probe_query(content: dict) -> tuple[list[str], list[list]]:
    """从 execute_probe_query 的批量结果中，选取最后一个成功子查询。"""
    results = content.get("results", [])
    for result in reversed(results):
        if result.get("ok") and result.get("columns") and result.get("rows") is not None:
            return list(result["columns"]), [list(row) for row in result["rows"]]
    raise ValueError("No successful query result found in execute_probe_query output.")

def _extract_answer_from_python(content: dict) -> tuple[list[str], list[list]]:
    """从 execute_python 的 stdout 中解析 JSON 格式的答案。"""
    output = content.get("output", "")
    # 尝试从 stdout 中查找 JSON 对象
    import json
    # 从后往前找最后一个 JSON 对象
    start = output.rfind("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(
            "execute_python output does not contain a valid JSON object. "
            "The Python code must print a JSON object with 'columns' and 'rows' keys."
        )
    parsed = json.loads(output[start:end + 1])
    columns = parsed.get("columns")
    rows = parsed.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Parsed JSON must contain 'columns' (list) and 'rows' (list).")
    return list(columns), [list(row) for row in rows]

def _extract_answer_from_context_sql(content: dict) -> tuple[list[str], list[list]]:
    """从 execute_context_sql 的结果中直接提取。"""
    columns = content.get("columns")
    rows = content.get("rows")
    if not columns or rows is None:
        raise ValueError("execute_context_sql did not return columns/rows.")
    return list(columns), [list(row) for row in rows]

# 注册提取器映射
_ANSWER_EXTRACTORS = {
    "execute_probe_query": _extract_answer_from_probe_query,
    "execute_python": _extract_answer_from_python,
    "execute_context_sql": _extract_answer_from_context_sql,
}
```

#### 3. 新增 handler（registry.py）

```python
def _submit_tool_result(
    runtime_context: ToolRuntimeContext,
    action_input: dict[str, Any],
) -> ToolExecutionResult:
    tool_name = str(action_input["tool_name"])
    tool_args = action_input.get("tool_args", {})
    requested_columns = action_input.get("columns")

    # 1. 校验源工具是否支持
    if tool_name not in _ANSWER_EXTRACTORS:
        supported = ", ".join(sorted(_ANSWER_EXTRACTORS.keys()))
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Unsupported source tool: {tool_name}. Supported: {supported}"},
        )

    # 2. 执行源工具
    try:
        source_result = runtime_context.registry.execute(
            runtime_context, tool_name, tool_args
        )
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Failed to execute {tool_name}: {exc}"},
        )

    if not source_result.ok:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"{tool_name} execution failed", "details": source_result.content},
        )

    # 3. 提取答案数据
    try:
        extractor = _ANSWER_EXTRACTORS[tool_name]
        columns, rows = extractor(source_result.content)
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Failed to extract answer from {tool_name}: {exc}"},
        )

    # 4. 处理可选的列覆盖
    if requested_columns is not None:
        if len(requested_columns) != len(columns):
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": f"Column count mismatch: specified {len(requested_columns)}, "
                             f"but tool returned {len(columns)} columns ({columns})."
                },
            )
        columns = list(requested_columns)

    # 5. 构造 AnswerTable
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "source_tool": tool_name,
            "column_count": len(columns),
            "row_count": len(rows),
        },
        is_terminal=True,
        answer=answer,
    )
```

#### 4. 注册工具（registry.py - create_default_tool_registry）

```python
specs["submit_tool_result"] = ToolSpec(
    name="submit_tool_result",
    description=(
        "Submit the final answer by executing a data tool and using its output directly. "
        "The final result must be produced by a supported tool call "
        "(e.g., a SQL query or Python script). The system will execute the specified "
        "tool with the given arguments and convert the output to the answer table. "
        "Supported tools: execute_probe_query, execute_python, execute_context_sql."
    ),
    args_schema=SubmitToolResultArgs,
)

handlers["submit_tool_result"] = _submit_tool_result
```

#### 5. 不截断 submit_tool_result 的输出（registry.py - format_result）

在 `format_result` 中，不对 `submit_tool_result` 的提交内容进行截断：

```python
def format_result(self, action: str, result: ToolExecutionResult) -> dict[str, Any]:
    payload = {"ok": result.ok, "content": result.content}
    if result.answer is not None:
        payload["answer"] = result.answer.to_dict()
    if action != "submit_tool_result":
        payload["content"] = truncate_content(...)
    return payload
```

#### 6. 适配 force_answer（langgraph_runtime.py）

`force_answer_step` 只绑定最终提交工具 `submit_tool_result`：

```python
final_submission_tools = [
    tool for tool in langchain_tools
    if tool.name == "submit_tool_result"
]
```

同理，强制提交阶段的工具名和 schema 过滤逻辑都只保留 `submit_tool_result`。

#### 7. ToolRegistry 结构调整

`_submit_tool_result` handler 内部需要调用 `runtime_context.registry.execute()`，但当前 `ToolRegistry` 没有暴露给 `ToolRuntimeContext`。需要做一个小调整：

```python
# 方案 A：在 ToolRuntimeContext 中暴露 registry 引用
@dataclass(slots=True)
class ToolRuntimeContext:
    task: PublicTask
    python_workspace: TaskContextWorkspace
    budget: DataInspectorSampleBudget = ...
    model: object | None = ...
    registry: ToolRegistry | None = None  # ← 新增

# 在 ToolRegistry.bind() 中自动注入
def bind(self, runtime_context: ToolRuntimeContext) -> BoundToolRegistry:
    runtime_context.registry = self  # ← 注入
    ...
```

#### 8. 更新 System Prompt

在系统提示中增加 `submit_tool_result` 的使用说明，让模型知道这个新工具的存在和使用方式。具体追加到 `prompt.py` / `prompt2.py` 的工具说明部分：

```
### submit_tool_result
When your final answer is the direct output of a data query or computation,
call `submit_tool_result`. Specify the tool name and arguments, and the system
will execute it and use the output as your answer.

- For `execute_probe_query`: put your final SQL query in the `queries` list.
  The last successful query's result becomes the answer.
- For `execute_python`: your code must print a JSON object with "columns"
  (list of column names) and "rows" (list of row lists) to stdout.
- Use the optional `columns` parameter to rename or reorder the output columns.
```

### 不变的部分

以下组件无需修改，现有逻辑自动兼容：

- **tool_step 节点**：已有的 `if result.answer is not None: terminal_answer = result.answer` 逻辑直接生效。
- **route_after_tool**：检测到 `answer is not None` 后自动走 `validate_answer` 路径。
- **validate_answer_step**：对 `submit_tool_result` 产生的答案做同样的格式校验。
- **runner.py 落盘逻辑**：`AnswerTable` 格式不变，`prediction.csv` 写出逻辑无需改动。
- **trace 写入**：trace 中记录的 answer 结构不变。

### 风险点与应对

| 风险 | 应对策略 |
|------|----------|
| 模型可能在未充分探索时就调用 submit_tool_result | system prompt 中明确指示"仅在有充分证据后使用" |
| execute_python 的 stdout 中可能没有有效 JSON | 提取器给出清晰的错误信息，引导模型修正代码格式 |
| 批量查询中模型想要的不是最后一个结果 | 可在后续迭代中增加 `query_index` 参数；当前先采用"最后一个成功查询"的简单策略 |
| submit_tool_result 执行可能超时 | 复用 execute_python 的 30s 超时和 execute_probe_query 的内置超时 |
| 强制答案阶段模型不调用 submit_tool_result | force_answer 阶段只绑定 submit_tool_result，并强制 tool_choice 为该工具 |
