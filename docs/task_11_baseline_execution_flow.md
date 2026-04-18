# Baseline 单任务执行全流程教学文档

本文用一个具体样例把 baseline 的执行链路讲清楚：假设你执行的是 `task_11`。

如果你口头写成了 `taskk_11`，在当前仓库里真正存在的任务 ID 是 `task_11`，任务文件位于 `data/public/input/task_11/`。

---

## 1. 先看你输入的命令

最典型的单任务运行命令是：

```bash
uv run dabench run-task task_11 --config configs/react_baseline.example.yaml
```

这条命令对应的是：

1. `uv run dabench ...`
2. `dabench` 这个 CLI 入口在 `pyproject.toml:34`，映射到 `data_agent_baseline.cli:main`
3. `main()` 在 `src/data_agent_baseline/cli.py:274`
4. `run-task` 子命令实际对应 `run_task_command()`，在 `src/data_agent_baseline/cli.py:144`

所以，从“你输入命令”到“Python 真正开始跑逻辑”，第一个入口链路是：

```text
uv run dabench run-task task_11 --config ...
-> pyproject.toml [project.scripts]
-> data_agent_baseline.cli:main
-> cli.py / run_task_command()
```

---

## 2. CLI 收到命令后先做什么

`run_task_command()` 的核心事情非常少，主要做 3 件事：

1. 读取配置
2. 创建本次运行的输出目录
3. 调 `run_single_task()` 真正执行任务

对应代码在 `src/data_agent_baseline/cli.py:144`。

关键代码可以理解成：

```python
app_config = load_app_config(config)
_, run_output_dir = create_run_output_dir(...)
artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)
```

也就是说，CLI 本身不负责“解题”，它只是一个很薄的调度入口。

---

## 3. 配置是怎么加载进来的

配置加载入口在 `src/data_agent_baseline/config.py:66` 的 `load_app_config()`。

它会把 YAML 解析成 3 组配置：

- `dataset`：数据集根目录
- `agent`：模型名、API 地址、API Key、最大步数、温度
- `run`：输出目录、并发数、超时

对应的数据类在：

- `DatasetConfig`：`src/data_agent_baseline/config.py:24`
- `AgentConfig`：`src/data_agent_baseline/config.py:30`
- `RunConfig`：`src/data_agent_baseline/config.py:40`
- `AppConfig`：`src/data_agent_baseline/config.py:49`

以 `configs/react_baseline.example.yaml` 为例，这份配置表达的是：

- 数据集根目录：`data/public/input`
- 输出目录：`artifacts/runs`
- 模型参数来自 `agent.*`
- `max_steps: 16`，表示一个任务最多给 agent 16 次 ReAct 机会

注意两点：

1. `config.py` 会把相对路径统一解析到项目根目录下
2. `api_key` 只是配置字段，真正发请求时是否可用，要到模型适配层才会检查

---

## 4. `task_11` 是怎么从数据集里被取出来的

任务数据集封装在 `src/data_agent_baseline/benchmark/dataset.py`。

真正取单个任务的入口是：

- `DABenchPublicDataset.get_task()`：`src/data_agent_baseline/benchmark/dataset.py:66`

这个函数会做几件事：

1. 找到任务目录：`data/public/input/task_11/`
2. 读取 `task.json`
3. 校验 `task.json` 里的 `task_id` 是否真的等于目录名 `task_11`
4. 校验 `context/` 目录是否存在
5. 组装成 `PublicTask`

任务对象的数据结构在 `src/data_agent_baseline/benchmark/schema.py`：

- `TaskRecord`：`src/data_agent_baseline/benchmark/schema.py:10`
- `TaskAssets`：`src/data_agent_baseline/benchmark/schema.py:18`
- `PublicTask`：`src/data_agent_baseline/benchmark/schema.py:25`

对 `task_11` 来说，当前仓库里的真实题目是：

```text
For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.
```

也就是：

```text
找出 thrombosis 严重程度为 severe 的患者，输出他们的 ID、性别、诊断疾病。
```

`task_11/context/` 里当前有这些文件：

- `knowledge.md`
- `json/Patient.json`
- `json/Examination.json`

这一步之后，程序手里已经拿到了一个完整的 `PublicTask`，包含：

- `task.question`
- `task.task_dir`
- `task.context_dir`

---

## 5. 输出目录是怎么创建的

输出目录逻辑在 `src/data_agent_baseline/run/runner.py`：

- `create_run_id()`：`src/data_agent_baseline/run/runner.py:43`
- `resolve_run_id()`：`src/data_agent_baseline/run/runner.py:48`
- `create_run_output_dir()`：`src/data_agent_baseline/run/runner.py:61`

默认行为是：

1. 如果你没手动指定 `run.run_id`
2. 就自动生成一个 UTC 时间戳，例如 `20260329T073559Z`
3. 然后创建目录：`artifacts/runs/<run_id>/`

单任务运行时，最终通常会写出：

```text
artifacts/runs/<run_id>/task_11/trace.json
artifacts/runs/<run_id>/task_11/prediction.csv
```

这里要特别注意：

- `run-task` 不会写 `summary.json`
- `summary.json` 只在 `run_benchmark()` 里批量运行时才会写

---

## 6. 真正开始执行任务：`run_single_task()`

单任务入口在 `src/data_agent_baseline/run/runner.py:207` 的 `run_single_task()`。

它的逻辑是：

1. 记录整个任务的开始时间
2. 如果没有外部传入共享的 `model` / `tools`，就走 `_run_single_task_with_timeout()`
3. 拿到结果后追加 `e2e_elapsed_seconds`
4. 把结果写到磁盘

### 6.1 超时分支

外层超时逻辑在 `src/data_agent_baseline/run/runner.py:142` 的 `_run_single_task_with_timeout()`。

默认设计是：

1. 父进程启动一个子进程来跑任务
2. 父进程 `join(timeout_seconds)`
3. 如果超时，就强制结束子进程

这样可以防止任务卡死。

但这里有一个很关键的实际细节：

当前示例配置 `configs/react_baseline.example.yaml` 里写的是：

```yaml
run:
  task_timeout_seconds: 0
```

而 `_run_single_task_with_timeout()` 里规定：

- 当 `timeout_seconds <= 0` 时
- 直接调用 `_run_single_task_core()`
- 不再启用外层进程级超时

所以如果你真的按这份示例配置运行 `task_11`，实际路径会是：

```text
run_single_task()
-> _run_single_task_with_timeout()
-> timeout_seconds == 0
-> 直接 _run_single_task_core()
```

---

## 7. `_run_single_task_core()` 到底做了什么

核心函数是 `src/data_agent_baseline/run/runner.py:104` 的 `_run_single_task_core()`。

它做 4 件事：

1. 创建 `DABenchPublicDataset`
2. 通过 `get_task(task_id)` 取到 `PublicTask`
3. 创建模型适配器 `OpenAIModelAdapter`
4. 创建工具注册表 `ToolRegistry`
5. 创建 `ReActAgent`
6. 调 `agent.run(task)`

也就是：

```text
_run_single_task_core()
-> get_task("task_11")
-> build_model_adapter(config)
-> create_default_tool_registry()
-> ReActAgent(...)
-> agent.run(task)
```

---

## 8. 模型适配层怎么工作的

模型适配层在 `src/data_agent_baseline/agents/model.py`。

关键点：

- `ModelMessage`：`src/data_agent_baseline/agents/model.py:11`
- `ModelStep`：`src/data_agent_baseline/agents/model.py:18`
- `OpenAIModelAdapter`：`src/data_agent_baseline/agents/model.py:32`
- `OpenAIModelAdapter.complete()`：`src/data_agent_baseline/agents/model.py:47`

这层的作用不是“思考”，而是：

1. 接收一组 messages
2. 通过 OpenAI 兼容接口发给模型
3. 取第一条候选文本返回

如果 `api_key` 为空，它会直接报错：

```text
Missing model API key in config.agent.api_key.
```

所以模型适配层只负责“把 prompt 发出去，再把文本拿回来”。

---

## 9. Prompt 是怎么拼的

Prompt 相关逻辑在 `src/data_agent_baseline/agents/prompt.py`：

- `REACT_SYSTEM_PROMPT`：`src/data_agent_baseline/agents/prompt.py:9`
- `RESPONSE_EXAMPLES`：`src/data_agent_baseline/agents/prompt.py:27`
- `build_system_prompt()`：`src/data_agent_baseline/agents/prompt.py:41`
- `build_task_prompt()`：`src/data_agent_baseline/agents/prompt.py:55`
- `build_observation_prompt()`：`src/data_agent_baseline/agents/prompt.py:64`

这里 baseline 的设计非常重要：

它没有直接使用“函数调用 API”或“structured output API”，而是要求模型始终返回一个固定格式的 fenced JSON：

```json
{"thought":"...","action":"...","action_input":{...}}
```

也就是说，这个 baseline 的协议是：

1. 模型先输出“我现在要做什么工具动作”
2. 程序把这段 JSON 解析出来
3. 本地代码去真的执行工具
4. 再把工具结果作为 observation 回喂给模型

这就是一个非常典型的 ReAct 闭环。

---

## 10. ReActAgent 主循环怎么跑

核心在 `src/data_agent_baseline/agents/react.py`：

- `parse_model_step()`：`src/data_agent_baseline/agents/react.py:53`
- `ReActAgent._build_messages()`：`src/data_agent_baseline/agents/react.py:93`
- `ReActAgent.run()`：`src/data_agent_baseline/agents/react.py:110`

### 10.1 每轮发给模型的消息长什么样

`_build_messages()` 会拼出如下消息序列：

1. `system`
   内容是系统提示词 + 工具说明 + 输出示例
2. 第一条 `user`
   内容是任务题目，也就是 `Question: ...`
3. 对于已经执行过的每一步：
   - 先补一条 `assistant`，内容是模型上一轮原始 JSON
   - 再补一条 `user`，内容是工具执行后的 `Observation: {...}`

因此，模型在第 N 步看到的不是“纯问题”，而是：

```text
系统规则
+ 任务题目
+ 历史动作
+ 历史 observation
```

### 10.2 每一轮循环内部发生了什么

`run()` 里每一轮都按这个顺序执行：

1. `self.model.complete(...)`
2. `parse_model_step(raw_response)`
3. `self.tools.execute(task, action, action_input)`
4. 把工具结果包装成 `observation`
5. 记录成 `StepRecord`
6. 如果当前工具是终止型工具，就结束

如果中间任何一步报错，例如：

- 模型返回的不是合法 JSON
- 工具名不存在
- 工具执行失败

异常不会立刻把整个任务打断，而是会被写成一条失败 step：

```text
action = "__error__"
observation = {"ok": false, "error": "..."}
```

然后 agent 还有机会在后续步数里继续自我修正。

这也是这个 baseline 比较值得学习的一点。

---

## 11. 工具系统是怎么接入的

工具注册表在 `src/data_agent_baseline/tools/registry.py`：

- `ToolSpec`：`src/data_agent_baseline/tools/registry.py:23`
- `ToolExecutionResult`：`src/data_agent_baseline/tools/registry.py:31`
- `ToolRegistry`：`src/data_agent_baseline/tools/registry.py:128`
- `create_default_tool_registry()`：`src/data_agent_baseline/tools/registry.py:149`

当前 baseline 默认工具有：

- `list_context`
- `read_csv`
- `read_json`
- `read_doc`
- `inspect_sqlite_schema`
- `execute_context_sql`
- `execute_python`
- `answer`

对 `task_11` 来说，实际用到的是：

- `list_context`
- `read_doc`
- `read_json`
- `execute_python`
- `answer`

### 11.1 工具为什么不能乱读文件

文件访问约束在 `src/data_agent_baseline/tools/filesystem.py:11` 的 `resolve_context_path()`。

它会强制所有路径都落在当前任务的 `context/` 目录下，避免模型读到题目外的数据。

### 11.2 `execute_python` 为什么比较强

`execute_python` 的底层在 `src/data_agent_baseline/tools/python_exec.py:110` 的 `execute_python_code()`。

它会：

1. 在独立进程里执行 Python
2. 把工作目录切到 `task.context_dir`
3. 捕获 stdout / stderr
4. 施加 30 秒超时

所以从模型角度看，它几乎拿到了一个“可以自己写脚本分析 context 数据”的能力。

这也是为什么 `task_11` 最后是通过 Python 直接 join 两个 JSON 文件完成的。

---

## 12. `task_11` 的真实运行过程复盘

下面不是推测，而是来自这次真实成功运行的产物：

```text
artifacts/runs/20260329T073559Z/task_11/trace.json
artifacts/runs/20260329T073559Z/task_11/prediction.csv
```

这次运行总耗时：

```text
e2e_elapsed_seconds = 69.425
```

### Step 1: 先列出 context 里有哪些文件

动作：

```json
{"action":"list_context","action_input":{"max_depth":4}}
```

结果：

- 看到了 `knowledge.md`
- 看到了 `json/Patient.json`
- 看到了 `json/Examination.json`

这一步对应的代码流是：

```text
ReActAgent.run()
-> ToolRegistry.execute()
-> _list_context()
-> list_context_tree()
```

### Step 2: 先读知识文档 `knowledge.md`

动作：

```json
{"action":"read_doc","action_input":{"path":"knowledge.md","max_chars":4000}}
```

模型在这里读到了一个很关键的信息：

```text
Examination.Thrombosis 中，2 表示 severe cases
Patient 里有 ID / SEX / Diagnosis
```

注意这里有一个很有价值的细节：

- 题目说的是“severe degree of thrombosis”
- 模型并不是直接猜 `2`
- 而是先从 `knowledge.md` 里拿到了字段语义

这说明 baseline 的“先观察、再行动”设计确实在发挥作用。

### Step 3: 预览 `Patient.json`

动作：

```json
{"action":"read_json","action_input":{"path":"json/Patient.json","max_chars":4000}}
```

模型确认了：

- `Patient.json` 顶层有 `records`
- 每条记录里确实有 `ID`、`SEX`、`Diagnosis`

### Step 4: 预览 `Examination.json`

动作：

```json
{"action":"read_json","action_input":{"path":"json/Examination.json","max_chars":4000}}
```

模型确认了：

- `Examination.json` 顶层也有 `records`
- 每条记录里确实有 `ID` 和 `Thrombosis`
- 预览里已经看到了一个 `Thrombosis = 2` 的例子：`ID = 163109`

### Step 5: 第一次 `execute_python`

这一步模型没有继续靠 preview 人肉找，而是直接写 Python 做全量分析。

核心逻辑是：

1. 读 `Patient.json`
2. 读 `Examination.json`
3. 找出所有 `Thrombosis == 2` 的检查记录
4. 按 `ID` 去 `Patient.json` 里找对应患者
5. 输出 `ID`、`SEX`、`Diagnosis`

脚本输出的关键信息是：

```text
Found 18 examinations with severe thrombosis (Thrombosis=2)
Results (3 patients):
{'ID': 163109, 'SEX': 'F', 'Diagnosis': 'SLE'}
{'ID': 2803470, 'SEX': 'F', 'Diagnosis': 'SLE'}
{'ID': 4395720, 'SEX': 'F', 'Diagnosis': 'SLE'}
```

这里能看出一个非常重要的真实数据现象：

- `Examination.json` 里有 18 条 `Thrombosis = 2`
- 但只有 3 个 ID 能在 `Patient.json` 里匹配到患者记录

也就是说，baseline 不是简单“按检查表输出”，而是根据题目要求去补齐 `SEX` 和 `Diagnosis`。如果匹配不到患者记录，就没法合法输出完整答案。

### Step 6: 第二次 `execute_python`

这一步其实是“复核”。

模型再次全量统计，确认：

```text
Total patient records: 1238
Total examination records: 806
Examinations with Thrombosis=2: 18
Patients with severe thrombosis and found in Patient.json: 3
```

最后确认能输出的 3 行是：

```text
[163109, 'F', 'SLE']
[4395720, 'F', 'SLE']
[2803470, 'F', 'SLE']
```

这里顺便能看到一个实现细节：

- 第 5 步结果顺序是 `163109 -> 2803470 -> 4395720`
- 第 6 步结果顺序变成了 `163109 -> 4395720 -> 2803470`

原因不是 runner 排序了，而是模型在 Python 里用了 `set` 去重，再转回列表，最终顺序由它自己脚本决定。

这说明：

- baseline 不会自动帮你排序答案
- 最终 `prediction.csv` 的行顺序，基本就是模型在 `answer` 里提交的顺序

### Step 7: `answer`

最后一步调用终止型工具：

```json
{
  "action":"answer",
  "action_input":{
    "columns":["ID","SEX","Diagnosis"],
    "rows":[
      [163109,"F","SLE"],
      [4395720,"F","SLE"],
      [2803470,"F","SLE"]
    ]
  }
}
```

`answer` 工具在 `src/data_agent_baseline/tools/registry.py:96`。

它会校验：

1. `columns` 必须是非空字符串列表
2. `rows` 必须是列表
3. 每一行都必须和列数一致

一旦通过校验，就返回：

- `is_terminal=True`
- `answer=AnswerTable(...)`

然后 `ReActAgent.run()` 检测到终止信号，就结束循环。

---

## 13. 最终结果是怎么写成 `trace.json` 和 `prediction.csv` 的

落盘逻辑在 `src/data_agent_baseline/run/runner.py:180` 的 `_write_task_outputs()`。

它做两件事：

### 13.1 写 `trace.json`

`trace.json` 一定会写。

它来自 `run_result`，里面包含：

- `task_id`
- `answer`
- `steps`
- `failure_reason`
- `succeeded`
- `e2e_elapsed_seconds`

也就是说，`trace.json` 是完整复盘文件。

你以后排查一个任务为什么失败，第一看的一定是它。

### 13.2 写 `prediction.csv`

只有当 `run_result["answer"]` 是一个合法字典时，才会额外写 `prediction.csv`。

因此：

- 成功提交答案 -> 会有 `prediction.csv`
- 没有答案 / 中途失败 -> 只有 `trace.json`

对这次 `task_11` 成功运行，最终 `prediction.csv` 是：

```csv
ID,SEX,Diagnosis
163109,F,SLE
4395720,F,SLE
2803470,F,SLE
```

---

## 14. 用一句话串起整个执行流程

如果把整个单任务过程压缩成一句话，就是：

```text
CLI 读取配置 -> runner 加载 task_11 -> 构造模型和工具 -> ReActAgent 不断让模型输出 JSON 动作 -> 本地代码执行工具 -> observation 回灌给模型 -> 模型最终调用 answer -> runner 把结果写成 trace.json 和 prediction.csv
```

如果展开成更具体的链路，就是：

```text
uv run dabench run-task task_11 --config ...
-> cli.run_task_command()
-> load_app_config()
-> create_run_output_dir()
-> run_single_task()
-> _run_single_task_with_timeout()
-> _run_single_task_core()
-> DABenchPublicDataset.get_task("task_11")
-> build_model_adapter()
-> create_default_tool_registry()
-> ReActAgent.run(task)
   -> build_system_prompt()
   -> build_task_prompt()
   -> model.complete()
   -> parse_model_step()
   -> tools.execute()
   -> build_observation_prompt()
   -> 重复上述循环
   -> answer
-> _write_task_outputs()
-> trace.json / prediction.csv
```

---

## 15. 这个 baseline 最值得你记住的几个设计点

### 15.1 它是“模型出动作，本地代码执行动作”

不是模型直接返回最终答案，而是：

- 模型负责决策下一步
- 本地代码负责执行工具

这就是 agent runtime 的核心分工。

### 15.2 它不是 function calling，而是“文本协议 + 本地解析”

模型并没有真正调用 API 级函数，而是输出：

```json
{"thought":"...","action":"...","action_input":{...}}
```

然后 `parse_model_step()` 自己解析。

### 15.3 `trace.json` 比 `prediction.csv` 更重要

`prediction.csv` 只是最终答案。

`trace.json` 才能告诉你：

- 模型每一步怎么想
- 调了哪个工具
- 拿到了什么 observation
- 为什么失败

### 15.4 `execute_python` 是高能力工具

很多题最后其实不是靠 `read_json` 或 `read_csv` 直接做完，而是：

- 先预览数据结构
- 再用 `execute_python` 做完整分析

`task_11` 就是一个非常标准的例子。

---

## 16. 你之后如果想自己调试，建议按这个顺序看

### 看入口

- `pyproject.toml:34`
- `src/data_agent_baseline/cli.py:144`

### 看配置

- `src/data_agent_baseline/config.py:66`

### 看任务加载

- `src/data_agent_baseline/benchmark/dataset.py:66`

### 看主执行链路

- `src/data_agent_baseline/run/runner.py:207`
- `src/data_agent_baseline/run/runner.py:104`

### 看 agent 核心循环

- `src/data_agent_baseline/agents/react.py:93`
- `src/data_agent_baseline/agents/react.py:110`

### 看 prompt 协议

- `src/data_agent_baseline/agents/prompt.py:9`
- `src/data_agent_baseline/agents/prompt.py:41`

### 看工具实现

- `src/data_agent_baseline/tools/registry.py:149`
- `src/data_agent_baseline/tools/python_exec.py:110`
- `src/data_agent_baseline/tools/filesystem.py:11`

### 看真实样例

- `data/public/input/task_11/task.json`
- `data/public/input/task_11/context/knowledge.md`
- `data/public/input/task_11/context/json/Patient.json`
- `data/public/input/task_11/context/json/Examination.json`
- `artifacts/runs/20260329T073559Z/task_11/trace.json`
- `artifacts/runs/20260329T073559Z/task_11/prediction.csv`

---

## 17. 最后给你一个“读源码时的大脑地图”

你可以把这个 baseline 理解成 5 层：

### 第 1 层：CLI 层

负责接命令、读配置、打印结果。

### 第 2 层：runner 层

负责单任务/批任务调度、超时控制、输出落盘。

### 第 3 层：agent 层

负责 ReAct 循环，也就是“发 prompt -> 收 JSON -> 执行工具 -> 回灌 observation”。

### 第 4 层：tool 层

负责真正访问任务上下文，例如读 CSV、读 JSON、查 SQLite、跑 Python。

### 第 5 层：benchmark data 层

负责把 `task.json + context/` 包装成统一的 `PublicTask`。

只要你始终带着这 5 层去看代码，就不容易迷路。

---

## 18. `task_11` 的最终答案

当前仓库中这次成功运行写出的结果是：

```csv
ID,SEX,Diagnosis
163109,F,SLE
4395720,F,SLE
2803470,F,SLE
```

这个结果来自：

1. 在 `Examination.json` 中找到 `Thrombosis = 2`
2. 再去 `Patient.json` 中按 `ID` 匹配 `SEX` 和 `Diagnosis`
3. 只能保留能匹配到患者主表的 3 条记录

---

如果你愿意，下一步我可以继续帮你做两件事中的任意一个：

1. 再写一份“只看源码、不看 trace 时该如何自己推导这条链路”的简化版笔记
2. 再画一张更直观的时序图，把 `CLI -> runner -> agent -> tools -> artifacts` 画出来
