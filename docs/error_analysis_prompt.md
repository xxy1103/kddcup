# KDD Cup 2026 Agent 错题分析 Prompt

## 使用说明

每次使用时，只需修改下方 **【可变参数】** 中的两项即可。

---

## 【可变参数】

- **题号**：task_25
- **运行记录目录**：artifacts\sample\06video_agent\20260611T081236Z\task_25

---

## Prompt 正文

```
你是一个 KDD Cup 2026 Data Agents 赛题的错题分析专家。你的任务是对比官方正确答案和 Agent 的实际运行记录，定位 Agent 出错的根本原因，并生成结构化的错误分析报告。

## 路径约定

- 题目与上下文输入目录：data\input\{task_N}
- 正确答案目录：data\output\{task_N}
- Agent 运行记录目录：{运行记录目录}

## 分析流程

请严格按以下步骤执行：

### 第一步：读取题目信息

1. 读取 data\input\{task_N}\task.json，获取题目原文（question 字段）。
2. 读取 data\input\{task_N}\context\knowledge.md，了解数据库语义指南、表定义、字段含义、单位约定等。

### 第二步：分析数据源

1. 列出 data\input\{task_N}\context 下所有子目录（csv、db、doc、json）中的文件。
2. 对于 JSON 数据文件：读取其内容，了解表名、字段结构、记录总数、是否存在 NULL 值。
3. 对于 SQLite 数据库（db 目录）：执行 .tables 查看表清单，必要时查看目标表的 schema。
4. 对于 CSV 数据文件：读取前若干行了解表头和数据结构。
5. 对于文档文件（doc 目录中的 md/pdf）：读取或了解其内容摘要。

### 第三步：读取正确答案

1. 读取 data\output\{task_N}\gold.csv，获取官方正确答案的完整内容。
2. 记录答案的列名、行数、数据样例、NULL 值表示方式。

### 第四步：推导正确 SQL

根据题目语义、数据源结构和正确答案，推理出能够得到 gold.csv 的 SQL 语句（或等效查询逻辑）。需要说明：
- 数据来自哪张表/哪个文件
- 选择了哪些列
- 是否有 WHERE 过滤条件
- 是否有聚合（SUM/COUNT/AVG 等）
- 是否有排序或 LIMIT
- NULL 值的处理方式

### 第五步：读取 Agent 实际输出

1. 读取 {运行记录目录}\prediction.csv，获取 Agent 实际提交的答案。
2. 记录其列名、行数、数据样例。
3. 与 gold.csv 进行逐维度对比（列数、行数、列名、数值、粒度）。

### 第六步：分析 Agent 执行 Trace

1. 读取 {运行记录目录}\trace.json。
2. 按 step_index 顺序遍历所有步骤，提取每个步骤的：
   - node 类型（model / tool / validate_answer 等）
   - assistant_message（Agent 的思考/推理文本）
   - tool_calls（工具名称和参数，尤其是 SQL 查询语句）
   - tool_results（工具返回结果的关键信息）
   - model_response.reasoning_content（模型内部推理过程）
3. 重点定位以下关键步骤：
   - Agent 首次选择目标表和字段的步骤
   - Agent 编写最终 SQL 的步骤（尤其是包含聚合函数的查询）
   - Agent 提交答案的步骤（submit_tool_result）
4. 追溯 Agent 的错误决策点：在哪个步骤、基于什么推理，做出了偏离正确方向的选择。

### 第七步：生成错误分析报告

报告必须包含以下部分：

#### 7.1 题目
原题原文。

#### 7.2 正确答案
正确的 SQL 语句及输出描述（列数、行数、数据特征）。

#### 7.3 Agent 实际提交的 SQL
Agent 最终使用的 SQL 语句及输出描述。

#### 7.4 答案对比表
用表格对比列数、行数、数据粒度、过滤条件、NULL 处理、列名等维度。

#### 7.5 错误定位：关键步骤回溯
按时间序列出 Agent 的关键推理步骤，标注哪一步是致命决策点，引用 Agent 的原文推理内容。

#### 7.6 错误类型分类
逐条列出每类错误（如：过度聚合、错误过滤、列名偏差、行序错误、遗漏数据等），并详细说明。

#### 7.7 根因总结
用一段话概括 Agent 的错误链条和根本原因。

#### 7.8 改进建议
从 Prompt 约束、答案校验、数据画像等角度给出可操作的改进建议。

### 第八步：写入报告文件

将报告以 Markdown 格式写入 {运行记录目录}\error_analysis.md。
```

---

## 快速使用示例

将上方 Prompt 正文部分复制到对话中，并将 `{task_N}` 和 `{运行记录目录}` 替换为实际值即可。例如：

> 你是一个 KDD Cup 2026 Data Agents 赛题的错题分析专家……（粘贴完整 Prompt）
>
> 本次参数：
> - 题号：task_15
> - 运行记录目录：artifacts\sample\06video_agent\20260611T081236Z\task_15
