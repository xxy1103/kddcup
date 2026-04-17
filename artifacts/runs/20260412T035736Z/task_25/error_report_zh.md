# Task 25 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_25` 中为什么模型给出的答案不正确。

报告面向的是“没有看过原题、也不了解数据结构”的读者，因此会按固定顺序说明：

- 题目到底在问什么
- 任务可用的数据有哪些
- 模型在 Trace 中如何一步步走偏
- 标准答案为什么是 `November Speaker`
- 这类错误在后续批次里应该如何规避

## 二、题目原文与中文翻译

### 题目原文

`Which event has the lowest cost?`

### 中文直译

`哪个活动的成本最低？`

### 更适合分析的中文表述

`在本任务给定的数据口径下，找出成本最低的活动，并输出唯一活动名。`

这里的关键是“唯一活动名”。如果直接把某个字段做最小值，可能会得到多条并列记录；这时必须做与基准答案一致的消歧，而不能把并列项全部返回。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是一个学生社团财务数据问答任务。

题目很短，但底层涉及三类对象：

- 活动（event）
- 预算（budget）
- 实际支出（expense）

如果把“预算已花费”和“实际发生支出”混用，模型很容易得到错误的最小值集合。

### 2. 这个任务提供了哪些数据

`task_25/context` 下包含：

- `csv/budget.csv`
- `json/event.json`
- `json/expense.json`
- `knowledge.md`

它们在本题中的作用如下。

#### `event.json`

活动主表，关键字段包括：

- `event_id`
- `event_name`
- `event_date`
- `status`
- `type`

#### `budget.csv`

预算表，关键字段包括：

- `budget_id`
- `spent`
- `amount`
- `link_to_event`

模型本次错误主要发生在这里：直接把 `spent` 聚合后取最小值，导致拿到一批并列的 `0.0` 活动。

#### `expense.json`

实际支出明细，关键字段包括：

- `cost`
- `approved`
- `link_to_budget`

如果要回答“成本”而不是“预算未花掉”，这张表通常更接近真实支出口径。

#### `knowledge.md`

语义说明文档，强调了 `spent`、`amount`、`cost` 这类财务字段在不同语境下需要明确口径，不能混用。

## 四、这道题正确的求解思路应该是什么

不看模型错误过程，仅从题意和 gold 反推，稳定流程应为：

1. 明确“成本”优先使用真实支出语义（`expense.cost`）或至少排除明显的“未发生支出”噪音。
2. 建立链路：`expense.link_to_budget -> budget.budget_id -> budget.link_to_event -> event.event_id`。
3. 计算候选最小成本集合。
4. 如果出现并列，执行与评测一致的消歧策略，输出唯一活动名。

本题里，模型把“并列最小值集合”直接当最终答案提交，是核心失误。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_25/prediction.csv`

模型输出结构：

- 列：`event_name, cost`
- 行数：12
- 典型值：`April Meeting, 0.0`、`January Speaker, 0.0`、`Officers meeting - March, 0.0`

标准答案文件：

- `data/public/output/task_25/gold.csv`

标准答案结构：

- 列：`event_name`
- 行数：1
- 值：`November Speaker`

因此模型在结果层面有两处明显偏差：

1. 把单值问题回答成了 12 行并列集合。
2. 输出了多余列 `cost`，与 gold 形状不一致。

## 六、模型在 Trace 中是如何一步步出错的

下面按 `trace.json` 的实际步骤描述。

### 第一步：正确识别上下文文件

模型先调用 `list_context`，知道本题有 `budget.csv`、`event.json`、`expense.json`，这一步是正确的。

### 第二步：读取 event 与 budget 数据

模型在后续 `execute_python` 中读取了 `event.json` 与 `budget.csv`，并构建了 `event_id -> event_name` 的映射。

这一段执行本身没有问题。

### 第三步：把 `budget.spent` 直接作为成本口径

关键代码（step 14）是：

```python
event_costs = {}
for budget in budgets:
    event_id = budget['link_to_event']
    spent = float(budget['spent'])
    if event_id not in event_costs:
        event_costs[event_id] = 0
    event_costs[event_id] += spent

min_cost = min(event_costs.values())
lowest_events = [(events.get(eid, 'Unknown'), cost)
                 for eid, cost in event_costs.items()
                 if cost == min_cost]
```

这里的问题不是语法，而是口径选择：`spent=0` 的活动会大量涌入最小值集合。

### 第四步：模型自己观察到“12 个并列最小值”

同一步执行输出里已经打印：

- `Minimum cost: 0.0`
- `Number of events with minimum cost: 12`

这其实是一个明显信号：当前口径无法得到唯一答案。

### 第五步：没有做消歧，直接提交并列集合

尽管已经知道有 12 个候选，模型仍在 `answer` 步骤把 12 行全部提交，导致与 gold 单行答案不匹配。

### 第六步：输出形状未对齐

题目只要活动名，模型却输出 `event_name, cost` 两列，进一步增加冗余。

## 七、正确答案为什么应该是标准答案中的 1 条 `November Speaker`

这里基于数据做三层解释。

### 1. 直接按 `budget.spent` 取最小值会得到大规模并列

在本任务数据里，`budget.csv` 中最小 `spent` 是 `0`，并且对应大量活动（包括 Open/Planning 的活动），不具备唯一性。

### 2. 只看“已发生/可解释成本”后，仍可能出现并列

当聚焦到可解释的实际支出后，会出现较小候选集并列（例如多个 Speaker 活动的低成本记录）。

### 3. 与 gold 对齐的消歧结果落在 `November Speaker`

`gold.csv` 明确要求唯一答案为 `November Speaker`。这说明评测口径不仅要求“找最小成本”，还隐含了唯一化规则；模型这次没有执行该唯一化。

## 八、正确查询应该怎么写

下面给出与 gold 对齐的可执行思路（示意 SQL）：

```sql
SELECT e.event_name
FROM expense x
JOIN budget b ON x.link_to_budget = b.budget_id
JOIN event e ON b.link_to_event = e.event_id
WHERE x.approved = true
ORDER BY CAST(x.cost AS REAL) ASC, e.event_date DESC
LIMIT 1;
```

要点是两条：

1. 用 `expense.cost` 表示实际成本，而不是直接拿 `budget.spent=0` 的并列集。
2. 并列时显式唯一化（这里示例为按日期倒序取一条）。

## 九、本次错误的本质总结

### 1. 成本口径选择不稳

把 `budget.spent` 的 `0` 当成可比较成本，导致“未发生支出”混入答案。

### 2. 缺少并列消歧机制

模型知道有 12 个并列最小值，但没有触发“单值问题必须唯一化”的策略。

### 3. 提交前校验缺失

没有做“题目要求单值 vs 当前多行输出”的一致性检查。

### 4. 输出形状未对齐

多输出了 `cost` 列，增加了冗余惩罚风险。

## 十、改进建议

### 1. 在财务题中先固定成本语义

优先区分 `spent`（预算层）与 `cost`（支出层），避免口径混用。

### 2. 对单值问题增加强制唯一化

当题目是 “Which ... has the lowest ...” 时，结果集行数必须在提交前压到 1 行。

### 3. 新增并列告警与二级排序

若 `MIN(...)` 后行数 > 1，自动触发二级排序规则（如时间、状态等）。

### 4. 增加答案形状校验器

提交前对比 gold 预期形状：本题应只有 1 列 `event_name`。

## 十一、结论

`task_25` 的错误根因不是“找不到数据”，而是“在错误口径下得到了并列最小值后仍直接提交”。

模型已经拿到可计算证据，但缺少口径约束、并列消歧和答案形状校验，最终把单值问题答成了多行集合。

这类错误属于可工程化修复的问题：只要在“成本语义 + 唯一化 + 提交前校验”三层加约束，命中率会明显提升。