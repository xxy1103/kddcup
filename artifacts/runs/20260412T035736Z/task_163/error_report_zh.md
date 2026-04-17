# Task 163 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_163` 中模型为什么没有命中标准答案。

本报告将严格按范式展开：先讲题意和数据，再讲 Trace 错误路径，最后给出正确口径与可执行修复建议。

## 二、题目原文与中文翻译

### 题目原文

`Identify the type of expenses and their total value approved for 'October Meeting' event.`

### 中文直译

`识别 'October Meeting' 事件中已批准支出的类型及其总金额。`

### 更适合分析的中文表述

`先定位 October Meeting，再统计该事件下 approved 的费用总额，并返回事件层面的 type 与总金额。`

本题最大的误区是“type 属于哪一层”：

- 模型答成了预算类别（Food/Advertisement）
- gold 要的是事件类型（Meeting）

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是一个跨源财务聚合任务：

- 事件主信息在 SQLite（`event.db`）
- 预算在 JSON（`budget.json`）
- 支出在 CSV（`expense.csv`）

需要跨三源完成实体对齐和聚合。

### 2. 这个任务提供了哪些数据

`task_163/context` 下包含：

- `db/event.db`
- `json/budget.json`
- `csv/expense.csv`
- `knowledge.md`

#### `event.db`

`event` 表里可查到：

- `event_name='October Meeting'`
- `event_id='recggMW2eyCYceNcy'`
- `type='Meeting'`

这里的 `type` 就是 gold 第一列来源。

#### `budget.json`

预算记录通过 `link_to_event` 指向事件。

对 `October Meeting`，可关联到两个预算项（Food、Advertisement）。

#### `expense.csv`

费用明细通过 `link_to_budget` 指向预算。

过滤 `approved=true` 后再汇总可得到该事件总支出。

#### `knowledge.md`

用于约束“字段语义归属”。本题关键是区分：

- 事件类型（event.type）
- 预算类别（budget.category）

## 四、这道题正确的求解思路应该是什么

正确求解链路：

1. 在 `event.db` 查 `October Meeting`，得到 `event_id` 与 `type=Meeting`。
2. 在 `budget.json` 找 `link_to_event=该 event_id` 的预算集合。
3. 在 `expense.csv` 中筛 `approved=true` 且 `link_to_budget` 落在预算集合内。
4. 汇总 `cost` 得总额。
5. 输出列应是：事件 `type` 与总额。

按该口径可得：`Meeting, 175.39`。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_163/prediction.csv`

模型输出：

- 列：`expense_type, total_value`
- 行：2
- 值：
  - `Food, 121.14`
  - `Advertisement, 54.25`

标准答案文件：

- `data/public/output/task_163/gold.csv`

标准答案：

- 列：`type, SUM(T3.cost)`
- 行：1
- 值：`Meeting, 175.39`

模型其实算对了总额分解（121.14 + 54.25 = 175.39），但答错了维度。

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：正确查到了事件主记录

step 9/10 的 SQL：

```sql
SELECT event_id, event_name, event_date, type
FROM event
WHERE event_name = 'October Meeting'
```

返回记录明确包含 `type='Meeting'`。

### 第二步：后续聚合转向了预算类别

模型在 step 14 的 Python 中：

- 读取 `expense.csv`
- 读取 `budget.json`
- 过滤 `approved=true`
- 按 `b['category']` 分组累计

最终打印：

- Advertisement: 54.25
- Food: 121.14

### 第三步：模型知道总额是 175.39

同一步输出里已有：

`Total approved expenses for October Meeting: 175.39`

说明模型掌握了正确总额。

### 第四步：最终答案拼装错维度

提交时输出 `expense_type,total_value` 两行，而不是 `type,SUM(T3.cost)` 一行。

### 第五步：缺少“题面字段归属”复核

题面中的 `type` 应回到事件层（Meeting），模型却沿用了预算类别层。

### 第六步：结果形状与 gold 不一致

gold 是单行，模型给了双行，导致完全不匹配。

## 七、正确答案为什么应该是标准答案中的 1 行 `Meeting, 175.39`

### 1. `October Meeting` 的事件类型是 `Meeting`

从 `event.db` 查询结果可直接得到：`type='Meeting'`。

### 2. approved 费用总额是 `175.39`

对该事件关联预算的 approved 支出求和，得到总额 `175.39`。

### 3. budget category 只是中间分解，不是最终 type

`Food=121.14` 与 `Advertisement=54.25` 只是构成总额的中间维度。

题目和 gold 都要求的是事件层 type，因此最终应输出单行：`Meeting, 175.39`。

## 八、正确查询应该怎么写

该题跨 SQLite + JSON/CSV，下面给出等价伪 SQL 逻辑：

```sql
WITH target_event AS (
  SELECT event_id, type
  FROM event
  WHERE event_name = 'October Meeting'
),
linked_budget AS (
  SELECT budget_id
  FROM budget_json
  WHERE link_to_event IN (SELECT event_id FROM target_event)
),
approved_expense AS (
  SELECT cost
  FROM expense_csv
  WHERE approved = true
    AND link_to_budget IN (SELECT budget_id FROM linked_budget)
)
SELECT te.type, SUM(ae.cost)
FROM target_event te, approved_expense ae
GROUP BY te.type;
```

输出应为：`Meeting, 175.39`。

## 九、本次错误的本质总结

### 1. 维度错位

把事件层 `type` 误答成预算层 `category`。

### 2. 聚合后答案拼装失败

虽然算出了正确总额，但最终答案字段没有对齐题目。

### 3. 结果唯一化失败

gold 要单行，模型提交了两行分组结果。

### 4. 缺少字段归属检查

提交前没有验证“题面关键词 type 对应哪张表”。

## 十、改进建议

### 1. 增加字段归属白名单

对常见字段（`type`、`category`、`status`）维护“默认来源表”，提交前强校验。

### 2. 在多源题中强制写出最终输出合同

在执行工具前先写明：最终列名、来源实体、聚合粒度。

### 3. 增加“中间分解 vs 最终答案”分离机制

允许模型内部按 category 分解，但提交层必须按题面目标维度收敛。

### 4. 提交前做形状校验

若题目期望单行单维，禁止直接输出分组明细表。

## 十一、结论

`task_163` 的问题不是“算错数”，而是“答错维度”。

模型已经找到了事件、筛出了 approved 记录，也得到了正确总额 `175.39`；但在最后一步把事件 `type` 错替换成预算 `category`，导致从 `Meeting, 175.39` 偏离为 `Food/Advertisement` 两行明细。

这类错误可通过“字段归属约束 + 提交前输出合同校验”稳定修复。