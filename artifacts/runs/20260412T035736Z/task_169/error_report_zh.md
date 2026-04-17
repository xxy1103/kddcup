# Task 169 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_169` 中模型为什么输出了数量级明显异常的结果。

本题属于典型“聚合口径错位”问题：

- 模型做了 `SUM(Consumption)/12`
- gold 要的是 `AVG(Consumption)/12`

## 二、题目原文与中文翻译

### 题目原文

`What was the average monthly consumption of customers in SME for the year 2013?`

### 中文直译

`2013 年 SME 客户的月均消费是多少？`

### 更适合分析的中文表述

`先筛出 2013 年且客户 Segment=SME 的消费记录，再对 Consumption 求平均并除以 12。`

关键是“average monthly”落在平均口径，而不是全年总量口径。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是跨源聚合题：

- 客户分群在 SQLite
- 月度消费在 CSV

需要先定人群，再定时间，再定聚合算子。

### 2. 这个任务提供了哪些数据

`task_169/context` 下包含：

- `db/customers.db`
- `csv/yearmonth.csv`
- `knowledge.md`

#### `customers.db`

用于筛选 `Segment='SME'` 的 `CustomerID`。

#### `yearmonth.csv`

关键字段：

- `CustomerID`
- `Date`（`YYYYMM`）
- `Consumption`

用于筛选 `Date` 前四位为 `2013` 的记录并做聚合。

#### `knowledge.md`

给出“Average Monthly Consumption”语义提示，但实际评测口径以 gold 为准：`AVG(T2.Consumption) / 12`。

## 四、这道题正确的求解思路应该是什么

正确流程：

1. 从 `customers.db` 取出 `Segment='SME'` 的客户集合。
2. 在 `yearmonth.csv` 中筛：
   - `CustomerID` 在 SME 集合内
   - `Date` 以 `2013` 开头
3. 对筛后记录计算 `AVG(Consumption)`。
4. 将该平均值再除以 `12`。
5. 输出单列单行结果。

按该逻辑可得：`459.9562642871061`（浮点尾差允许微小误差）。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_169/prediction.csv`

模型输出：

- 列：`Average Monthly Consumption`
- 行：1
- 值：`82027220.30416964`

标准答案文件：

- `data/public/output/task_169/gold.csv`

标准答案：

- 列：`AVG(T2.Consumption) / 12`
- 行：1
- 值：`459.9562642871061`

数值差异巨大，说明模型把平均题做成了总量题。

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：正确识别 SME 客户集合

模型先查：

```sql
SELECT CustomerID FROM customers WHERE Segment = 'SME'
```

这一步方向正确。

### 第二步：首次 Python 扫描超时

step 14 用列表做成员判断，执行超时（30s），工具返回超时错误。

### 第三步：优化执行性能但保留错误公式

step 18 改成 `set` 后执行成功，并输出：

- `Total consumption for SME in 2013: 984326643.6500357`
- `Average monthly consumption for SME in 2013: 82027220.30416964`

### 第四步：关键错误在聚合算子

模型公式是：

```python
average_monthly = total_consumption_2013 / 12
```

即 `SUM/12`，没有先做 `AVG`。

### 第五步：未使用记录数做平均

模型没有在最终公式中使用记录数（尽管中途曾统计过记录数量）。

### 第六步：提交前缺少数量级 sanity check

`8.2e7` 作为“月均消费”明显异常，但模型未触发回检。

## 七、正确答案为什么应该是标准答案中的 1 个值 `459.9562642871061`

### 1. 2013 年 SME 样本记录可复现

按条件筛选后，样本记录量为 178337 条。

### 2. 三种口径对比能直接定位错误

在同一筛选集上：

- `SUM(Consumption) = 984326643.65`
- `SUM/12 = 82027220.304...`（模型结果）
- `AVG(Consumption)/12 = 459.956264287...`（gold）

### 3. gold 与 `AVG/12` 严格一致

因此本题错误并非取数失败，而是聚合算子错误。

## 八、正确查询应该怎么写

```sql
WITH sme AS (
  SELECT CustomerID
  FROM customers
  WHERE Segment = 'SME'
),
y2013 AS (
  SELECT y.Consumption
  FROM yearmonth y
  JOIN sme s ON y.CustomerID = s.CustomerID
  WHERE SUBSTR(y.Date, 1, 4) = '2013'
)
SELECT AVG(Consumption) / 12 AS avg_monthly_consumption
FROM y2013;
```

输出应为约 `459.9562642871061`。

## 九、本次错误的本质总结

### 1. 聚合口径错位

“average monthly”被误实现为“全年总量除以 12”。

### 2. 性能修复掩盖语义错误

模型修复了超时问题，但没有修复核心公式。

### 3. 缺少公式对照

未将候选公式（`SUM/12` vs `AVG/12`）并行验证。

### 4. 缺少数量级检查

没有利用业务常识识别异常量纲。

## 十、改进建议

### 1. 在题面关键词触发公式模板

遇到 `average` 时，默认首选 `AVG`，并把 `SUM` 作为备选而非主路径。

### 2. 增加聚合公式双轨验证

对同一筛选集并行计算 `SUM/12` 与 `AVG/12`，用 gold 口径或常识规则筛选。

### 3. 强制输出中间统计

提交前打印并校验：`record_count`、`sum`、`avg`，避免黑盒跳步。

### 4. 增加数量级阈值告警

当“月均”结果达到异常数量级时，自动回退到公式复核节点。

## 十一、结论

`task_169` 的核心问题是公式层面的聚合口径错误。

模型在取数和筛选上基本正确，也成功修复了性能超时，但最终把 `AVG(Consumption)/12` 误写成 `SUM(Consumption)/12`，导致结果从正确的 `459.956...` 膨胀到 `82027220.304...`。

这是典型的“执行正确、公式错误”问题，需通过聚合模板约束与数量级校验来修复。