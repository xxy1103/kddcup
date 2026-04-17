# Task 180 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_180` 中模型为什么返回了大规模冗余结果并与 gold 不匹配。

本题看似简单筛选，实则有两个关键坑：

- `more than 29.00 per unit` 的含义是单价（`Price/Amount`）而不是总价 `Price`
- 最终输出只需要 `Consumption`，不应携带 `CustomerID`

## 二、题目原文与中文翻译

### 题目原文

`For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.`

### 中文直译

`对于所有为 5 号产品支付单价超过 29.00 的人，给出他们在 2012 年 8 月的消费情况。`

### 更适合分析的中文表述

`先在交易表筛出 ProductID=5 且 (Price/Amount)>29 的客户，再到 201208 月度表取这些客户的 Consumption，仅输出 Consumption。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是跨表筛选题：

- 交易条件在 `transactions_1k.db`
- 月消费状态在 `yearmonth.csv`

需要先构造正确人群，再取目标月份消费值。

### 2. 这个任务提供了哪些数据

`task_180/context` 下包含：

- `db/transactions_1k.db`
- `csv/yearmonth.csv`
- `knowledge.md`

#### `transactions_1k.db`

`transactions_1k` 表关键字段：

- `CustomerID`
- `ProductID`
- `Amount`
- `Price`
- `Date`

题面 `per unit` 对应的是 `Price / Amount`。

#### `yearmonth.csv`

关键字段：

- `CustomerID`
- `Date`（`YYYYMM`）
- `Consumption`

用于取 2012-08（`201208`）的消费值。

#### `knowledge.md`

给出日期与消费字段语义，提醒按口径过滤后再聚合/输出。

## 四、这道题正确的求解思路应该是什么

正确流程：

1. 在 `transactions_1k` 中筛：
   - `ProductID = 5`
   - `Price / Amount > 29.00`
   - （`Amount > 0` 防除零）
2. 取去重 `CustomerID`。
3. 在 `yearmonth.csv` 取 `Date='201208'` 且 `CustomerID` 属于上述集合的 `Consumption`。
4. 仅输出 `Consumption` 列。

按该口径应得到 9 条消费值（与 gold 一致）。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_180/prediction.csv`

模型输出：

- 列：`CustomerID, Consumption`
- 行：153
- 示例：`5113,1425.56`、`5328,106067.39`...

标准答案文件：

- `data/public/output/task_180/gold.csv`

标准答案：

- 列：`Consumption`
- 行：9
- 值集合：`1903.2, 88265.39, 1129.2, ...`

模型结果行数与字段都明显过宽。

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：把“per unit > 29”误写成 `Price > 29`

step 10 SQL：

```sql
SELECT DISTINCT CustomerID, Price
FROM transactions_1k
WHERE ProductID = 5 AND Price > 29.00
```

这会把大量总价高但单价不高的记录纳入。

### 第二步：候选人群被放大到 153 人

后续 Python 步骤将该 153 人集合带入 `yearmonth.csv`。

### 第三步：取 201208 消费后得到 153 条

step 16/18 执行输出明确显示：

- `Found 153 records`
- `Total records: 153`

### 第四步：输出粒度错误

模型提交了 `CustomerID, Consumption`，而题目只要求消费状态（Consumption）。

### 第五步：缺少“人群规模 sanity check”

对该类条件题，153 条结果本身就是异常信号，但模型未回检筛选条件。

### 第六步：未做“per unit”词义回查

题面有明确 `per unit`，模型没有把它映射为 `Price/Amount`。

## 七、正确答案为什么应该是标准答案中的 9 条 `Consumption`

### 1. 正确筛选条件会得到 9 位客户

在 `transactions_1k` 上使用 `ProductID=5 AND Price/Amount>29`，可得 9 个唯一客户。

### 2. 对应 201208 的消费值正好 9 条

将这 9 个客户映射到 `yearmonth.csv` 的 `Date='201208'`，得到 9 个 `Consumption` 值。

### 3. 值集合与 gold 完全一致

计算结果集合为：

`58.19, 1129.2, 1142.95, 1903.2, 8878.07, 45937.22, 69331.72, 88265.39, 126157.7`

与 gold 仅排序可能不同，值集合一致。

## 八、正确查询应该怎么写

```sql
WITH target_customer AS (
  SELECT DISTINCT CustomerID
  FROM transactions_1k
  WHERE ProductID = 5
    AND Amount > 0
    AND (Price * 1.0 / Amount) > 29.00
)
SELECT y.Consumption
FROM yearmonth y
JOIN target_customer t ON y.CustomerID = t.CustomerID
WHERE y.Date = '201208';
```

输出只需 `Consumption` 一列。

## 九、本次错误的本质总结

### 1. 条件语义误读

`per unit` 被错读为总价 `Price`。

### 2. 人群过滤过宽

导致客户从应有 9 人扩张到 153 人。

### 3. 输出形状错误

提交了不需要的 `CustomerID`。

### 4. 缺少规模回检

没有利用结果规模（153 vs 9）触发条件复核。

## 十、改进建议

### 1. 建立 `per unit` 规则映射

遇到 `per unit` 自动转写为 `(Price / Amount)` 类型表达式。

### 2. 条件拆解后逐条验证

先验证人群规模，再进入下游取值，避免错误条件一路放大。

### 3. 增加输出最小化约束

题目未要求主键时，禁止输出 ID 列。

### 4. 增加“结果规模异常”回退机制

当条件题返回结果远超预期时，自动回到条件解析节点重试。

## 十一、结论

`task_180` 的核心错误是把“单价阈值”误解成“总价阈值”，导致筛选人群大幅放大，并最终提交了 153 行带 `CustomerID` 的结果。

正确口径应是 `Price/Amount > 29`，这样只会得到 9 条 `Consumption`，与 gold 完全对齐。