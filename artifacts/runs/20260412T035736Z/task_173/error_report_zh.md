# Task 173 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_173` 中模型为什么没有提交答案，并解释该题在多数据源条件下的正确解题口径。

本题的核心问题不是“算错”，而是：

- 数据源之间存在时间粒度与字段粒度差异
- 模型在 `transactions_1k.db` 与 `yearmonth.csv` 之间反复切换
- 最终未在步数限制内完成提交

## 二、题目原文与中文翻译

### 题目原文

`Please list the countries of the gas stations with transactions taken place in June, 2013.`

### 中文直译

`请列出在 2013 年 6 月发生交易的加油站所在国家。`

### 更适合分析的中文表述

`先确认“2013-06 存在交易”，再输出对应加油站国家的去重列表。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是一个多源对齐题：

- 交易时间筛选（June 2013）
- 站点国家映射（Country）
- 最终输出国家去重集合

### 2. 这个任务提供了哪些数据

`task_173/context` 下主要有：

- `db/transactions_1k.db`
- `csv/yearmonth.csv`
- `json/gasstations.json`
- `knowledge.md`

#### `transactions_1k.db`

包含明细交易字段（如 `Date`, `CustomerID`, `GasStationID` 等），但该样本库中的日期仅覆盖 `2012-08`。

#### `yearmonth.csv`

包含按 `CustomerID` + `Date(YYYYMM)` 聚合后的消费记录，确实包含 `201306`。

#### `gasstations.json`

包含加油站维度信息，关键字段有：

- `GasStationID`
- `Country`

该文件中的国家域为 `CZE` 和 `SVK`。

#### `knowledge.md`

给出字段语义和日期格式约定（`YYYYMM`），提示了 `yearmonth` 的时间表示方式。

## 四、这道题正确的求解思路应该是什么

该题在实际上下文里应按以下顺序处理：

1. 用 `yearmonth.csv` 确认 `201306` 存在交易记录（本题中存在）。
2. 用 `gasstations.json` 提取加油站国家集合。
3. 输出国家去重列表。

按官方评测口径，本题应输出：

- `CZE`
- `SVK`

## 五、模型最终给出了什么答案

本题目录下没有生成 `prediction.csv`：

- `artifacts/runs/20260412T035736Z/task_173/prediction.csv` 不存在

运行在 `max_steps` 后终止，失败原因是：

- `Agent did not submit an answer within max_steps.`

标准答案文件为：

- `data/public/output/task_173/gold.csv`

内容是两行国家：

- `CZE`
- `SVK`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：进入多源探查

模型先后读取 DB/CSV/JSON，路径选择是合理的。

### 第二步：在 `transactions_1k.db` 里发现时间不匹配

trace 显示：

- `transactions_1k` 的日期仅有 `2012-08-23` 到 `2012-08-26`
- 进一步按年月汇总也只有 `2012-08`

这与题目要求的 `2013-06` 不一致。

### 第三步：切到 `yearmonth.csv` 后确认 `201306` 存在

trace 又显示：

- `yearmonth.csv` 的 `Date` 为 `int64`
- 含 `201306`
- `201306` 对应记录数为 `25378`

### 第四步：读取 `gasstations.json` 但未完成收敛

模型读取了站点 JSON（含 `Country`），但未把“六月交易存在性”与“国家去重输出”收敛到最终答案。

### 第五步：在源之间来回切换直至超步

后续步骤重复回看 DB 与 CSV，未进入提交阶段，最终超出步数限制。

## 七、正确答案为什么应该是标准答案中的 `CZE, SVK`

从结果口径看，gold 明确要求国家集合：

- `CZE`
- `SVK`

结合上下文可复核：

1. `yearmonth.csv` 明确有 `201306` 交易。
2. `gasstations.json` 的国家集合就是 `CZE/SVK`。
3. 题目要求输出“加油站国家列表”，因此去重国家输出与 gold 一致。

## 八、正确查询应该怎么写

可用 Python/伪 SQL 表达为：

```python
import pandas as pd, json

ym = pd.read_csv('csv/yearmonth.csv')
assert (ym['Date'] == 201306).any()  # June 2013 transactions exist

with open('json/gasstations.json', 'r', encoding='utf-8') as f:
    stations = json.load(f)['records']

countries = sorted({r['Country'] for r in stations})
# -> ['CZE', 'SVK']
```

输出列为 `Country`，行值为 `CZE`,`SVK`。

## 九、本次错误的本质总结

### 1. 数据源冲突下的路径锁死

模型识别到了 DB 与 CSV 的时间不一致，但没有形成明确的降级策略。

### 2. 过度纠缠“完整可连接路径”

本题上下文并不要求构造复杂联结，模型却反复尝试在源间闭环，导致收敛失败。

### 3. 缺少“可提交最小答案”策略

在证据已足够覆盖 gold 的情况下，没有及时提交国家去重结果。

### 4. 工具调用成功但决策终止失败

绝大多数工具执行成功，失败点在最后的决策与提交，而非单点代码报错。

## 十、改进建议

### 1. 先做“题目最小可答”判定

当题目只要国家列表时，优先寻找可直接支撑输出的最小字段集合。

### 2. 多源冲突时引入“主源+佐证源”策略

可将 `yearmonth.csv` 作为时间存在性主源，`gasstations.json` 作为国家维度主源。

### 3. 设置提交前强制门槛

若已经得到：

- 目标时间存在
- 目标维度可去重

则应强制进入提交流程，避免无效循环。

### 4. 对超步风险加早停告警

当出现重复读取同一来源且无新增字段时，应触发早停并提交当前最优答案。

## 十一、结论

`task_173` 的失败是典型的“可解但未提交”问题。

模型实际上拿到了关键证据：

- `201306` 在交易聚合表中存在
- 加油站国家域是 `CZE/SVK`

但由于在多源对齐中反复切换，最终超步未提交。按标准答案口径，本题正确输出应为 `CZE` 与 `SVK`。