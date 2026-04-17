# Task 80 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_80` 中模型为何未命中标准答案。

本报告将按固定范式说明：

- 题目到底在问什么
- 数据字段里有哪些容易混淆点
- 模型在 Trace 中如何一步步走偏
- 为什么 gold 是 `3` 和 `5`
- 后续如何避免同类错误

## 二、题目原文与中文翻译

### 题目原文

`What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?`

### 中文直译

`在第 903 场排位赛中，Q3 成绩为 0:01:54 的车手，他的号码是多少？`

### 更适合分析的中文表述

`先在 raceId=903 的 Q3 结果里定位到 1:54.xxx 的车手，再返回该车手在 drivers 维表中的号码（number）。`

这道题的核心是“号码字段来源”与“时间匹配口径”两个消歧点。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是 F1 排位赛检索题，涉及：

- 按 raceId 与 q3 时间筛选候选车手
- 再回连到 drivers 取标准号码字段

### 2. 这个任务提供了哪些数据

`task_80/context` 下包含：

- `csv/qualifying.csv`
- `json/drivers.json`
- `knowledge.md`

#### `qualifying.csv`

关键字段：

- `raceId`
- `driverId`
- `q3`
- `number`

注意：这里的 `number` 并不一定是题目最终要求的司机号码口径。

#### `drivers.json`

关键字段：

- `driverId`
- `number`

这个 `number` 才是车手主档案号码字段。

#### `knowledge.md`

提示了 ID 与实体字段应优先从主维表取值，避免在事实表里拿到语义漂移字段。

## 四、这道题正确的求解思路应该是什么

正确流程：

1. 在 `qualifying.csv` 中筛 `raceId=903`。
2. 将题面 `0:01:54` 归一到数据格式 `1:54.xxx`，筛出满足 `q3` 前缀 `1:54` 的候选车手。
3. 取候选 `driverId` 后，去 `drivers.json` 回连 `driverId -> number`。
4. 仅输出 `number` 列。

按该逻辑可得到两个号码：`3` 与 `5`。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_80/prediction.csv`

模型输出：

- 列：`driverId, number`
- 行：2
- 值：
  - `817, 3`
  - `20, 1`

标准答案文件：

- `data/public/output/task_80/gold.csv`

标准答案：

- 列：`number`
- 行：2
- 值：
  - `3`
  - `5`

因此模型存在两类偏差：

1. 第二个号码错误：`1` 应为 `5`。
2. 输出多余列：`driverId`。

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：先尝试 SQL，但把 CSV 当数据库

模型执行：

```sql
SELECT * FROM qualifying
WHERE raceId = 903 AND q3 = '0:01:54' OR q3 = '1:54'
```

路径是 `csv/qualifying.csv`，工具报错：`file is not a database`。

### 第二步：切换到 Python 读取 CSV（方向正确）

模型后续用 Python 读取 `qualifying.csv`，并多次尝试匹配 `1:54` 时间段。

### 第三步：时间口径处理为 `1:54.xxx` 候选集合

这一步总体方向是可接受的，最终拿到了 driverId=817 与 driverId=20 两个候选。

### 第四步：关键错误，号码取自 `qualifying.number`

在 `qualifying.csv` 里：

- driverId=817 的 `number=3`
- driverId=20 的 `number=1`

模型直接使用了该列。

### 第五步：没有回连 `drivers.json` 做主档案号码映射

`drivers.json` 中同两位车手的号码是：

- driverId=817 -> number=3
- driverId=20 -> number=5

gold 的 `5` 就来自这里。

### 第六步：提交前未做输出形状收敛

题目要“号码”，模型却提交 `driverId, number` 两列，增加了冗余。

## 七、正确答案为什么应该是标准答案中的 2 个号码 `3` 和 `5`

### 1. raceId=903 的 Q3 存在 `1:54.xxx` 候选

在该 race 的 q3 列中，前缀为 `1:54` 的记录对应两位车手：817 与 20。

### 2. 号码应来自 drivers 主档案

`drivers.json` 明确给出：

- 817 -> `3`
- 20 -> `5`

### 3. 与 gold 完全一致

gold 仅有 `number` 一列，值为 `3`、`5`。

## 八、正确查询应该怎么写

示意伪 SQL / 逻辑：

```sql
-- Step 1: 在 qualifying 中找 raceId=903 且 q3 命中 1:54.xxx 的 driverId
SELECT DISTINCT q.driverId
FROM qualifying q
WHERE q.raceId = 903
  AND q.q3 LIKE '1:54.%';

-- Step 2: 回连 drivers 取标准号码
SELECT d.number
FROM drivers d
WHERE d.driverId IN ( ... 上一步 driverId 集合 ... )
ORDER BY d.number;
```

最终输出应只有一列 `number`。

## 九、本次错误的本质总结

### 1. 字段来源选错

把事实表 `qualifying.number` 当成最终号码，未使用主维表 `drivers.number`。

### 2. 时间歧义处理不完整

虽然识别到 `1:54` 前缀，但没有对“候选后字段来源”做二次核验。

### 3. 工具失败后只修执行方式

从 SQL 改到 Python 后，语义口径（字段来源）没有同步修正。

### 4. 输出形状未对齐

多输出 `driverId`，不符合题目最小输出要求。

## 十、改进建议

### 1. 增加“字段来源优先级”规则

当题目问司机号码时，优先从 `drivers` 主档案取值。

### 2. 时间模糊匹配后必须回连维表

任何 `LIKE`/前缀匹配得到的候选都要做维表二次确认，避免字段漂移。

### 3. 对 CSV SQL 失败增加路径纠错

若 `execute_context_sql` 目标是 CSV，自动转建议为 Python/duckdb 路径，减少无效步骤。

### 4. 增加单字段输出校验

题目目标是“number”时，提交前禁止附带多余 ID 列。

## 十一、结论

`task_80` 的关键问题不是候选车手找错，而是“候选后取值字段选错”。

模型正确拿到了相关 driverId，但用的是 `qualifying.number`，导致第二个号码从正确的 `5` 变成 `1`；同时还输出了多余列 `driverId`。本质属于字段来源错误叠加输出形状错误。