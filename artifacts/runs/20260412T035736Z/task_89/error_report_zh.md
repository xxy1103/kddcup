# Task 89 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_89` 中为什么模型提交的答案与标准答案不一致。

报告按照与 `task_86` 相同的讲解顺序展开，重点回答四个问题：

- 题目实际在问哪个字段
- 模型在 Trace 里具体做了什么
- 为什么模型给的是 `+14.925`，而 gold 是 `+16.445`
- 这类 F1 题在后续应如何避免字段误读

## 二、题目原文与中文翻译

### 题目原文

`What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?`

### 中文直译

`2008 年中国大奖赛中，排名第二的车手完赛时间是多少？`

### 更适合分析的中文表述

`先定位 2008 Chinese Grand Prix 对应 race，再在结果表中按“ranked second”匹配正确排名字段，最后只返回 time。`

这里最关键的是：`ranked second` 并不等价于 `position = 2`。在该任务数据里，这两个字段可能分离。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是 F1 赛事结果检索任务。

任务流程通常是：

1. 从赛程表定位目标比赛（年份 + 名称）。
2. 从成绩表取目标车手记录（排名条件）。
3. 输出目标指标（`time`）。

### 2. 这个任务提供了哪些数据

`task_89/context` 下包含：

- `csv/results.csv`
- `json/races.json`
- `knowledge.md`

#### `races.json`

赛程维度，关键字段：

- `raceId`
- `year`
- `name`
- `round`

用于把 `2008 Chinese Grand Prix` 精确定位到 `raceId=34`。

#### `results.csv`

比赛结果维度，关键字段：

- `raceId`
- `driverId`
- `position`
- `positionOrder`
- `rank`
- `time`

本题的核心陷阱就在这里：`position` 与 `rank` 语义不一致。

#### `knowledge.md`

语义说明文件，用于减少字段歧义。对于本题，关键是提醒“排名语义要对齐到正确字段”。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 在 `races.json` 中定位 `year=2008` 且 `name='Chinese Grand Prix'`，得到 `raceId=34`。
2. 在 `results.csv` 中筛 `raceId=34`。
3. 按题目里的“ranked second”筛选 `rank=2`（不是 `position=2`）。
4. 只输出 `time` 一列。

在这份数据中：

- `position=2` 对应时间是 `+14.925`
- `rank=2` 对应时间是 `+16.445`

gold 选择了后者，因此口径应是 `rank=2`。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_89/prediction.csv`

模型输出：

- 列：`driverId, position, time`
- 行：1
- 值：`13, 2, +14.925`

标准答案文件：

- `data/public/output/task_89/gold.csv`

标准答案：

- 列：`time`
- 行：1
- 值：`+16.445`

差异非常明确：

1. 值错：`+14.925` vs `+16.445`
2. 形状错：模型多输出了 `driverId` 和 `position`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：成功定位目标比赛

模型先读取 `races.json`，锁定到 `2008 Chinese Grand Prix`，并拿到 `raceId=34`。这一步是正确的。

### 第二步：尝试 SQL 查询但工具路径用错

模型执行了：

```sql
SELECT driverId, position, time, milliseconds
FROM results
WHERE raceId = 34 AND position = 2
```

调用 `execute_context_sql` 时把路径指向了 `csv/results.csv`，工具报错：`file is not a database`。

### 第三步：转为 Python 读取 CSV

SQL 失败后，模型改用 Python 读取 `results.csv`，这是合理恢复动作。

### 第四步：关键误判，按 `position` 排序并选第 2 名

step 16 的执行输出是：

- Position 1: `time=1:31:57.403`
- Position 2: `time=+14.925`
- Position 3: `time=+16.445`

模型直接按 `position=2` 拿了 `+14.925`。

### 第五步：忽略了 `rank` 字段差异

同一批结果里，`+16.445` 那行对应 `rank=2`。这说明题目语义应落在 `rank`，而模型没有做字段消歧。

### 第六步：提交前未做字段一致性校验

模型提交了 `driverId, position, time` 三列，没有把结果收敛到题目要求的单列 `time`。

## 七、正确答案为什么应该是标准答案中的 1 个时间值 `+16.445`

### 1. 目标赛事唯一

`races.json` 中 2008 Chinese Grand Prix 唯一对应 `raceId=34`，没有赛事层面的歧义。

### 2. 在该 race 内，`position` 与 `rank` 存在分离

从 `results.csv` 可见：

- `position=2` 行：`time=+14.925`，`rank=4`
- `position=3` 行：`time=+16.445`，`rank=2`

题目用词是 `ranked second`，与 `rank=2` 对齐，因此应取 `+16.445`。

### 3. gold 与 `rank=2` 完整一致

`gold.csv` 的唯一值是 `+16.445`，恰好对应 `rank=2` 记录而非 `position=2` 记录。

## 八、正确查询应该怎么写

```sql
SELECT r.time
FROM results r
JOIN races rr ON r.raceId = rr.raceId
WHERE rr.year = 2008
  AND rr.name = 'Chinese Grand Prix'
  AND CAST(r.rank AS INT) = 2;
```

返回应为单列单行：`+16.445`。

## 九、本次错误的本质总结

### 1. 语义字段映射错误

把 `ranked second` 映射成了 `position=2`，忽略了 `rank` 字段。

### 2. 工具失败后的恢复只修“执行方式”，没修“语义口径”

SQL 失败后改用 Python 是对的，但逻辑仍沿用 `position`，根因未修复。

### 3. 缺少字段一致性检查

没有在提交前检查“题面关键词 ranked 对应哪个字段”。

### 4. 输出形状未对齐

题目只要时间值，模型却附带了额外列。

## 十、改进建议

### 1. 建立关键词到字段的强约束映射

在 F1 任务中，将 `ranked` 默认映射到 `rank`，将 `finished second` 才映射到 `position`（必要时双向验证）。

### 2. 增加同义字段冲突检测

当同一表含 `position`、`positionOrder`、`rank` 时，提交前必须执行一次冲突检查。

### 3. 工具失败后强制重审语义

若从 SQL 切到 Python，先复述过滤条件并确认字段，不允许直接复制旧条件。

### 4. 增加答案形状约束

若题目是 “What is the time ...”，默认输出单列 `time`，除非题面显式要求额外列。

## 十一、结论

`task_89` 的核心错误不是“找错比赛”，而是“找对比赛后用错排名字段”。

模型在 `raceId=34` 上完成了数据检索，但把 `ranked second` 误做成 `position=2`，最终提交 `+14.925`，与 gold 的 `+16.445` 偏离。

该问题属于典型字段语义错误，修复重点应放在“rank/position 消歧 + 提交前字段一致性校验”。