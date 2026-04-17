# Task 200 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_200` 中模型为何从正确候选分子推导出了错误计数。

本题的典型问题是：

- 前半段筛选基本正确
- 最后一步“计数对象”选错，导致 `4` 而不是 `1`

## 二、题目原文与中文翻译

### 题目原文

`Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.`

### 中文直译

`计算含有磷或溴元素、且具有三键分子的原子总数。`

### 更适合分析的中文表述

`先定位“有三键且包含 P/Br 元素”的分子，再统计该条件下目标元素原子数量。`

本题的关键歧义是“统计对象”到底是：

- 条件分子中的全部原子
- 还是条件分子中的目标元素原子（P/Br）

gold 选择后者。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是化学结构查询题，涉及：

- 键类型筛选（triple bond）
- 元素存在性筛选（P/Br）
- 原子计数口径

### 2. 这个任务提供了哪些数据

`task_200/context` 下包含：

- `db/bond.db`
- `csv/atom.csv`
- `json/molecule.json`
- `knowledge.md`

#### `bond.db`

`bond` 表关键字段：

- `molecule_id`
- `bond_type`（`#` 表示 triple bond）

用于筛出具有三键的分子。

#### `atom.csv`

关键字段：

- `atom_id`
- `molecule_id`
- `element`

用于判断分子是否含 P/Br 以及最终计数。

#### `molecule.json`

提供分子附加属性（本题可作为辅助，不是核心计算来源）。

#### `knowledge.md`

明确了 `bond_type` 与原子/分子关系，支持“先筛分子再计数”的路径。

## 四、这道题正确的求解思路应该是什么

正确流程：

1. 在 `bond.db` 中取 `bond_type='#'` 的 `molecule_id` 集合。
2. 在 `atom.csv` 中筛这些分子，判断是否包含 `element in ('p','br')`。
3. 对满足条件的分子，统计目标元素原子数（P 或 Br 的 atom 数量）。
4. 输出单值。

按此口径，本题结果应为 `1`。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_200/prediction.csv`

模型输出：

- 列：`total_atoms`
- 行：1
- 值：`4`

标准答案文件：

- `data/public/output/task_200/gold.csv`

标准答案：

- 列：`COUNT(T1.atom_id)`
- 行：1
- 值：`1`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：先尝试错误 SQL 路径

模型在 `bond.db` 上执行：

```sql
SELECT DISTINCT element
FROM atom
WHERE element LIKE '%p%' OR element LIKE '%br%'
```

报错：`no such table: atom`（因为 `atom` 在 CSV，不在 `bond.db`）。

### 第二步：切换为跨源处理（方向正确）

模型改为：

- 从 `bond.db` 取 triple bond 分子
- 从 `atom.csv` 读元素

### 第三步：正确识别目标分子

trace 输出显示：

- triple bond 分子：`TR041, TR377, TR447, TR499`
- 含 P/Br 的分子：`TR499`

这一步是正确的。

### 第四步：关键错误发生在计数对象

模型随后输出：

- `Molecules with triple bonds containing P or Br: {'TR499'}`
- `Total atoms in these molecules: 4`

即它统计的是 `TR499` 的全部原子。

### 第五步：忽略了“目标元素原子数”口径

`TR499` 元素列表包含 `['y', 'p', 'h', 'h']`，其中只有 1 个目标元素原子（`p`）。

### 第六步：提交了放大后的计数

最终提交 `4`，与 gold 的 `1` 不匹配。

## 七、正确答案为什么应该是标准答案中的 1 个值 `1`

### 1. triple bond 分子集合可复现

从 `bond.db` 可得：`TR041, TR377, TR447, TR499`。

### 2. 含 P/Br 的只有 `TR499`

在这些分子中，只有 `TR499` 包含目标元素（`p`）。

### 3. 目标元素原子数是 1

`TR499` 的原子元素为 `y, p, h, h`，属于 P/Br 的仅 `p` 一项，因此计数为 1。

这与 gold 完全一致。

## 八、正确查询应该怎么写

示意 SQL / 伪 SQL：

```sql
WITH triple_molecule AS (
  SELECT DISTINCT molecule_id
  FROM bond
  WHERE bond_type = '#'
),
target_molecule AS (
  SELECT DISTINCT a.molecule_id
  FROM atom_csv a
  JOIN triple_molecule t ON a.molecule_id = t.molecule_id
  WHERE LOWER(a.element) IN ('p', 'br')
)
SELECT COUNT(a.atom_id)
FROM atom_csv a
JOIN target_molecule m ON a.molecule_id = m.molecule_id
WHERE LOWER(a.element) IN ('p', 'br');
```

返回应为 `1`。

## 九、本次错误的本质总结

### 1. 计数对象错位

应计“目标元素原子数”，模型计成“目标分子全部原子数”。

### 2. 中间结论正确，终值拼装错误

模型已正确找到 `TR499`，但最后一步口径漂移。

### 3. 跨源查询中缺少最终口径锚定

`bond.db + atom.csv` 联合时没有对“COUNT 的对象”做显式约束。

### 4. 提交前缺少语义复核

题面包含 `containing the element phosphorus or bromine`，应驱动计数对象落在 P/Br 原子。

## 十、改进建议

### 1. 在 COUNT 前强制写出“计数对象声明”

如：`count_target = atoms where element in {p, br}`，避免默认 `count(*)`。

### 2. 对“contains X”类题加入模板

当题面含 `containing element X`，默认计数对象优先绑定到 `X`，不是全体。

### 3. 中间结果到最终答案增加一致性检查

若中间筛选只锁定了分子，提交前再次确认是否还需元素级过滤。

### 4. 对跨源表结构增加显式映射层

避免把“分子筛选条件”和“原子计数条件”混成一步。

## 十一、结论

`task_200` 的失败属于典型“最后一公里口径漂移”。

模型前半段已经正确识别出唯一目标分子 `TR499`，但在最终计数时把“目标元素原子数”误写成“分子总原子数”，从而把正确值 `1` 放大为 `4`。

该问题可通过“计数对象声明 + 提交前语义复核”稳定修复。