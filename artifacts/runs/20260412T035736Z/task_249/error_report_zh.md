# Task 249 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_249` 中模型为什么只命中了一半答案（Recall=0.5）。

目标是把问题拆清楚：

- 模型为什么把 `avg_upvotes` 算成了 `340.0`
- 为什么 `avg_age` 又恰好是正确的 `34.083333333333336`
- 正确聚合口径到底是什么

## 二、题目原文与中文翻译

### 题目原文

`What is the average of the up votes and the average user age for users creating more than 10 posts?`

### 中文直译

`对于发帖数超过 10 的用户，他们的 up votes 平均值和年龄平均值分别是多少？`

### 更适合分析的中文表述

`先筛出发帖数 > 10 的用户集合，再分别计算该集合上的 UpVotes 平均值与 Age 平均值（两个指标各自按非空值独立求平均）。`

这道题最容易错在“两个平均值是否必须使用同一子集”。

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是一个跨源聚合题：

- 发帖记录在 JSON
- 用户属性在 SQLite

需要先从帖子侧筛用户，再在用户侧做两个指标聚合。

### 2. 这个任务提供了哪些数据

`task_249/context` 下包含：

- `db/users.db`
- `json/posts.json`
- `knowledge.md`

#### `posts.json`

关键字段：`OwnerUserId`。

本题先按 `OwnerUserId` 统计发帖数，筛出 `count > 10` 的用户。

#### `users.db`

关键字段：

- `Id`
- `UpVotes`
- `Age`

用于计算两个平均值。

#### `knowledge.md`

语义辅助文档。本题关键点是“指标独立聚合口径”：

- `AVG(UpVotes)` 只需要 `UpVotes` 非空
- `AVG(Age)` 只需要 `Age` 非空

不应强制两者都非空才入样本。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 从 `posts.json` 按 `OwnerUserId` 统计帖子数。
2. 过滤得到 `post_count > 10` 的用户集合。
3. 在 `users.db` 中取这些用户的 `UpVotes` 与 `Age`。
4. 分别计算两个平均值：
   - `AVG(UpVotes)`：对 `UpVotes IS NOT NULL` 的样本
   - `AVG(Age)`：对 `Age IS NOT NULL` 的样本
5. 输出两列单行结果。

这道题的关键不是 join，而是“两个指标的样本集是否独立”。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_249/prediction.csv`

模型输出：

- 列：`avg_upvotes, avg_age`
- 行：1
- 值：`340.0, 34.083333333333336`

标准答案文件：

- `data/public/output/task_249/gold.csv`

标准答案：

- 列：`AVG(T1.UpVotes), AVG(T1.Age)`
- 行：1
- 值：`182.2832618025751, 34.083333333333336`

可以看到：

- `avg_age` 命中
- `avg_upvotes` 严重偏高（340.0 vs 182.283...）

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：先尝试纯 SQL，但数据源假设错误

模型在 `users.db` 上执行：

```sql
SELECT COUNT(*) as post_count, AVG(UpVotes) as avg_upvotes, AVG(Age) as avg_age
FROM users
WHERE Id IN (
  SELECT UserId FROM posts GROUP BY UserId HAVING COUNT(*) > 10
)
```

报错：`no such table: posts`。

原因是 `posts` 不在 SQLite，而在 `posts.json`。

### 第二步：切换到 Python，先做 posts 计数

模型改为读取 `posts.json`，统计 `OwnerUserId` 的发帖数，并筛出 `>10` 的用户，方向正确。

### 第三步：第一次聚合因 NULL 崩溃

在 step 12，模型直接 `sum(ages_list)`，`ages_list` 含 `None`，报错：

`unsupported operand type(s) for +: 'int' and 'NoneType'`。

### 第四步：修复时引入了新的口径错误

step 14 的修复代码写成：

```python
if upvotes is not None and age is not None:
    upvotes_list.append(upvotes)
    ages_list.append(age)
```

即只有 `UpVotes` 和 `Age` 同时非空才入样本。

### 第五步：导致两个指标被绑定到同一交集样本

执行输出显示：

- `Users with more than 10 posts: 1165`
- `Users with valid data (both UpVotes and Age not None): 312`
- `Average UpVotes: 340.0`
- `Average Age: 34.083333333333336`

这说明 `avg_upvotes` 被错误限制在 312 人交集上，抬高了均值。

### 第六步：提交前没有做“指标独立口径”校验

模型直接提交 `340.0, 34.0833...`，没有检查 `AVG(UpVotes)` 是否应使用更大样本。

## 七、正确答案为什么应该是标准答案中的 1 行 `182.2832618025751, 34.083333333333336`

### 1. 用户筛选基集是 1165 人

由 `posts.json` 统计可得：`post_count > 10` 的用户数是 1165。

### 2. 两个指标的非空样本规模不同

在这 1165 人中：

- `UpVotes` 非空：1165 人
- `Age` 非空：312 人

### 3. 指标独立计算可复现 gold

分别聚合得到：

- `AVG(UpVotes) = 182.2832618025751`
- `AVG(Age) = 34.083333333333336`

这与 `gold.csv` 完全一致。

模型给出的 `340.0` 则是把 `AVG(UpVotes)` 错算成了“交集 312 人”的均值。

## 八、正确查询应该怎么写

示意 SQL / 伪 SQL：

```sql
WITH active_users AS (
  SELECT OwnerUserId AS uid
  FROM posts_json
  GROUP BY OwnerUserId
  HAVING COUNT(*) > 10
)
SELECT
  AVG(CASE WHEN u.UpVotes IS NOT NULL THEN u.UpVotes END) AS avg_upvotes,
  AVG(CASE WHEN u.Age IS NOT NULL THEN u.Age END) AS avg_age
FROM users u
JOIN active_users a ON CAST(u.Id AS TEXT) = CAST(a.uid AS TEXT);
```

关键是：两个 `AVG` 各自处理自己的空值，不强制交集。

## 九、本次错误的本质总结

### 1. 空值修复策略引入口径耦合

为了避免 `None` 报错，模型把两个指标强行绑定到同一交集样本。

### 2. 指标独立性被破坏

`AVG(UpVotes)` 本应覆盖 1165 人，结果被缩到 312 人。

### 3. 错误具有迷惑性

`avg_age` 恰好正确，容易让模型误以为整体逻辑都对。

### 4. 缺少样本规模审计

提交前没有输出并校验各指标的样本数，导致口径偏差未被发现。

## 十、改进建议

### 1. 为每个指标单独维护样本集

不要复用统一 `valid_rows`；应使用 `upvotes_rows` 与 `age_rows` 两套过滤。

### 2. 聚合前打印样本规模

在答案提交前强制输出：

- `n_upvotes`
- `n_age`

若两者异常相等或异常过小，触发复核。

### 3. 对空值修复采用“局部过滤”而非“交集过滤”

避免 `if upvotes is not None and age is not None` 这类一刀切条件。

### 4. 引入口径回归测试

把本题加入“独立 AVG 口径”专项回归，防止以后再次把两个指标绑定到交集样本。

## 十一、结论

`task_249` 的核心错误是“空值处理导致指标口径耦合”。

模型先正确找到 `post_count > 10` 的用户集合，但在修复 `None` 报错时，把 `AVG(UpVotes)` 和 `AVG(Age)` 强行限定到 `UpVotes` 与 `Age` 同时非空的 312 人，最终把 `avg_upvotes` 从正确的 `182.283...` 拉高到 `340.0`。

该问题不是数据获取失败，而是聚合口径设计错误；通过“指标独立样本 + 样本规模审计”可以稳定修复。