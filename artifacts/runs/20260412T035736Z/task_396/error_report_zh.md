# Task 396 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_396` 中模型为什么在大量解析后仍未提交答案。

本题的主要失败形态是：

- 数据并非标准表结构（叙述型文档）
- 模型做了大量正则抽取但没有形成稳定联结
- 在 max_steps 前未输出最终百分比

## 二、题目原文与中文翻译

### 题目原文

`In superheroes with height between 150 to 180, what is the percentage of heroes published by Marvel Comics?`

### 中文直译

`在身高 150 到 180 的超级英雄中，由 Marvel Comics 发行的英雄占比是多少？`

### 更适合分析的中文表述

`先筛身高区间 [150,180] 的英雄，再计算其中 publisher=Marvel Comics 的比例。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是比例计算题，核心公式是：

\[
\text{Percentage} = \frac{\text{Marvel heroes in range}}{\text{All heroes in range}} \times 100
\]

### 2. 这个任务提供了哪些数据

`task_396/context` 下主要有：

- `doc/superhero.md`
- `json/publisher.json`
- `knowledge.md`

#### `superhero.md`

叙述型文本，包含大量英雄档案、ID、身高、publisher 相关信息，但不是规整数据表。

#### `publisher.json`

提供 `publisher id -> publisher_name` 映射，其中：

- `id = 13` 对应 `Marvel Comics`

#### `knowledge.md`

给出语义说明：

- 身高字段应为 `height_cm`
- 出版社字段为 `publisher_id`

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 从 `superhero.md` 抽取结构化三元组：`(id, height_cm, publisher_id)`。
2. 过滤 `150 <= height_cm <= 180`。
3. 用 `publisher.json` 将 `publisher_id` 映射为 `publisher_name`。
4. 计算 `Marvel Comics` 占比并输出百分数。

按标准答案，本题结果应为：`54.83870967741935`。

## 五、模型最终给出了什么答案

本题没有生成 `prediction.csv`：

- `artifacts/runs/20260412T035736Z/task_396/prediction.csv` 不存在

失败原因：

- `Agent did not submit an answer within max_steps.`

标准答案文件：

- `data/public/output/task_396/gold.csv`

gold 值：

- `54.83870967741935`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：先查结构，确认没有可直接 SQL 的数据库

模型检查上下文后发现：

- 只有 `knowledge.md`, `doc/superhero.md`, `json/publisher.json`
- 没有可直接查询的 DB 表

### 第二步：进入大规模正则抽取

trace 显示模型反复抽取：

- 身高候选（例如 85 个匹配）
- publisher 相关候选（例如 132 个匹配）
- ID 片段

### 第三步：出现“名称匹配”与“ID 匹配”错位

模型曾直接在文档里搜 `Marvel Comics`，结果为 0 次，说明其一度偏向名称关键词，而非先通过 `publisher_id` 映射。

### 第四步：后期开始构造结构化片段但仍未收敛

末段已经出现：

- 解析出一批 biometric 条目
- 解析出一批 publisher 条目

但两者尚未形成完整稳定联结并完成最终比例计算。

### 第五步：超步终止

在反复解析后未触发 `answer` 提交，最终超出步数限制。

## 七、正确答案为什么应该是标准答案中的 `54.83870967741935`

gold 给出的公式头部是：

- `CAST(COUNT(CASE WHEN T2.publisher_name = 'Marvel Comics' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)`

说明口径明确是：

- 分母：身高区间内全部英雄数
- 分子：同区间且 publisher 为 Marvel Comics 的英雄数

因此官方评测值为：

- `54.83870967741935`

## 八、正确查询应该怎么写

可用 Python 伪代码表示：

```python
# 1) parse triples from superhero.md: (id, height_cm, publisher_id)
# 2) filter 150 <= height_cm <= 180
# 3) map publisher_id via publisher.json
# 4) percentage = marvel_count * 100.0 / total_count

percentage = 54.83870967741935
```

## 九、本次错误的本质总结

### 1. 任务是“半结构化解析”，模型未形成稳定抽取协议

在叙述文本中抓数字容易，但抓“正确字段绑定关系”更难。

### 2. 关键词路径与键值路径混用

直接搜 `Marvel Comics` 名称会漏掉大量通过 `publisher_id` 表达的信息。

### 3. 中间结果多，终态结果缺失

模型有大量中间抽取产物，却没有进入最终比例计算和提交。

### 4. 缺少收敛触发器

当 `(id,height,publisher)` 已达到可计算阈值时，应立即计算并提交，而不是继续扩展匹配。

## 十、改进建议

### 1. 对半结构化文档采用“字段绑定优先”策略

优先保证同一 ID 下 `height` 与 `publisher_id` 的绑定，再做统计。

### 2. 统一使用 ID 映射到名称

先按 `publisher_id` 聚合，再映射 `publisher_name`，避免名称关键词漏检。

### 3. 增加中间可计算门槛

当已能构建有效分子分母时，立刻计算一次并保留可提交结果。

### 4. 对长文本任务加入早收敛机制

连续多轮解析若新增有效三元组很少，应提前停止扩展并提交当前最优答案。

## 十一、结论

`task_396` 的失败属于“解析过程很充分，但缺少最终收敛提交”。

模型正确识别了问题需要身高与出版社联合判断，也抽取了大量候选信息，但未在步数内完成稳定联结与比例输出。按标准答案，本题正确值应为 `54.83870967741935`。