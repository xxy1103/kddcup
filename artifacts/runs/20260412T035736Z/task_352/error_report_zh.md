# Task 352 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_352` 中模型为什么在已经接近正确映射的情况下仍未提交答案。

本题属于典型的“文本映射题”：

- 预算金额在 `budget.md`
- 会议名称在 `event.csv`
- 需要通过 `event_id` 做跨文档映射后再计算比值

## 二、题目原文与中文翻译

### 题目原文

`How many times was the budget in Advertisement for "Yearly Kickoff" meeting more than "October Meeting"?`

### 中文直译

`Advertisement 类预算中，“Yearly Kickoff” 会议的预算是 “October Meeting” 的多少倍？`

### 更适合分析的中文表述

`先找到两个会议对应的 Advertisement 预算金额，再做 Yearly Kickoff / October Meeting 的比值。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是“跨文档键关联 + 数值计算”任务，包含两层映射：

1. `budget.md` 内：预算条目 -> 事件 ID
2. `event.csv` 内：事件 ID -> 事件名称

### 2. 这个任务提供了哪些数据

`task_352/context` 包含：

- `doc/budget.md`
- `csv/event.csv`
- `knowledge.md`

#### `budget.md`

叙述性文本，包含：

- 预算条目（含 category、amount）
- 与事件记录的链接（`rec...`）

#### `event.csv`

结构化事件表，关键字段：

- `event_id`
- `event_name`

其中包含：

- `Yearly Kickoff`
- `October Meeting`

#### `knowledge.md`

提供预算与事件语义背景，但本题的关键在 `budget.md` 中的映射细节。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 从 `budget.md` 提取 Advertisement 预算条目及其 `amount`。
2. 在同一文档中提取每个预算条目关联的 `event_id`。
3. 在 `event.csv` 把 `event_id` 映射成 `event_name`。
4. 取出：
   - `Yearly Kickoff` 的金额
   - `October Meeting` 的金额
5. 计算比值：
   - `Yearly Kickoff / October Meeting`

按标准答案，应得到：`2.727272727272727`。

## 五、模型最终给出了什么答案

本题没有生成 `prediction.csv`：

- `artifacts/runs/20260412T035736Z/task_352/prediction.csv` 不存在

失败原因：

- `Agent did not submit an answer within max_steps.`

标准答案文件：

- `data/public/output/task_352/gold.csv`

gold 值：

- `2.727272727272727`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：识别到需要联结 event 与 budget

模型读取了 `event.csv` 与 `budget.md`，方向正确。

### 第二步：在 `budget.md` 直接搜会议名失败

trace 显示模型尝试直接搜：

- `Yearly Kickoff`
- `October Meeting`

得到“not found”，因为 `budget.md` 更多是通过 `event_id` 间接引用。

### 第三步：进入高频正则试探

模型反复做：

- 数字抽取
- 上下文片段匹配
- 预算条目枚举

并多次接近正确映射。

### 第四步：执行中出现代码错误

trace 出现一次典型错误：

- `name 're' is not defined`

虽然后续继续执行，但流程稳定性下降。

### 第五步：接近正确结果但未提交

trace 后段已经识别出关键映射线索，仍未进入最终 `answer` 提交，导致超步失败。

## 七、正确答案为什么应该是标准答案中的 `2.727272727272727`

可复核得到两条关键映射：

- `recTxecmwIhCdIKvl` -> `event_id=recggMW2eyCYceNcy` -> `October Meeting`，金额 `55`
- `recvKTAWAFKkVNnXQ` -> `event_id=recykdvf4LgsyA3wZ` -> `Yearly Kickoff`，金额 `150`

因此：

\[
\frac{150}{55} = 2.727272727272727
\]

与 gold 完全一致。

## 八、正确查询应该怎么写

可用 Python 方案（示意）：

```python
# 1) 从 budget.md 抽取 Advertisement 预算条目: (budget_id, amount, linked_event_id)
# 2) 用 event.csv 映射 linked_event_id -> event_name
# 3) ratio = amount['Yearly Kickoff'] / amount['October Meeting']

ratio = 150 / 55
# 2.727272727272727
```

## 九、本次错误的本质总结

### 1. 错把“直接字符串匹配”当主路径

本题核心是 ID 映射，不是会议名在 `budget.md` 的直接出现。

### 2. 解析策略过于分散

在金额抽取、段落匹配、模式试探之间频繁切换，导致收敛变慢。

### 3. 代码细节错误放大了超步风险

`re` 未导入这类错误虽可修复，但会打断推理节奏。

### 4. 缺少“找到关键数值即提交”的终态策略

在关键映射已形成时未及时提交，是本题失败的直接原因。

## 十、改进建议

### 1. 优先建立“键映射图”

先明确：`budget_id -> event_id -> event_name`，再做金额计算。

### 2. 对叙述文档采用分阶段解析

先提实体（ID、amount），再提关系（link），最后做计算，避免混做。

### 3. 加入代码执行前自检

对常见依赖（如 `import re`）做预检查，减少中途中断。

### 4. 增加提交触发条件

当两个目标会议金额都已确定时，强制进入提交流程。

## 十一、结论

`task_352` 的失败是“映射已接近完成但未提交”的超步型失败。

模型路径基本正确，但在文本解析阶段过度迭代、缺少终态提交策略，最终未产出答案。按标准映射与金额计算，本题正确值为 `2.727272727272727`。