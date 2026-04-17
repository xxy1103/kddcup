# Task 344 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_344` 中模型为什么在接近答案时陷入口径反复，最终未提交结果。

本题最关键的问题是：

- 性别来源不完整
- SQL/CSV 工具路径混用
- FG 异常判定口径反复切换
- 超步结束而非输出错误值

## 二、题目原文与中文翻译

### 题目原文

`Among the male patients who have a normal level of white blood cells, how many of them have an abnormal fibrinogen level?`

### 中文直译

`在白细胞水平正常的男性患者中，有多少人纤维蛋白原水平异常？`

### 更适合分析的中文表述

`先筛男性，再筛 WBC 正常，再按 FG 异常做患者去重计数。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是医学检验条件过滤题，涉及三层过滤：

1. 性别过滤（male）
2. 白细胞是否正常（WBC normal）
3. 纤维蛋白原是否异常（FG abnormal）

最后要求的是“患者人数（distinct patient count）”。

### 2. 这个任务提供了哪些数据

`task_344/context` 主要包含：

- `csv/Laboratory.csv`
- `patient_sex.csv`
- `doc/Patient.md`
- `knowledge.md`

#### `Laboratory.csv`

包含实验室指标，关键字段：

- `ID`
- `WBC`
- `FG`

#### `patient_sex.csv`

提供一批患者的性别映射；该文件中的 `SEX` 均为 `M`，且覆盖并非全量患者。

#### `Patient.md`

文本化患者档案，包含更多患者的性别线索（例如 `4934716` 被描述为 male）。

#### `knowledge.md`

给出通用字段语义，但未直接给出本题 FG 的唯一阈值实现细节。

## 四、这道题正确的求解思路应该是什么

本题应使用“患者级去重计数”流程：

1. 构建 male 患者集合（不能只依赖单一子集文件）。
2. 在 `Laboratory.csv` 中筛 `WBC` 正常记录。
3. 在同一患者维度判断 `FG` 是否异常。
4. 对满足条件的患者做 `COUNT(DISTINCT ID)`。

按评测口径复算，最终应为 `4`。

## 五、模型最终给出了什么答案

本题未生成 `prediction.csv`：

- `artifacts/runs/20260412T035736Z/task_344/prediction.csv` 不存在

运行在 `max_steps` 后结束，失败原因为：

- `Agent did not submit an answer within max_steps.`

标准答案文件为：

- `data/public/output/task_344/gold.csv`

gold 值：

- `COUNT(DISTINCT T1.ID) = 4`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：SQL 路径与数据类型不匹配

模型先尝试对 CSV 走 SQL：

- `SELECT ... FROM csv/Laboratory.csv ...` -> `near "/": syntax error`
- `SELECT ... FROM Laboratory ...`（path 仍指向 CSV）-> `file is not a database`

说明其在 SQLite/CSV 工具边界上出现了混用。

### 第二步：转 Python 后进入口径试探

模型开始做统计，得到：

- male 患者（来自 `patient_sex.csv`）= 92
- male 记录中 WBC 正常记录 = 596

### 第三步：FG 判定反复，结果在 2/3/其他值之间震荡

trace 显示模型多次改阈值与解释方式，并反复观察：

- `FG` 非空很稀疏
- 仅少量 male 患者携带 `FG`
- 统计结果不稳定

### 第四步：关键漏计患者未被纳入性别映射

模型主要依赖 `patient_sex.csv` 子集，遗漏了文档里明确的 male 患者（如 `4934716`），导致计数偏小。

### 第五步：接近答案但未提交

在反复试探后，模型未形成最终稳定口径，最终超步退出。

## 七、正确答案为什么应该是标准答案中的 `4`

gold 文件明确为：

- `4`

按任务口径复算可得到 4 个患者：

- `4618443`
- `4865142`
- `4934716`
- `5092228`

其中导致模型漏计的关键点是：

- `4934716` 在 `Patient.md` 中明确为 male
- 但不在 `patient_sex.csv` 子集中

因此仅依赖子集映射会把答案压低。

## 八、正确查询应该怎么写

可用“先补齐 male，再计数”的写法：

```python
# 伪代码
male_ids = union(
    ids_from_patient_sex_csv_where_SEX_M,
    ids_parsed_from_Patient_md_where_gender_is_male
)

df = Laboratory_csv
cond = (
    df.ID in male_ids
    and 4.0 <= WBC <= 10.0
    and (FG < 2.0 or FG > 4.0)
)

answer = count_distinct(ID under cond)  # -> 4
```

## 九、本次错误的本质总结

### 1. 工具层错误与业务层错误叠加

先在 SQL/CSV 边界上失败，随后在业务口径上继续漂移。

### 2. 依赖不完整主键映射

将 `patient_sex.csv` 当作全量性别表，导致遗漏合法 male 患者。

### 3. 判定口径未冻结

WBC/FG 区间在多次尝试中摇摆，缺乏一次性固定与回归检查。

### 4. 缺少“可提交终态”策略

即使已逼近 gold，也没有及时提交中间最优结果。

## 十、改进建议

### 1. 先做数据覆盖率检查

在使用维表（如 `patient_sex.csv`）前，先验证其对事实表 `ID` 的覆盖比例。

### 2. 明确“患者级”统计粒度

本题必须在 `ID` 层去重，避免被记录数干扰。

### 3. 口径冻结后再迭代

先固定：

- male 来源
- WBC 正常区间
- FG 异常规则

再做一次性复算，不要在执行中频繁改规则。

### 4. 设置超步前兜底提交

当连续多步无新增信息时，提交当前最佳可解释答案，避免空结果失败。

## 十一、结论

`task_344` 的失败不是“完全不会做”，而是“口径与数据覆盖处理不稳，导致未提交”。

模型经历了：

- 工具路径错误
- 阈值反复
- 性别映射漏覆盖

最终在 max steps 内没有完成提交。按标准答案与可复算口径，本题正确值应为 `4`。