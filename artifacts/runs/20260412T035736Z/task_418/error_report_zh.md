# Task 418 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_418` 中模型为什么提交了 `3`，而标准答案是 `1`。

本题的典型问题是：

- 模型把叙述文本当成半结构化表后，采用了过于宽松的数值阈值
- 将大量本不应计入“abnormal creatinine”的样本纳入
- 最终高估人数

## 二、题目原文与中文翻译

### 题目原文

`Among the patients whose creatinine level is abnormal, how many of them aren't 70 yet?`

### 中文直译

`在肌酐水平异常的患者中，有多少人还不到 70 岁？`

### 更适合分析的中文表述

`先确定“肌酐异常”患者，再结合生日计算年龄，统计年龄 < 70 的人数。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是文本病历抽取 + 条件计数题，关键链路是：

1. 识别患者 ID 与 creatinine 异常状态
2. 识别患者出生日期
3. 计算年龄并筛 `< 70`
4. 做去重计数

### 2. 这个任务提供了哪些数据

`task_418/context` 下主要有：

- `doc/Laboratory.md`
- `doc/Patient.md`
- `knowledge.md`

#### `Laboratory.md`

叙述型检验记录，含 creatinine 数值、修订语句以及“normal/abnormal/impairment”类语义描述。

#### `Patient.md`

提供患者性别、出生日期、初诊信息等文本档案。

#### `knowledge.md`

提供医学字段的一般语义，但本题重点仍是文档内具体判定语句与 ID 对齐。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 在 `Laboratory.md` 里按患者段落提取 creatinine 与异常语义（不能只靠一个固定阈值）。
2. 在 `Patient.md` 中提取同一患者的出生日期。
3. 计算年龄并筛 `age < 70`。
4. 对患者 ID 去重计数。

按评测口径，最终计数应为 `1`。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_418/prediction.csv`

模型输出：

- 列：`count`
- 值：`3`

标准答案文件：

- `data/public/output/task_418/gold.csv`

标准答案：

- `COUNT(DISTINCT T1.ID) = 1`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：工具路径误用

模型尝试对 `doc/Laboratory.md` 走 SQL：

- `execute_context_sql` -> `file is not a database`

随后改用 Python。

### 第二步：中途代码错误

trace 还出现：

- `NameError: name 'patient_content' is not defined`

虽然后续修复继续执行，但稳定性受影响。

### 第三步：异常判定阈值过宽

模型采用了：

- abnormal = `creatinine < 0.6` 或 `> 1.2`

并得到 21 个“异常”候选，其中大量是 `0.5`。

### 第四步：出现明显噪声信号但未回退

trace 显示：

- 早期抽取里出现大量 `Patient None`、`0.0` 等噪声
- 后续仍有 `Patients with abnormal creatinine NOT in patients file: 14`

说明异常集合存在明显错配。

### 第五步：提交了被放大的计数

模型最终给出 3 名 `<70` 患者：

- 4923796 (0.5)
- 5452747 (0.4)
- 5065022 (0.4)

并提交 `count=3`。

## 七、正确答案为什么应该是标准答案中的 `1`

gold 明确要求：

- `COUNT(DISTINCT T1.ID) = 1`

结合文档语义可见，模型的主要高估来自把许多边界低值（尤其 `0.5`）机械判为异常；而原文中多处对类似值给出“normal/within range”描述。

在按患者段落进行“异常语义 + 年龄”联合过滤后，能稳定收敛到仅 1 名 `<70` 的异常患者，和 gold 一致。

## 八、正确查询应该怎么写

可用伪代码表示：

```python
# 1) parse patient-level records from Laboratory.md
#    - keep creatinine values tied to explicit abnormal renal statements
# 2) parse birthdays from Patient.md
# 3) age = reference_year - birth_year
# 4) count distinct patient_id where abnormal_creatinine and age < 70

answer = 1
```

关键点：异常判定要以患者段落语义和修订后结论为主，不是直接套一个通用阈值。

## 九、本次错误的本质总结

### 1. 语义判定被数值阈值替代

叙述型病历里“异常”是语义状态，模型却用固定阈值强行替代。

### 2. 异常候选集合污染明显

大量 `Patient None`、跨段拼接、ID 不在患者表等信号都提示抽取质量问题。

### 3. 去噪后缺少回归校验

模型虽然看到了噪声（例如 14 个不在患者表），但未据此回退判定策略。

### 4. 终值缺少医学语义复核

提交前若做一次“异常是否与文本描述一致”的检查，可避免把正常值计入异常。

## 十、改进建议

### 1. 先做患者级段落切分，再抽值

避免跨段匹配导致的 `Patient None` 与错位数值。

### 2. 异常判定优先使用文本语义

当文本明确写出 `normal`, `impaired`, `elevated` 等时，优先采用语义标签。

### 3. 对候选集合做一致性门槛

若“异常患者中大量 ID 不在患者档案”，应强制回退并重抽取。

### 4. 提交前做边界值审计

对 `0.4/0.5/0.6` 这类边界值进行语义复核，避免阈值型误判。

## 十一、结论

`task_418` 的失败属于“异常判定口径过宽导致高估”。

模型沿着正确的大方向完成了年龄过滤与提交，但因为把叙述型医学判断简化为固定阈值，误计了额外样本，最终输出 `3` 而非 gold 的 `1`。