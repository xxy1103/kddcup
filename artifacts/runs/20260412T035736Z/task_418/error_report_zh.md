# Task 418 中文错误报告

## 零、三句话先看结论

1. 这道题不是“程序没跑完”，而是“程序跑完了，但判定口径错了”。
2. 模型没有从文档中稳定提取“abnormal creatinine”的语义，而是自行套用了一个通用阈值：`creatinine < 0.6` 或 `> 1.2`。
3. 这个阈值把多名文档中明示或暗示为正常的低值患者误计入，同时漏掉了至少 1 名真正 `<70` 且文本明确异常的患者，最终把答案做成了 `3`，而 gold 是 `1`。

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_418` 中模型为什么提交了 `3`，而标准答案是 `1`。

本题的典型问题是：

- 模型把叙述文本当成半结构化表后，采用了过于宽松的数值阈值
- 将大量本不应计入“abnormal creatinine”的样本纳入
- 同时漏掉了 1 名真正符合条件的患者
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

叙述型检验记录，含 creatinine 数值、修订语句以及 `normal / elevated / impaired / healthy kidney function` 这类肾功能语义描述。

#### `Patient.md`

提供患者性别、出生日期、初诊信息等文本档案，但出生日期的表达方式并不统一，有：

- `born on February 7th, 1966`
- `born in late June of 1968`
- `with a birthdate on the eleventh of November, 1967`

这意味着生日抽取不能只写一种正则格式。

#### `knowledge.md`

提供医学字段的一般语义，但本题重点仍是文档内具体判定语句与 ID 对齐。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 在 `Laboratory.md` 里按患者段落提取 creatinine 与异常语义。
2. 优先依据原文的肾功能语义判定是否 abnormal，而不是直接套固定医学参考阈值。
3. 在 `Patient.md` 中提取同一患者的出生日期。
4. 计算年龄并筛 `age < 70`。
5. 对患者 ID 去重计数。

按评测口径，最终计数应为 `1`。

### 为什么这题不能直接套阈值

因为文档并没有给出一张统一的 creatinine 正常范围表；相反，它经常在患者段落中直接给出结论性描述，例如：

- `These are normal findings.`
- `These values suggest healthy kidney function at this time point.`
- `indicating impaired renal filtration`
- `indicating significant renal dysfunction`

因此本题更像“段落级语义判定”，而不是“查一个固定参考区间”的表格题。

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

### 第一步：工具路径误用，但不构成最终失败主因

模型先尝试对 `doc/Laboratory.md` 走 SQL：

- `execute_context_sql` -> `file is not a database`

随后它改用 Python，说明这一步属于探索失误，不是最终失败原因。

### 第二步：中途代码错误，说明抽取过程不稳定

trace 中还出现：

- `NameError: name 'patient_content' is not defined`

虽然后续修复继续执行，但这说明当时的抽取脚本并不稳。

### 第三步：异常判定被简化成固定阈值

模型最终采用：

- abnormal = `creatinine < 0.6` 或 `> 1.2`

并得到 21 个“异常”候选，其中大量是 `0.4 / 0.5`。

这一步是本题最核心的错误，因为它没有真正遵循文档中的患者段落语义。

### 第四步：抽取结果已经出现明显异常信号，但模型没有回退

trace 显示：

- 早期抽取里出现大量 `Patient None`、`0.0` 等噪声
- 后续仍有 `Patients with abnormal creatinine NOT in patients file: 14`

这其实已经是很强的失败信号，说明：

- 患者切分可能有误
- creatinine 值可能跨段匹配
- patient ID 和生日档案没有稳定对齐

但模型没有因为这些异常而回退抽取策略。

### 第五步：年龄解析也有漏检

这次错误不只是“误计”，还有“漏计”。

例如患者 `4634342` 在 `Laboratory.md` 中被明确写成：

- `creatinine was significantly elevated ... indicating impaired renal filtration`

但模型最后没有把她计入 `<70` 的异常患者里。更合理的解释是：它在 `Patient.md` 里没有稳定识别这位患者的生日表达：

- `with a birthdate on the eleventh of November, 1967`

也就是说，生日解析规则写得过窄，漏掉了真阳性。

### 第六步：提交了被放大的计数

模型最终给出 3 名 `<70` 患者：

- 4923796 (0.5)
- 5452747 (0.4)
- 5065022 (0.4)

并提交 `count=3`。

## 七、为什么 `3` 会被放大成错误答案

### 1. 被误计入的 3 名患者

#### 患者 4923796

文档只写到：

- uric acid `3.6 mg/dL`
- creatinine `0.5 mg/dL`
- urea nitrogen 由 `21.0` 修正为 `11.0 mg/dL`

但原文没有给出 abnormal 肾功能结论，因此不能仅凭 `0.5` 就机械判异常。

#### 患者 5452747

文档明确写到：

- creatinine `0.4 mg/dL`
- `These are normal findings.`

这是本题最典型的反例：数值低，但原文直接判定为正常。

#### 患者 5065022

文档写到：

- uric acid 由 `4.3` 修正为 `3.4 mg/dL`
- urea nitrogen `12.0 mg/dL`
- creatinine `0.4 mg/dL`

但没有出现 `elevated / impaired / dysfunction` 等异常肾功能语义，因此也不应因 `0.4` 被直接纳入 abnormal。

### 2. 被模型漏掉、但更像真阳性的患者

#### 患者 4634342

`Laboratory.md` 中明确写到：

- `creatinine was significantly elevated`
- 最终值为 `1.5 mg/dL`
- `indicating impaired renal filtration`

这是非常明确的异常肾功能语义。

而 `Patient.md` 中对应出生信息是：

- `with a birthdate on the eleventh of November, 1967`

这意味着该患者明显 `<70`，且更符合题目要求。模型没有把这位患者计入，说明它不仅误计了低值患者，也漏掉了真正应计入的异常患者。

### 3. 文本里还有“异常但不应计入 `<70`”的患者

这也能帮助我们理解为什么最终答案不是更大。

#### 患者 3182521

文档明确写到：

- creatinine 由 `2.1` 修正为 `3.1 mg/dL`
- `acute renal impairment`
- `significant reduction in renal clearance`

这是明确异常，但 `Patient.md` 里的出生年是 `1952`，不属于 `<70`。

#### 患者 444499

文档明确写到：

- creatinine 由 `1.6` 修正为 `1.9 mg/dL`
- `significant renal dysfunction`

但 `Patient.md` 里的出生日期是 `January 24th, 1954`，按题目口径不属于“还不到 70 岁”。

### 4. 因此为什么 gold 是 `1`

综合段落语义和年龄条件后，更稳定的候选是：

- 真正 `<70` 且 creatinine 异常：`4634342`
- 异常但年龄不满足：`3182521`、`444499`
- 低值但原文判正常或未支持异常：`4923796`、`5452747`、`5065022` 等

这样就会自然收敛到 `1`，与 gold 一致。

## 八、一个更合理的解题伪代码

```python
# 1) split Laboratory.md into patient-level renal sections
# 2) for each patient section:
#       if text says normal / healthy / within range:
#           mark normal
#       elif text says elevated / impaired / dysfunction / reduced clearance:
#           mark abnormal
#       else:
#           keep as uncertain, do not auto-promote by a generic threshold
# 3) parse birthdays from Patient.md with multiple date templates
# 4) count distinct patient_id where abnormal_creatinine and age < 70

answer = 1
```

关键点有两个：

- abnormal 判定要以患者段落语义和修订后结论为主
- 生日解析要兼容多种英文日期表达

## 九、本次错误的本质总结

### 1. 语义判定被数值阈值替代

叙述型病历里“异常”是语义状态，模型却用固定阈值强行替代。

### 2. 误计与漏计同时存在

这次不是单纯“把人数算多了”，而是：

- 把多个低值正常病例误计入
- 又把真正异常且 `<70` 的病例漏掉

### 3. 生日抽取规则过窄

`Patient.md` 中出生日期并非统一格式，只用一种正则会漏掉诸如：

- `the eleventh of November, 1967`
- `late June of 1968`
- `Christmas Eve of 1986`

这会直接影响年龄筛选结果。

### 4. 候选集合污染明显，但缺少回退机制

大量 `Patient None`、跨段拼接、ID 不在患者表等信号都提示抽取质量问题；模型虽然看到了噪声，但未据此回退判定策略。

### 5. 提交前缺少边界值复核

对 `0.4 / 0.5 / 0.6 / 1.1 / 1.2` 这类边界值，如果不回看原文语义，很容易误判。

## 十、改进建议

### 1. 先做患者级段落切分，再抽值

避免跨段匹配导致的 `Patient None` 与错位数值。

### 2. 异常判定优先使用文本语义

当文本明确写出 `normal`, `healthy kidney function`, `impaired`, `elevated`, `renal dysfunction` 等时，优先采用语义标签。

### 3. 对“未显式判异常”的边界值保持保守

像 `0.4 / 0.5 / 1.1 / 1.2` 这种值，如果文档没有给出明确异常语义，不应直接升级为 abnormal。

### 4. 生日抽取要做多模板兼容

不能只支持 `Month Day, Year` 一种格式；应兼容文字化日期和模糊日期表达。

### 5. 对候选集合做一致性门槛

若“异常患者中大量 ID 不在患者档案”，应强制回退并重抽取。

### 6. 提交前做病例级人工语义审计

对最终被计入答案的每个 patient，至少输出一行：

- patient id
- creatinine 最终值
- 支持 abnormal 的原文短语
- 出生年 / 年龄

这样可以显著降低计数题的无声误判。

## 十一、结论

`task_418` 的失败并不是因为 agent 没有完成任务，而是因为它在最关键的一步上用“固定阈值”替代了“文档语义判定”，同时生日解析又漏掉了真阳性患者。

因此，这次错误的更准确归因是：

- `异常判定口径过宽`
- `生日抽取规则过窄`
- `缺少病例级复核`

最终模型输出 `3`，而按 gold 口径应为 `1`。
