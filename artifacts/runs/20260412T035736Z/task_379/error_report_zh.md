# Task 379 中文错误报告

## 一、报告目的

这份报告用于分析 `artifacts/runs/20260412T035736Z/task_379` 中模型为何提交了大量逐分子明细，而不是标准答案要求的元素汇总结果。

本题的关键失误是“输出粒度错位”：

- 题目要的是第 4 原子的毒理元素统计结果
- 模型提交成了“每个分子一行”的明细表

## 二、题目原文与中文翻译

### 题目原文

`Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.`

### 中文直译

`统计每个致癌分子中第 4 个原子的毒理元素。`

### 更适合分析的中文表述

`先筛出致癌分子，再取每个分子的第4个原子元素，最后输出元素层面的统计结果。`

## 三、任务和数据集背景说明

### 1. 这是什么类型的任务

这是化学结构统计题，包含三个动作：

1. 识别 carcinogenic 分子
2. 在原子序列中定位“第4个原子”
3. 按元素做 tally（汇总）

### 2. 这个任务提供了哪些数据

`task_379/context` 主要包含：

- `doc/molecule.md`
- `csv/atom.csv`
- `knowledge.md`

#### `molecule.md`

以叙述文本给出各分子的毒理判断（含部分“先否后改”的修订语句）。

#### `atom.csv`

提供原子明细：

- `atom_id`
- `molecule_id`
- `element`

可用于按原子序号提取第 4 原子元素。

#### `knowledge.md`

给出了 molecule/atom/bond 的关系语义，支持“先筛分子，再抽取原子位次”的路径。

## 四、这道题正确的求解思路应该是什么

正确流程应为：

1. 从 `molecule.md` 稳定识别致癌分子集合。
2. 在 `atom.csv` 中按 `molecule_id` 分组，并按原子顺序定位第 4 原子。
3. 取第 4 原子的 `element` 做 tally。
4. 输出元素层面的结果（本题 gold 为元素列表）。

## 五、模型最终给出了什么答案

模型输出文件：

- `artifacts/runs/20260412T035736Z/task_379/prediction.csv`

模型输出结构：

- 列：`molecule_id, element`
- 行数：`90`

即每个分子输出一行第4原子元素，而不是元素汇总结果。

标准答案文件：

- `data/public/output/task_379/gold.csv`

gold 只有一列 `element`，共 7 行：

- `c`
- `br`
- `cl`
- `s`
- `o`
- `n`
- `f`

## 六、模型在 Trace 中是如何一步步出错的

### 第一步：从文本抽取致癌分子集合

trace 显示：

- 先得到 84 个候选
- 后扩展到 90 个候选

该阶段已出现文本修订语句带来的噪声累积风险。

### 第二步：工具路径有一次无效尝试

模型对 `csv/atom.csv` 调 `inspect_sqlite_schema`，报错：

- `file is not a database`

随后改回 Python 读取，继续执行。

### 第三步：形成“每分子一行”的中间结果

模型构造了 `molecule_id + 第4原子element` 的明细表。

### 第四步：未执行 tally/去重收敛，直接提交

最终在 `answer` 阶段提交了 90 行二列表，row_count=90。

这一步就是本题失败的直接原因。

## 七、正确答案为什么应该是标准答案中的 7 个元素

gold 只保留元素层面的输出，说明评测口径不是“逐分子明细”，而是“元素汇总结果”。

也就是说，本题最终应回到元素维度：

- 去掉分子粒度
- 输出第4原子可能出现的目标元素集合

因此标准答案是：

- `c, br, cl, s, o, n, f`

## 八、正确查询应该怎么写

可用伪代码表达：

```python
# 1) carcinogenic_molecules = parse_from_molecule_md(...)
# 2) fourth_atom_element = atom_csv.groupby(molecule_id).nth(3)['element']
# 3) keep only carcinogenic molecules
# 4) result = sorted(unique(elements))

# 输出单列 element
```

若使用 SQL 思路，也应在最后使用 `DISTINCT element` 或按 `element` 聚合。

## 九、本次错误的本质总结

### 1. 输出粒度错位

模型停在“分子粒度中间表”，未收敛到“元素粒度最终表”。

### 2. 文本抽取致癌集合存在噪声

`molecule.md` 中大量“修订叙述”会使集合膨胀，进一步放大明细输出。

### 3. 缺少最终结果模式检查

题目关键词 `tally` 应触发“聚合/汇总”动作，但模型未执行。

### 4. 提交前未对齐 gold 形状

提交为 2 列 90 行，而 gold 是 1 列 7 行，模式差异明显。

## 十、改进建议

### 1. 把“中间表”与“提交表”强制分离

中间表可保留分子维度，但提交前必须显式聚合到题目要求粒度。

### 2. 对 `tally/count/aggregate` 关键词加规则

遇到这类词，默认执行 `group by` 或 `distinct` 收敛。

### 3. 文本修订场景增加去噪策略

对“先否后改”的叙述建立状态机，避免重复或误收录。

### 4. 提交前做输出形状校验

自动检查列数、行数、粒度是否与任务类型匹配。

## 十一、结论

`task_379` 的失败本质是“会算中间结果，但没做最后一步聚合”。

模型已经完成了第4原子提取，但将分子明细直接提交，导致与 gold 的元素汇总口径不一致。正确输出应为 7 个元素：`c, br, cl, s, o, n, f`。