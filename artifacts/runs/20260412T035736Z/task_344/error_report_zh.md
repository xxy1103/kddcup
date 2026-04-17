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

### 3. 为什么这题读起来容易绕

这题难点不在于公式复杂，而在于它同时埋了 3 个容易混淆的点：

- `male` 信息不是只在一个文件里。`patient_sex.csv` 提供了一批男性 ID，但 `Patient.md` 里还有额外的 male 描述。
- `WBC normal` 和 `FG abnormal` 不是同样容易判。`WBC` 看起来更像常规数值筛选，而 `FG` 在这份数据里非常稀疏，不能机械套教材阈值。
- 题目问的是“有多少患者”，所以最后统计粒度是患者级，不是实验记录级。

换句话说，这题表面像一道简单筛选题，实际更像：

`先补全男性患者集合，再在患者层判断“是否有正常 WBC”以及“是否出现异常 FG 信号”，最后做 distinct patient count。`

## 四、这道题正确的求解思路应该是什么

本题应使用“患者级去重计数”流程：

1. 构建 male 患者集合，不能只依赖 `patient_sex.csv`。
2. 在 `Laboratory.csv` 中先回答“哪些男性患者出现过正常 WBC”。
3. 再回答“这些患者里，哪些人出现过异常 FG 信号”。
4. 最后对患者 ID 去重计数。

按评测口径复算，最终应为 `4`。

这里要特别说明：

- 从 trace 看，模型一直没有把 FG 的任务口径稳定下来。
- 从 `gold=4` 以及数据分布反推，一个与 gold 一致的最小解释是：这题更接近“患者级 FG 异常信号识别”，而不是简单把所有 FG 当普通连续变量后硬套 `2.0-4.0` 这种通用参考区间。

下面的错误分析重点放在“模型是怎么一步步偏掉的”，而不是把某一个临床阈值伪装成文档里明写的规则。

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

### 第一步：先读到了关键文件，但很早就出现了一次空转

前面几步里，模型其实已经读了：

- `csv/Laboratory.csv`
- `patient_sex.csv`
- `doc/Patient.md`
- `knowledge.md`

方向并没有错，它知道这题涉及：

- 性别
- `WBC`
- `FG`

但在真正开始计算前，trace 很早就出现了一次空响应，随后系统插入修复提示：

- `Your previous response stopped with no content and no tool call.`

这说明模型虽然读到了材料，但没有立刻把思路收敛成一个稳定的执行计划。

### 第二步：它先想直接用 SQL 查 CSV，结果在工具边界上绊住了

模型随后尝试对 CSV 走 SQL：

- `SELECT ... FROM csv/Laboratory.csv ...` -> `near "/": syntax error`
- `SELECT ... FROM Laboratory ...`（path 仍指向 CSV）-> `file is not a database`

这里的问题不是题意理解错了，而是工具类型搞混了：

- `Laboratory.csv` 是普通 CSV
- 不是 SQLite 数据库

所以它还没开始真正分析 `WBC/FG`，就在工具层先损失了几步。

### 第三步：切到 Python 后，它先做了一个“看起来合理、但其实不完整”的男性集合

转 Python 以后，模型先读出表结构，确认了：

- `Laboratory.csv` 里确实有 `WBC` 和 `FG`
- `patient_sex.csv` 有 `ID, SEX`

然后它直接把：

- `patient_sex.csv` 中 `SEX == 'M'`

当成完整的男性患者集合，并得到：

- `Number of male patients: 92`
- `Number of lab records for males: 1008`

问题在这里已经埋下了：

- 这个 male 集合只是来自 `patient_sex.csv`
- 后面即使它再读 `Patient.md`，后续统计代码依然继续使用这个 92 人子集

也就是说，模型从一开始就把“男性全集”缩小成了“`patient_sex.csv` 里出现过的男性”。

### 第四步：它把 WBC 先固定成 `4.0-10.0`，这一步相对稳定，但统计粒度开始摇摆

模型随后把正常白细胞范围固定成：

- `4.0 <= WBC <= 10.0`

并先后得到几组结果：

- `Male records with normal WBC (4.0-10.0): 596`
- `Unique male patients with normal WBC: 21`

这里虽然 `WBC` 口径相对稳定，但已经出现了一个新的混淆：

- 有时它在看“记录数” `596`
- 有时它在看“患者数” `21`

而题目真正要的其实是患者级计数。

### 第五步：FG 一上来就被它套进了错误的通用医学阈值里

模型先试图把 FG 当成普通临床连续变量，直接借用常见参考范围，输出里出现了：

- `Normal FG range: 2.0-4.0`

但同一个 trace 很快又显示：

- `FG statistics: Min: 23.8, Max: 106.5`

这两件事放在一起，其实已经在提醒它：

- 这份数据里的 `FG` 数值尺度并不支持直接套 `2.0-4.0`

可模型没有在这里及时停下来重建规则，而是继续往下算，于是得到一个非常不自然的结果：

- `Among normal WBC records, those with abnormal FG: 596`

这个数字之所以危险，不是因为“596 一定错”，而是因为它和前面的数据稀疏性明显矛盾，说明模型当时的 FG 判定逻辑已经失真了。

### 第六步：它意识到不对后开始回撤，但回撤方向也不稳定

发现 `2.0-4.0` 和当前数据规模对不上后，模型开始改口径，转而检查：

- `FG` 非空记录到底有多少
- 正常 WBC 的男性里到底谁真的有 FG 值

于是它又得到另一组关键输出：

- `FG values for male records with normal WBC: 36.1, 43.8`
- `FG null count: 594`
- `FG not null count: 2`

然后它又进一步写出了一个新的临时假设：

- `If abnormal = non-null FG: 2`

这一步说明模型已经从“通用医学阈值”跳到了另一个极端：

- 不再按数值范围判 abnormal
- 而是把“只要 FG 有值”近似当成“异常”

这当然比前面的 `596` 更接近一个可解释的数量级，但它仍然不是一个经过验证的稳定规则，只是一次临时补救。

### 第七步：它又换了一个视角，从“记录”切到“患者”，于是答案从 2 摇到了 3

接下来模型不再只看“正常 WBC 的记录里有哪些 FG”，而是改成看：

- 哪些男性患者整体上有 FG 记录
- 这些患者是否“曾经有过”正常 WBC

于是它得到：

- `Male patients with FG values:`
- `Patient 4865142 ... has_normal_wbc=True`
- `Patient 4618443 ... has_normal_wbc=True`
- `Patient 5092228 ... has_normal_wbc=True`

也就是说，在它当前只依赖 `patient_sex.csv` 的男性子集里，它一度已经看到了 3 个候选患者。

这就是为什么这题读 trace 时会让人觉得“模型像在反复横跳”：

- 按记录级，它刚刚说是 `2`
- 按患者级，它又变成了 `3`

并不是数据突然变了，而是它自己在切换统计粒度。

### 第八步：这时真正关键的漏计点出现了，`4934716` 没被它纳入 male 集合

`Patient.md` 中明确写着：

- `The subject identified as 4934716 is a male ...`

而 `4934716`：

- 不在 `patient_sex.csv`
- 但在 `Laboratory.csv` 中同时有正常 WBC 和多次非空 FG

这意味着如果 male 集合只来自 `patient_sex.csv`，模型最多只能在：

- `4618443`
- `4865142`
- `5092228`

这 3 个已知男性 FG 患者里打转，天然到不了 gold 的 `4`。

更关键的是，后段 trace 虽然重新读取了 `Patient.md`，但它后续 Python 代码里仍然是：

- `male_ids = set(sex_df[sex_df['SEX'] == 'M']['ID'])`

也就是说，它读到了新证据，却没有把新证据真正接回计算流程。

### 第九步：它最后进入“分布猜阈值”模式，继续消耗步数，但没有真正解决问题

在发现自己既有 `2` 又有 `3` 之后，模型没有回到“先补全 male 集合，再冻结患者级规则”这条主线，而是继续做一系列分布试探：

- `Values < 30: 59, Values > 30: 395`
- `Values < 35: 146, Values > 35: 308`
- `Values < 40: 245, Values > 40: 209`
- `Values < 45: 299, Values > 45: 156`

这说明它已经进入一种典型的“看到分布就想猜阈值”的状态。

问题在于，这些分布信息本身并不能回答两个更关键的问题：

- male 集合是不是完整的
- 题目到底按记录级还是患者级统计

所以它虽然做了很多数值试探，但并没有真正修复主错误。

### 第十步：多次空转后仍未调用 `answer`，最终以超步结束

trace 中后半段又多次出现：

- `Your previous response stopped with no content and no tool call.`

这说明模型已经反复进入“知道还有问题，但没有形成下一步稳定动作”的状态。

最终它没有生成 `prediction.csv`，也没有调用 `answer`，运行以：

- `Agent did not submit an answer within max_steps.`

结束。

## 七、正确答案为什么应该是标准答案中的 `4`

gold 文件明确为：

- `4`

从数据与 gold 交叉看，一个与标准答案一致、同时也最能解释 trace 混乱来源的患者集合是：

- `4618443`
- `4865142`
- `4934716`
- `5092228`

这里需要区分“确定事实”和“基于 gold 的推断”：

- `确定事实`：gold 就是 `4`
- `确定事实`：`4934716` 在 `Patient.md` 中明确为 male，但不在 `patient_sex.csv`
- `推断`：这题的评测口径更接近患者级异常信号统计，而不是简单把 FG 套进模型自己发明的通用阈值

其中导致模型明显漏计的关键点是：

- `4934716` 在 `Patient.md` 中明确为 male
- 但不在 `patient_sex.csv` 子集中

因此，仅依赖 `patient_sex.csv` 的男性子集，答案天然会被压低。

## 八、正确查询应该怎么写

更稳妥的写法不是先拍脑袋定 FG 阈值，而是先把患者集合和统计粒度固定下来：

```python
# 伪代码
male_ids = union(
    ids_from_patient_sex_csv_where_SEX_M,
    ids_parsed_from_Patient_md_where_gender_is_male
)

wbc_normal_ids = distinct IDs where:
    ID in male_ids
    and 4.0 <= WBC <= 10.0

fg_abnormal_ids = distinct IDs where:
    ID in male_ids
    and FG shows abnormal signal under the task-consistent rule

answer = count_distinct(wbc_normal_ids ∩ fg_abnormal_ids)  # -> 4
```

这段伪代码故意没有把 FG 写死成 `2.0-4.0`，因为这正是 trace 中模型犯错的地方。

## 九、本次错误的本质总结

### 1. 工具层错误与业务层错误叠加

先在 SQL/CSV 边界上失败，随后在业务口径上继续漂移。

### 2. 依赖不完整主键映射

将 `patient_sex.csv` 当作全量性别表，导致遗漏合法 male 患者。

### 3. 同时混淆了“记录级统计”和“患者级统计”

一会儿看 `596` 条记录，一会儿看 `21` 个患者，一会儿又看 `2` 个非空 FG 记录和 `3` 个候选患者，导致答案来回漂移。

### 4. 判定口径未冻结

WBC/FG 区间在多次尝试中摇摆，缺乏一次性固定与回归检查。

### 5. 缺少“可提交终态”策略

即使已逼近 gold，也没有及时提交中间最优结果。

## 十、改进建议

### 1. 先做数据覆盖率检查

在使用维表（如 `patient_sex.csv`）前，先验证其对事实表 `ID` 的覆盖比例。

### 2. 明确“患者级”统计粒度

本题必须在 `ID` 层去重，避免被记录数干扰。

### 3. 先补全 male 集合，再讨论 FG 口径

如果连男性患者全集都不完整，后面无论怎么调 FG 阈值，结果上限都已经被锁死了。

### 4. 口径冻结后再迭代

先固定：

- male 来源
- WBC 正常区间
- FG 异常规则

再做一次性复算，不要在执行中频繁改规则。

### 5. 设置超步前兜底提交

当连续多步无新增信息时，提交当前最佳可解释答案，避免空结果失败。

## 十一、结论

`task_344` 的失败不是“完全不会做”，而是“先把 male 集合缩小了，又在记录级/患者级与 FG 口径之间来回切换，最后没能形成可提交答案”。

模型经历了：

- 工具路径错误
- 统计粒度摇摆
- 阈值反复
- 性别映射漏覆盖

最终在 max steps 内没有完成提交。标准答案文件给出的正确值为 `4`。
