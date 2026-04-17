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

### 第一步：起手方向是对的，但很快出现一次空转

模型一开始就意识到这题需要同时查看：

- `csv/event.csv`
- `doc/budget.md`

这一步的总体方向没有问题，因为题目本来就是跨文档映射题。

但 trace 里很早就出现了一次“没有内容、也没有工具调用”的空响应，随后系统插入修复提示：

- `Your previous response stopped with no content and no tool call.`

这说明模型当时虽然已经读到一些上下文，但没有把当前思路收敛成下一步动作，推理节奏第一次被打断。

### 第二步：它先想用 SQL 直接查 `event.csv`，但工具类型判断错了

模型随后尝试用 `execute_context_sql` 直接查询：

```sql
SELECT event_id, event_name
FROM csv.event
WHERE event_name IN ('Yearly Kickoff', 'October Meeting')
```

但 trace 返回的是：

- `file is not a database`

这里的错误不是题目理解错误，而是工具使用错误：`event.csv` 是普通 CSV 文件，不是 SQLite 数据库，不能直接用 SQL 引擎查。

这一步的影响是，模型没能通过最短路径拿到两个目标会议的 `event_id`，只能临时切到 Python 方案。

### 第三步：切到 Python 后，它其实正确拿到了两个目标会议的 event_id

切换实现方式后，模型从 `event.csv` 中成功读出了事件名到事件 ID 的映射，其中关键两条是：

- `Yearly Kickoff -> recykdvf4LgsyA3wZ`
- `October Meeting -> recggMW2eyCYceNcy`

到这里为止，它已经拿到了后续联结所需的正确主键，属于一次有效进展。

### 第四步：它先走了一条错误的“同一行直接联查”思路，因此得到空结果

拿到两个 `event_id` 后，模型最早的想法是：

- 找到 `Advertisement` 预算条目
- 同时要求这一行里还出现目标 `event_id`

于是它跑出了：

- `Advertisement entries found: []`

这一步错在对 `budget.md` 结构的假设不对。`budget.md` 不是“category、amount、event_id 都挤在同一行”的扁平表，而是叙述文本：

- 前面一段主要写预算条目属于什么 category
- 后面另一段再写这个预算条目关联哪个 `event_id`

所以“必须在同一行同时看到 Advertisement 和目标 event_id”这个条件太强，天然会漏掉真正答案。

### 第五步：它成功枚举出了所有 Advertisement 预算 ID，但仍没有把金额一起绑定出来

意识到同一行联查不成立后，模型改成先枚举所有 Advertisement 预算条目，得到了：

- `rec0QmEc3cSQFQ6V2`
- `rec59vErJo51glQRb`
- `recFZ47e0eVqcQD9O`
- `recKjd7dcURsmP0KY`
- `recMc8TbR76rmUSHG`
- `recN9yY7okNrFps0Y`
- `recQf86wBWXvebEBO`
- `recTxecmwIhCdIKvl`
- `recXZUYlYNiRmeoxX`
- `recXaqLgIlsJ4eppc`
- `reckyFmOuPIIya9hY`
- `recqkZoc6ucWrS8xg`
- `recsI0IzpUuxl2bPh`
- `rectoPMq4aQQikNEe`
- `recvKTAWAFKkVNnXQ`

这一步本身是有价值的，因为它已经把候选集合缩到 15 个预算 ID。

但它紧接着想在这些 ID 周边直接搜金额，结果所有候选条目都得到：

- `Amounts found: []`

这说明它虽然拿到了“候选预算是谁”，但还没有理解文档里金额信息和预算 ID 并不是这种简单近邻关系。

### 第六步：它验证了会议名不在 `budget.md` 中，这个判断是对的

模型随后直接搜索：

- `Yearly Kickoff`
- `October Meeting`

trace 返回：

- `'Yearly Kickoff' not found in budget.md`
- `'October Meeting' not found in budget.md`

这一点是正确判断。也就是说，到这里它其实已经知道：

- 不能用会议名直接查 `budget.md`
- 必须走 `event_name -> event_id -> budget_id` 这条中间映射路径

### 第七步：它找到了两个目标 event_id 在 `budget.md` 中的位置，这是最接近正确答案的一步

接下来模型开始直接在 `budget.md` 搜目标 `event_id`：

- `recykdvf4LgsyA3wZ` 对应 `Yearly Kickoff`
- `recggMW2eyCYceNcy` 对应 `October Meeting`

这是这条 trace 里最关键的一次推进，因为它终于进入了正确的联结层：

- `Yearly Kickoff` 对应的目标 budget 段落附近，确实出现了 `recvKTAWAFKkVNnXQ`
- `October Meeting` 对应的目标 budget 段落附近，确实出现了 `recTxecmwIhCdIKvl`

如果此时它按“同一独立段落内的 budget_id <-> event_id 关系”继续精确抽取，其实已经非常接近解出答案了。

### 第八步：真正的核心错误出现在这里，它把“大窗口内同时出现”误当成“存在关联”

为了找出和两个目标 `event_id` 关联的 Advertisement 预算，模型取了每个目标 `event_id` 前后大约 `1500` 字符的上下文窗口，然后检查：

- 哪些 Advertisement budget id 出现在这个窗口里

于是它得到了很多“Found Advertisement budget ... in context”的结果。比如：

- `Yearly Kickoff` 的窗口里，出现了 `reckyFmOuPIIya9hY`、`recqkZoc6ucWrS8xg`、`recsI0IzpUuxl2bPh`、`rectoPMq4aQQikNEe`、`recvKTAWAFKkVNnXQ`
- `October Meeting` 的窗口里，出现了 `recMc8TbR76rmUSHG`、`recN9yY7okNrFps0Y`、`recQf86wBWXvebEBO`、`recTxecmwIhCdIKvl`、`recXZUYlYNiRmeoxX` 等多个预算

这里就是本题最关键的逻辑偏差：

- 模型把“在同一个大窗口里出现”当成了“这个预算与这个 event_id 有关”
- 但 `budget.md` 这一段本来就是连续列很多预算状态，窗口拉得太大时，多个互不相关的预算会自然同时落在同一片文本里

也就是说，这一步不是简单的“候选太多”，而是关系抽取规则本身错了：

- 正确规则应是看单个预算段落中明确出现的 `event record/event link`
- 它实际用的是“距离比较近就算有关”

这会导致关联关系被系统性放大。

### 第九步：它后面虽然把两个正确 budget_id 缩了出来，但金额提取依然失败

在不断试探后，模型最终把目标缩到两个其实是正确的预算 ID：

- `recvKTAWAFKkVNnXQ` 对应 `Yearly Kickoff`
- `recTxecmwIhCdIKvl` 对应 `October Meeting`

然后它去看这两个 budget 的局部上下文，并抽取数字，结果是：

- `Yearly Kickoff - Budget recvKTAWAFKkVNnXQ` -> `Numbers found: ['2']`
- `October Meeting - Budget recTxecmwIhCdIKvl` -> `Numbers found: []`

这里暴露出第二个核心问题：即使 budget_id 已经基本找对，它还是没有把“金额字段”从正确位置提出来。

其中 `['2']` 明显是误提取到了附近标题里的：

- `#### Section 2: Hospitality and Event-Related Allocations`

也就是说，它抓到的是章节编号，不是预算金额。

### 第十步：最后它退回到“全局找数字”，噪声变得更大，最终没有提交答案

在局部抽取金额失败后，模型继续扩大上下文并做更泛化的数字匹配，试图从整份 `budget.md` 中再搜：

- 金额
- 数字
- 预算相关词附近的数值

但这一步只会引入更多无关数字，无法稳定绑定到：

- `recvKTAWAFKkVNnXQ -> 150`
- `recTxecmwIhCdIKvl -> 55`

而且 trace 中后段又出现了一次空响应修复提示，说明它已经再次进入“知道还差最后一步，但没有形成稳定提交动作”的状态。

最终失败原因不是“算错了一个比值”，而是：

1. 正确主键拿到了
2. 正确 budget_id 一度也接近锁定
3. 但金额绑定始终没完成
4. 最后也没有调用 `answer`

因此 run 以：

- `Agent did not submit an answer within max_steps.`

结束。

## 七、正确答案为什么应该是标准答案中的 `2.727272727272727`

可复核得到两条关键映射：

- `recTxecmwIhCdIKvl` -> `event_id=recggMW2eyCYceNcy` -> `October Meeting`，金额 `55`
- `recvKTAWAFKkVNnXQ` -> `event_id=recykdvf4LgsyA3wZ` -> `Yearly Kickoff`，金额 `150`

因此：

$$
\frac{150}{55} = 2.727272727272727
$$

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

### 1. 关系抽取规则错了，不是简单“搜不到”

本题最重要的错误不是不会搜，而是把“大窗口内共同出现”误判成“二者有关联”。

这会把本应一对一的 `budget_id -> event_id` 映射，扩大成一堆模糊候选。

### 2. 正确 budget_id 曾经接近锁定，但金额字段始终没绑定成功

模型最后已经非常接近：

- `recvKTAWAFKkVNnXQ -> Yearly Kickoff`
- `recTxecmwIhCdIKvl -> October Meeting`

但它没能继续拿到：

- `150`
- `55`

所以失败点不只是在“最后没提交”，也在于金额抽取链条没有闭合。

### 3. 解析策略在“精确定位”与“全局试探”之间来回切换

它一会儿按局部段落看，一会儿又退回去做全局数字搜索，导致噪声越来越大，收敛越来越慢。

### 4. 缺少“关键键值一旦确认就进入定点抽取”的终态策略

在关键映射已形成时未及时提交，是本题失败的直接原因。

## 十、改进建议

### 1. 优先建立“精确映射链”，不要用大窗口近邻代替关系

先明确：

- `event_name -> event_id`
- `budget_id -> linked event_id`
- `budget_id -> amount`

只有三条关系都落到同一条记录上，再做计算。

### 2. 对叙述文档采用分阶段解析

先提实体（budget_id、event_id、amount），再提单段落内关系，最后做计算，避免在大范围上下文里混合匹配。

### 3. 在锁定候选 budget_id 后，必须切换成“定点金额抽取”

一旦已经缩到 `recvKTAWAFKkVNnXQ` 和 `recTxecmwIhCdIKvl`，就不应该再回到全局搜数字，而应该回到这些预算的定义段落中精确找 `amount`。

### 4. 增加提交触发条件

当两个目标会议金额都已确定时，强制进入提交流程；如果金额还没确定，也应显式围绕这两个 budget_id 做最后一次定点复核，而不是继续泛化搜索。

## 十一、结论

`task_352` 的失败不是单纯“没来得及提交”，而是一个更具体的链式失败：

1. 正确认识到需要跨 `event.csv` 与 `budget.md` 联结
2. 成功拿到两个目标会议的 `event_id`
3. 一度逼近正确的两个 `budget_id`
4. 但用“大窗口近邻”代替精确关系抽取，导致关联判断变模糊
5. 金额字段始终没有稳定提取出来
6. 最终也没有提交答案

按标准映射与金额计算，本题正确值为 `2.727272727272727`。
