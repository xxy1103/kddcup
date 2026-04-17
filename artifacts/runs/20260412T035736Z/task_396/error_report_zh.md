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

$$
\text{Percentage} = \frac{\text{Marvel heroes in range}}{\text{All heroes in range}} \times 100
$$

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

这一题的 trace 不是“一步明显做错”，而是“不断尝试新解析口径，但每一轮都没有走到最后一步提交”。下面按时间线拆开。

### 第一步：先读取文档和映射表，但第一次就出现空转

模型先读了：

- `doc/superhero.md`
- `json/publisher.json`

并成功拿到关键信息：

- `publisher.json` 中 `id = 13 -> Marvel Comics`

但紧接着模型没有继续调用工具，也没有提交答案，而是直接输出了空响应。系统随后插入 repair 提示：

- `Your previous response stopped with no content and no tool call.`

这说明模型虽然拿到了第一批有效上下文，但当时还没有形成明确的后续计划。

### 第二步：转去确认“有没有数据库可直接查”，又发生一次空转

被 repair 拉回后，模型没有立刻开始抽取字段，而是先用 Python 搜索上下文里是否有 `.db / .sqlite / .sqlite3` 文件。

结果是：

- `Database files found: []`
- 上下文里只有 `knowledge.md`、`doc/superhero.md`、`json/publisher.json`

这一步本身没错，因为它确认了任务必须走文本解析路径，不能用 SQL 直接查。

但在确认完这一点之后，模型又出现了一次空响应，再次触发 repair。也就是说，在真正进入信息抽取前，它已经浪费了多个 step 在“确认环境”和“空转恢复”上。

### 第三步：开始用知识文档建立字段心智，但仍没有立刻进入稳定抽取

随后模型读了 `knowledge.md`，实际已经拿到了非常关键的字段定义：

- 身高字段应对应 `height_cm`
- 出版社字段应对应 `publisher_id`
- 百分比计算应为 `count(condition) / count(total) * 100`

从这里开始，正确路线其实已经很清晰了：

1. 从文档里抽 `id + height_cm`
2. 从文档里抽 `id + publisher_id`
3. 用 `publisher.json` 确认 `13 = Marvel Comics`
4. 统计 `150 <= height_cm <= 180` 区间里的 Marvel 占比

但模型后续并没有直接围绕这条三元组路径收敛，而是进入了多轮口径不一致的正则试探。

### 第四步：先做“候选行搜索”，但这一步只能看到噪声，不能直接得到答案

模型先用比较宽的正则去找：

- 文档中所有可能出现 `150-200` 数字的行
- 文档中所有包含 publisher 相关描述的行

这一步的结果是看到了 publisher section 的起始位置，也看到了很多形如：

- `publisher affiliation is logged with the code 13`
- `publisher affiliation is recorded as 4`

这样的文本片段。

问题在于，这一轮还是“看到局部句子”，没有把：

- 英雄 ID
- height
- publisher_id

稳定地绑定成同一条记录。

### 第五步：模型一度走错方向，直接搜名称 `Marvel Comics`

中段 trace 中有一个明显的偏航：模型直接在 `superhero.md` 文本里搜索：

- `Marvel Comics`
- `DC Comics`
- `Dark Horse Comics`
- `Image Comics`

输出结果是：

- `Marvel Comics: 0 occurrences`
- `DC Comics: 0 occurrences`
- `Dark Horse Comics: 0 occurrences`
- `Image Comics: 0 occurrences`
- `publisher_id values: []`

这一步非常关键，因为它暴露出模型当时的错误假设：

- 它一度希望在 `superhero.md` 里直接看到出版社名称
- 但这份文档真正保存的是 `publisher affiliation ... code 13` 这种 ID 形式

正确做法应当是：

- 在 `superhero.md` 抓 `publisher_id`
- 再去 `publisher.json` 做映射

而不是直接在长文里搜 `Marvel Comics` 字面串。

### 第六步：开始抓身高，但抽出来的是“高度列表”，不是“可统计记录”

后面模型开始真正抓身高，先得到：

- `Found 74 height entries`

样例中包含：

- `188.0`
- `165.0`
- `0.0`
- `61.0`
- `211.0`
- `168.0`
- `178.0`

这说明它已经能识别一批 height 值，但这里仍有两个问题：

1. 这些结果只是“带数字的行”，并不是结构化记录。
2. 其中混有明显的异常或占位值，例如 `0.0`，以及不在目标区间内的大量值。

也就是说，这一轮只能证明“身高能被抓到”，还不能直接得到分母。

### 第七步：尝试把姓名、ID、height、publisher 一次性从局部窗口里抽出来，但结果明显不稳定

模型接着写了一版更激进的脚本，想在同一轮里同时提取：

- name
- id
- height
- publisher_id

它的输出是：

- `Found 98 entries`

但样本内容显示这些“entries”大多只有：

- `{'name': 'Angel Dust', 'id': 26}`
- `{'name': 'Batman II', 'id': 74}`

缺少高度和 publisher 信息。

更重要的是，结果里已经能看到：

- 名称写法不统一
- 同一英雄可能重复出现
- 部分 entry 只有 name 和 id，没有其他字段

所以这轮不是“抽完了”，而是说明“一次性全抽”的方案失败了。

### 第八步：重新识别“文档分成两个 section”，但 section 起点本身就找得不稳

意识到不能一次性全抽后，模型转向更合理的思路：

- 先抽 biometric section
- 再抽 publisher section

但它一开始找 biometric 起点时，用的字符串定位并不稳定，先得到：

- `Biometric section start: 62619`
- `Found 67 height entries`

下一轮又换了定位字符串，得到：

- `Biometric section starts at: 92393`
- `Publisher section starts at: 148811`

这两个起点差异很大，说明模型并不确定自己到底从哪里开始切 section。section 边界不稳，就会直接影响：

- 哪些英雄被纳入 biometrics
- 哪些条目被漏掉
- 后续 join 是否完整

### 第九步：终于抽出了部分 biometrics，但覆盖仍不完整

在第二版 section 切分后，模型得到：

- `Parsed 52 entries from biometric section`

样例中出现了：

- `{'name': 'the operative Absorbing Man', 'id': 7, 'height': 188.0}`
- `{'name': 'the asset Angel Dust', 'id': 26, 'height': 165.0}`
- `{'name': 'the operative Angela', 'id': 28, 'height': 163.0}`
- `{'name': 'Atom IV', 'id': 58, 'height': 168.0}`
- `{'name': 'Batman II', 'id': 74, 'height': 178.0}`

这一轮比之前明显进步，因为终于开始形成：

- `id -> height`

的结构化片段。

但它仍有明显缺陷：

- 只解析出 52 条，而前面高度候选有 67 或 74 条，覆盖率在波动
- 有些英雄缺 height 字段，例如样例中的 `Beetle`
- 名称前缀仍不统一，如 `the operative ...`、`the asset ...`

因此这一轮仍不足以可靠计算分母。

### 第十步：publisher section 也只抽出了一个“部分映射表”

最后一轮，模型对 publisher section 做了类似抽取，得到：

- `Parsed 28 entries from publisher section`

样例包括：

- `ID 7 -> publisher_id 13`
- `ID 26 -> publisher_id 13`
- `ID 28 -> publisher_id 13`
- `ID 58 -> publisher_id 4`
- `ID 74 -> publisher_id 4`

这说明模型已经开始形成：

- `id -> publisher_id`

的局部映射。

但这里的问题更明显：

- publisher section 只抽到 28 条
- biometric section 抽到 52 条
- 两者覆盖规模严重不一致

此时如果直接 join，只会得到一个偏小而且不完整的交集。模型显然也意识到了这一点，因此没有敢直接算百分比。

### 第十一步：已经接近可计算状态，但始终没迈出“计算并提交”这一步

到 trace 末段，模型其实已经拥有三块关键中间结果：

- `publisher.json` 告诉它 `13 = Marvel Comics`
- biometric section 给出一批 `id -> height`
- publisher section 给出一批 `id -> publisher_id`

理论上这时即便覆盖不完美，也至少应该：

1. 先对交集计算一个候选百分比
2. 检查结果是否稳定
3. 若无更好方案，提交当前最优答案

但模型没有这么做，而是停在“刚把两个 section 分别解析出来”的阶段，没有执行最终的：

- 过滤 `150 <= height <= 180`
- 统计 publisher_id = 13 的数量
- 输出百分比
- 调用 `answer`

### 第十二步：最终不是“算错”，而是“没交卷”

本题最后的 failure reason 是：

- `Agent did not submit an answer within max_steps.`

所以更准确地说，这题不是“模型算出了错误百分比”，而是：

- 它一直在扩展解析覆盖面
- 但没有设置一个“已经够算了，先交”的收敛阈值
- 结果在 max_steps 之前停在中间状态，最终没有生成 `prediction.csv`

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
