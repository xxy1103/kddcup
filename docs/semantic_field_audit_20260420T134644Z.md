# 20260420T134644Z 输入数据字段语义审计

## 范围与判定标准

- 范围：只检查 `data/public/input/task_86`、`task_89`、`task_180`、`task_200` 的 `input` 数据与各自 `context/knowledge.md`。
- 目标：验证 `knowledge.md` 是否对输入数据里出现的每个字段提供了足够详细、可执行的语义说明。
- 判定：
  - **明确说明**：文档直接点名字段，并说明其业务含义或推荐用法。
  - **部分说明**：文档提到了字段名，但不足以支撑精确判断，或说明与输入文件上下文不完全一致。
  - **未说明**：字段存在于输入数据中，但知识文档没有给出可依赖的定义。

## 总结结论

- `task_180` 最弱：题目真正依赖的 `transactions_1k.db` 关键字段几乎没有被文档解释，无法仅凭知识文档稳定判断 `per unit`、`product id`、`price`、`amount` 的语义。
- `task_86` 偏弱：共享的 Formula_1 知识文档没有解释题目最关键的 `number` 字段，也没有解释 `driverStandings` 这张表的语义层级。
- `task_89` 中等：文档对 `rank / positionOrder / time / milliseconds` 的歧义有帮助，但仍不是全字段覆盖，而且 `position`、`time` 的说明还不够精确。
- `task_200` 最完整：所有实际输入字段基本都有 schema 级解释，但对元素取值编码没有给出值域映射，无法单靠文档把 “phosphorus / bromine” 稳定落到 `p / br`。

## Task 86

- 题目：`Which race was Alex Yoong in when he was in track number less than 20?`
- 输入文件：
  - `json/drivers.json`
  - `csv/driverStandings.csv`
  - `csv/races.csv`
- 结论：**不能**。知识文档不能为所有字段提供详细说明，且缺失正好落在题目关键字段上。

### 字段覆盖

| 文件 | 明确说明 | 部分说明 | 未说明 |
| --- | --- | --- | --- |
| `json/drivers.json` | `driverId`, `driverRef`, `forename`, `surname`, `dob`, `nationality` | 无 | `number`, `code`, `url` |
| `csv/driverStandings.csv` | `raceId`, `driverId`, `points` | `position` | `driverStandingsId`, `positionText`, `wins` |
| `csv/races.csv` | `raceId`, `year`, `round`, `circuitId`, `name`, `date` | `time` | `url` |

### 关键问题

- 文档没有定义 `drivers.number`，但题目里的 `track number less than 20` 最容易被映射到这个字段。
- 文档没有定义 `driverStandings` 这张表是什么层级的数据，只提了 `driverId`、`raceId`、`points` 这些零散字段，无法让模型稳定判断它是“每站后的车手积分/排名快照”，还是“实际参赛记录”。
- 文档对 `position` 的说明是放在 `position vs. positionOrder` 里，但那里针对的是比赛结果语义，不是 `driverStandings.position`。

### 输入数据证据

- `drivers.json` 里 Alex Yoong 的记录是：`driverId = 62`，但 `number = null`。这说明题目里的 `track number` 仅靠知识文档并不能稳定落字段，数据本身也会逼着模型重新解释题意。
- `driverStandings.csv` 里 Alex Yoong 有 18 行记录，字段只有 `driverStandingsId, raceId, driverId, points, position, positionText, wins`。知识文档没有解释这些 standings 字段与“参赛事实”之间的关系。

## Task 89

- 题目：`What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?`
- 输入文件：
  - `csv/results.csv`
  - `json/races.json`
- 结论：**部分可以**。关键歧义字段有一定说明，但仍达不到“每个字段都有详细说明”的标准。

### 字段覆盖

| 文件 | 明确说明 | 部分说明 | 未说明 |
| --- | --- | --- | --- |
| `csv/results.csv` | `raceId`, `driverId`, `constructorId`, `points`, `positionOrder`, `milliseconds`, `rank`, `fastestLapTime`, `fastestLapSpeed` | `position`, `time`, `laps`, `statusId` | `resultId`, `number`, `grid`, `positionText`, `fastestLap` |
| `json/races.json` | `raceId`, `year`, `round`, `circuitId`, `name`, `date` | `time` | `url` |

### 关键问题

- 这是四个任务里文档对关键语义帮助最大的一题，因为它明确写了：
  - `positionOrder` 用于 final race rankings
  - `rank` 是基于 `fastestLapTime` 的排序，不是最终完赛名次
  - `time` 是展示值，`milliseconds` 更适合精确计算
- 但文档仍不够细：
  - `position` 被写成 “grid or qualifying positions”，这和 `results.csv` 里同时存在的 `grid` 字段不一致，容易误导。
  - `time` 没有解释“冠军是完整时长，后续车手常常是相对领先者的差值字符串”。
  - `statusId` 出现在 KPI 公式里，但没有给出状态码字典。

### 输入数据证据

- `races.json` 中 2008 Chinese Grand Prix 的记录是 `raceId = 34`。
- `results.csv` 中该比赛前三名样例：
  - 第 2 名：`positionOrder = 2`，`time = +14.925`，`rank = 4`
  - `rank = 2` 的车手：`positionOrder = 3`，`time = +16.445`
- 这说明题面里的 `ranked second` 如果不被知识文档强行锚到 `rank`，模型就很容易误取 `position = 2` 或 `positionOrder = 2`。

## Task 180

- 题目：`For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.`
- 输入文件：
  - `csv/yearmonth.csv`
  - `db/transactions_1k.db`
- 结论：**不能**。知识文档只解释了月度汇总表，几乎没有解释题目真正依赖的交易事实表字段。

### 字段覆盖

| 文件 | 明确说明 | 部分说明 | 未说明 |
| --- | --- | --- | --- |
| `csv/yearmonth.csv` | `CustomerID`, `Date`, `Consumption` | 无 | 无 |
| `db/transactions_1k.db` | `CustomerID`, `Date` | 无 | `TransactionID`, `Time`, `CardID`, `GasStationID`, `ProductID`, `Amount`, `Price` |

### 关键问题

- 题目核心条件全部落在未说明字段上：
  - `product id No.5` 需要 `ProductID`
  - `paid more than 29.00 per unit` 需要解释 `Price` 与 `Amount` 的关系
  - 人群筛出后再去 `yearmonth.csv` 找 2012-08 的 `Consumption`
- 文档没有说明：
  - `Price` 是总价还是单价
  - `Amount` 是数量、升数还是别的单位
  - `Price / Amount` 是否才是 unit price
  - `transactions_1k` 与 `yearmonth` 是否通过 `CustomerID` 对接

### 输入数据证据

- `transactions_1k` 样例行：`ProductID = 5`, `Amount = 5`, `Price = 120.74`，对应 `Price / Amount = 24.148`。
- 这很像“`Price` 是总价、`Amount` 是数量”，但这完全是从数据模式里反推出来的，不是知识文档给出的定义。
- `yearmonth.csv` 只提供 `CustomerID, Date, Consumption` 三列，因此“consumption status in August of 2012” 也只能再靠推断为 `Date = 201208` 时的 `Consumption` 数值。

## Task 200

- 题目：`Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.`
- 输入文件：
  - `csv/atom.csv`
  - `db/bond.db`
  - `json/molecule.json`
- 结论：**大体可以，但仍不够完整**。字段级定义基本齐全，但值域映射没有讲透。

### 字段覆盖

| 文件 | 明确说明 | 部分说明 | 未说明 |
| --- | --- | --- | --- |
| `csv/atom.csv` | `atom_id`, `molecule_id`, `element` | `element` 的实际值域 | 无 |
| `db/bond.db` | `bond_id`, `molecule_id`, `bond_type` | `bond_type` 的空值语义 | 无 |
| `json/molecule.json` | `molecule_id`, `label` | 无 | 无 |

### 关键问题

- 文档把 `element` 解释成 chemical element，例如 hydrogen、carbon，但没有说明数据文件里实际保存的是小写元素符号。
- 题目用的是自然语言 `phosphorus`、`bromine`，而输入数据里真实取值是 `p`、`br`。这一步映射没有被文档显式提供。
- 文档额外介绍了 `Connected(atom_id, atom_id2, bond_id)`，但这个实体并不在当前输入文件里；这说明文档更像“领域知识总览”，不是严格贴着当前输入文件写的 field dictionary。

### 输入数据证据

- `atom.csv` 的 `element` 实际样例值包括：`cl`, `c`, `h`, `o`, `s`, `n`, `p`, `na`, `br`。
- `bond.db` 中 `bond_type` 的 distinct 值包括 `#`, `-`, `=`, 以及 `NULL`。
- 因此，虽然 `bond_type = '#'` 的三键语义被文档解释得很清楚，但 `phosphorus / bromine -> p / br` 的值映射仍需要模型自己补推。

## 最终判断

- 如果标准是“知识文档能否对 input 数据里每个字段都给出详细说明”，答案是：**不能**。
- 四个任务里只有 `task_200` 接近“字段级可用”，但它也停留在 schema 层，不够覆盖值域层。
- `task_89` 的关键歧义说明不错，但仍不是全字段字典，而且部分表述会误导。
- `task_86` 与 `task_180` 都存在“题目关键字段恰好没有被知识文档解释”的问题，其中 `task_180` 最严重。
