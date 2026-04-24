# 20260424T011929Z 评测报告

## 执行摘要

- 评分规则来源：[https://dataagent.top/rules](https://dataagent.top/rules)
- 评分范围来源：当前 run 目录下的 `summary.json.tasks`。
- 本地结果采用“默认主分 + 多 λ 分析”体系，默认主分固定为 `λ=0.1`。
- 当前评分网格为 `0, 0.05, 0.1, 0.2, 0.3, 0.5`。
- 默认主分用于稳定比较默认冗余惩罚下的表现，多 λ 结果用于观察敏感度；不代表官方未公开 λ 下的唯一得分。
- `Recall`：单题覆盖率，计算方式为 `覆盖Gold列数 / Gold列总数`。
- `Mean Recall`：所有任务 `Recall` 的平均值，表示平均每题覆盖了多少 gold 列。
- `Mean Redundancy` / `Mean Redundancy Rate`：所有任务冗余率的平均值，表示平均每题预测列中有多少比例是多余列。

## 主分与基础指标

| 指标 | 值 |
| --- | --- |
| 任务总数 | 50 |
| 生成 prediction.csv 的任务数 | 40 |
| Primary Score (λ=0.1) | 0.5000 |
| Mean Recall | 0.5000 |
| Mean Redundancy Rate | 0.3000 |

## 多 λ 代理分数

| λ | 代理分数 |
| --- | --- |
| 0 | 0.5000 |
| 0.05 | 0.5000 |
| 0.1 | 0.5000 |
| 0.2 | 0.5000 |
| 0.3 | 0.5000 |
| 0.5 | 0.5000 |

## 按难度拆分表现

| 难度 | 任务数 | 有预测 | 完全正确题数 | Primary(λ=0.1) | Mean Recall | Mean Redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 15 | 12 | 7 | 0.4667 | 0.4667 | 0.3333 |
| medium | 23 | 18 | 12 | 0.5217 | 0.5217 | 0.2609 |
| hard | 11 | 9 | 6 | 0.5455 | 0.5455 | 0.2727 |
| extreme | 1 | 1 | 0 | 0.0000 | 0.0000 | 1.0000 |

## 失败原因与耗时分析

### 失败原因

| 失败原因 | 任务数 |
| --- | --- |
| Agent did not submit an answer within max_steps. | 10 |

### 运行时摘要

| 指标 | 值 |
| --- | --- |
| 可用耗时任务数 | 50 |
| 平均耗时（秒） | 70.385 |
| 中位耗时（秒） | 44.542 |
| P95 耗时（秒） | 172.556 |
| 最长耗时（秒） | 425.245 |
| 可用模型轮数任务数 | 50 |
| 平均模型轮数 | 15.48 |
| 最大模型轮数 | 32 |

## 最值得复盘的任务（Primary < 0.5，共 25 题）

| 任务 | 难度 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 失败/备注 | 模型轮数 | 耗时(秒) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_11 | easy | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 165.372 |
| task_22 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 4 | 13.838 |
| task_25 | easy | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 157.480 |
| task_38 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 7 prediction column(s) were produced and 7 are redundant. | 9 | 200.755 |
| task_75 | easy | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 113.272 |
| task_80 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 8 | 25.385 |
| task_86 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 13 | 40.859 |
| task_89 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 18 | 63.889 |
| task_163 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 9 | 27.946 |
| task_169 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 9 | 34.445 |
| task_173 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 10 | 28.873 |
| task_180 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 10 | 57.146 |
| task_196 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 114.366 |
| task_199 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 15 | 67.237 |
| task_200 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 133.869 |
| task_218 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 119.997 |
| task_249 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 118.392 |
| task_250 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 108.268 |
| task_259 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 8 | 28.476 |
| task_344 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 22 | 126.185 |
| task_355 | hard | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 109.240 |
| task_379 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 11 | 57.372 |
| task_396 | hard | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 425.245 |
| task_418 | extreme | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 6 | 178.434 |
| task_420 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 22 | 93.553 |

## 全量任务附录

| 任务 | 难度 | Gold列数 | 预测列数 | 覆盖Gold列数 | 冗余列数 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 模型轮数 | 耗时(秒) | 失败/备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_11 | easy | 3 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 165.372 | Agent did not submit an answer within max_steps. |
| task_19 | easy | 2 | 1 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 29.291 | - |
| task_22 | easy | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 4 | 13.838 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_24 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 22.597 | - |
| task_25 | easy | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 157.480 | Agent did not submit an answer within max_steps. |
| task_26 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 6 | 19.231 | - |
| task_27 | easy | 3 | 2 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 5 | 15.638 | - |
| task_38 | easy | 1 | 7 | 0 | 7 | 0.0000 | 0.0000 | 1.0000 | no | 9 | 200.755 | No gold columns matched; 7 prediction column(s) were produced and 7 are redundant. |
| task_64 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 20.349 | - |
| task_67 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 5 | 14.554 | - |
| task_74 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 19.624 | - |
| task_75 | easy | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 113.272 | Agent did not submit an answer within max_steps. |
| task_80 | easy | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 8 | 25.385 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_86 | easy | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 13 | 40.859 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_89 | easy | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 18 | 63.889 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_145 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 10 | 28.134 | - |
| task_163 | medium | 2 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 9 | 27.946 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_169 | medium | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 9 | 34.445 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_173 | medium | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 10 | 28.873 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_180 | medium | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 10 | 57.146 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_194 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 17.818 | - |
| task_196 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 114.366 | Agent did not submit an answer within max_steps. |
| task_199 | medium | 2 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 15 | 67.237 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_200 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 133.869 | Agent did not submit an answer within max_steps. |
| task_214 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 17 | 55.792 | - |
| task_218 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 119.997 | Agent did not submit an answer within max_steps. |
| task_243 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 14 | 34.307 | - |
| task_249 | medium | 2 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 118.392 | Agent did not submit an answer within max_steps. |
| task_250 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 108.268 | Agent did not submit an answer within max_steps. |
| task_257 | medium | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 32.310 | - |
| task_259 | medium | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 8 | 28.476 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_261 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 19 | 67.290 | - |
| task_269 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 15 | 55.620 | - |
| task_283 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 18.990 | - |
| task_287 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 32 | 121.544 | - |
| task_292 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 5 | 12.452 | - |
| task_303 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 13 | 35.325 | - |
| task_305 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 23 | 72.218 | - |
| task_330 | hard | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 41.281 | - |
| task_344 | hard | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 22 | 126.185 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_349 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 16 | 51.254 | - |
| task_350 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 25.122 | - |
| task_352 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 12 | 47.803 | - |
| task_355 | hard | 3 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 109.240 | Agent did not submit an answer within max_steps. |
| task_379 | hard | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 11 | 57.372 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_396 | hard | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 425.245 | Agent did not submit an answer within max_steps. |
| task_408 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 32.421 | - |
| task_415 | hard | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 18.400 | - |
| task_418 | extreme | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 6 | 178.434 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_420 | hard | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 22 | 93.553 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
