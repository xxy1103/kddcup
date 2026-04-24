# 20260424T014353Z 评测报告

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
| 生成 prediction.csv 的任务数 | 47 |
| Primary Score (λ=0.1) | 0.6770 |
| Mean Recall | 0.6900 |
| Mean Redundancy Rate | 0.3700 |

## 多 λ 代理分数

| λ | 代理分数 |
| --- | --- |
| 0 | 0.6900 |
| 0.05 | 0.6835 |
| 0.1 | 0.6770 |
| 0.2 | 0.6640 |
| 0.3 | 0.6510 |
| 0.5 | 0.6250 |

## 按难度拆分表现

| 难度 | 任务数 | 有预测 | 完全正确题数 | Primary(λ=0.1) | Mean Recall | Mean Redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 15 | 15 | 10 | 0.6589 | 0.6667 | 0.4111 |
| medium | 23 | 22 | 17 | 0.7413 | 0.7609 | 0.3696 |
| hard | 11 | 10 | 7 | 0.6288 | 0.6364 | 0.3485 |
| extreme | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |

## 失败原因与耗时分析

### 失败原因

| 失败原因 | 任务数 |
| --- | --- |
| Agent did not submit an answer within max_steps. | 2 |
| Task timed out after 600 seconds. | 1 |

### 运行时摘要

| 指标 | 值 |
| --- | --- |
| 可用耗时任务数 | 50 |
| 平均耗时（秒） | 51.357 |
| 中位耗时（秒） | 24.786 |
| P95 耗时（秒） | 145.280 |
| 最长耗时（秒） | 600.031 |
| 可用模型轮数任务数 | 50 |
| 平均模型轮数 | 10.68 |
| 最大模型轮数 | 32 |

## 最值得复盘的任务（Primary < 0.5，共 16 题）

| 任务 | 难度 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 失败/备注 | 模型轮数 | 耗时(秒) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_25 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 6 | 19.575 |
| task_38 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 10 prediction column(s) were produced and 10 are redundant. | 14 | 87.188 |
| task_80 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. | 13 | 82.581 |
| task_86 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. | 9 | 28.456 |
| task_89 | easy | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. | 13 | 48.821 |
| task_163 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 8 | 22.829 |
| task_169 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. | 7 | 18.384 |
| task_173 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 92.031 |
| task_180 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 10 | 49.252 |
| task_200 | medium | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 14 | 39.617 |
| task_249 | medium | 0.4500 | 0.5000 | 0.5000 | no | Covered 1/2 gold column(s), with 1 extra prediction column(s). | 9 | 36.782 |
| task_344 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 27 | 156.230 |
| task_352 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 22 | 131.897 |
| task_379 | hard | 0.0000 | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 7 | 36.407 |
| task_396 | hard | 0.0000 | 0.0000 | 0.0000 | no | Task timed out after 600 seconds. | 0 | 600.031 |
| task_418 | extreme | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 177.247 |

## 全量任务附录

| 任务 | 难度 | Gold列数 | 预测列数 | 覆盖Gold列数 | 冗余列数 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 模型轮数 | 耗时(秒) | 失败/备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_11 | easy | 3 | 3 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 22.089 | - |
| task_19 | easy | 2 | 1 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 26.921 | - |
| task_22 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 18.878 | - |
| task_24 | easy | 1 | 2 | 1 | 1 | 0.9500 | 1.0000 | 0.5000 | yes | 6 | 14.586 | All gold columns covered, with 1 extra prediction column(s). |
| task_25 | easy | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 6 | 19.575 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_26 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 17.909 | - |
| task_27 | easy | 3 | 2 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 6 | 16.475 | - |
| task_38 | easy | 1 | 10 | 0 | 10 | 0.0000 | 0.0000 | 1.0000 | no | 14 | 87.188 | No gold columns matched; 10 prediction column(s) were produced and 10 are redundant. |
| task_64 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 18.322 | - |
| task_67 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 14.814 | - |
| task_74 | easy | 1 | 3 | 1 | 2 | 0.9333 | 1.0000 | 0.6667 | yes | 7 | 15.620 | All gold columns covered, with 2 extra prediction column(s). |
| task_75 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 25.266 | - |
| task_80 | easy | 1 | 3 | 0 | 3 | 0.0000 | 0.0000 | 1.0000 | no | 13 | 82.581 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
| task_86 | easy | 1 | 3 | 0 | 3 | 0.0000 | 0.0000 | 1.0000 | no | 9 | 28.456 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
| task_89 | easy | 1 | 3 | 0 | 3 | 0.0000 | 0.0000 | 1.0000 | no | 13 | 48.821 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
| task_145 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 6 | 16.685 | - |
| task_163 | medium | 2 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 8 | 22.829 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_169 | medium | 1 | 3 | 0 | 3 | 0.0000 | 0.0000 | 1.0000 | no | 7 | 18.384 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
| task_173 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 92.031 | Agent did not submit an answer within max_steps. |
| task_180 | medium | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 10 | 49.252 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_194 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 7 | 16.326 | - |
| task_196 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 9 | 20.391 | - |
| task_199 | medium | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 17 | 72.825 | - |
| task_200 | medium | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 14 | 39.617 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_214 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 22.911 | - |
| task_218 | medium | 1 | 2 | 1 | 1 | 0.9500 | 1.0000 | 0.5000 | yes | 9 | 19.905 | All gold columns covered, with 1 extra prediction column(s). |
| task_243 | medium | 1 | 4 | 1 | 3 | 0.9250 | 1.0000 | 0.7500 | yes | 7 | 14.228 | All gold columns covered, with 3 extra prediction column(s). |
| task_249 | medium | 2 | 2 | 1 | 1 | 0.4500 | 0.5000 | 0.5000 | no | 9 | 36.782 | Covered 1/2 gold column(s), with 1 extra prediction column(s). |
| task_250 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 18 | 85.461 | - |
| task_257 | medium | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 14 | 55.092 | - |
| task_259 | medium | 1 | 4 | 1 | 3 | 0.9250 | 1.0000 | 0.7500 | yes | 9 | 33.656 | All gold columns covered, with 3 extra prediction column(s). |
| task_261 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 10 | 28.778 | - |
| task_269 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 10 | 20.891 | - |
| task_283 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 11 | 19.133 | - |
| task_287 | medium | 1 | 3 | 1 | 2 | 0.9333 | 1.0000 | 0.6667 | yes | 12 | 22.589 | All gold columns covered, with 2 extra prediction column(s). |
| task_292 | medium | 1 | 3 | 1 | 2 | 0.9333 | 1.0000 | 0.6667 | yes | 7 | 14.034 | All gold columns covered, with 2 extra prediction column(s). |
| task_303 | medium | 1 | 3 | 1 | 2 | 0.9333 | 1.0000 | 0.6667 | yes | 9 | 27.918 | All gold columns covered, with 2 extra prediction column(s). |
| task_305 | medium | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 10 | 20.931 | - |
| task_330 | hard | 2 | 3 | 2 | 1 | 0.9667 | 1.0000 | 0.3333 | yes | 9 | 39.562 | All gold columns covered, with 1 extra prediction column(s). |
| task_344 | hard | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 27 | 156.230 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_349 | hard | 1 | 2 | 1 | 1 | 0.9500 | 1.0000 | 0.5000 | yes | 8 | 23.400 | All gold columns covered, with 1 extra prediction column(s). |
| task_350 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 26.003 | - |
| task_352 | hard | 1 | 1 | 0 | 1 | 0.0000 | 0.0000 | 1.0000 | no | 22 | 131.897 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_355 | hard | 3 | 2 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 8 | 23.702 | - |
| task_379 | hard | 1 | 2 | 0 | 2 | 0.0000 | 0.0000 | 1.0000 | no | 7 | 36.407 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_396 | hard | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 0 | 600.031 | Task timed out after 600 seconds. |
| task_408 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 12 | 52.587 | - |
| task_415 | hard | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 10 | 24.306 | - |
| task_418 | extreme | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 177.247 | Agent did not submit an answer within max_steps. |
| task_420 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 11 | 48.334 | - |
