# 20260412T035736Z 评测报告

## 执行摘要

- 评分规则来源：[https://dataagent.top/rules](https://dataagent.top/rules)
- 本地结果采用“双指标 + 多 λ 代理”体系，`λ` 网格为 `0, 0.05, 0.1, 0.2, 0.3, 0.5`。
- 本地分数用于全面评估 Recall 与冗余惩罚敏感度，不代表官方未公开 λ 下的唯一得分。

## 双指标总览

| 指标 | 值 |
| --- | --- |
| 任务总数 | 50 |
| 生成 prediction.csv 的任务数 | 46 |
| Full Cover 任务数 | 35 |
| Full Cover Rate | 0.7000 |
| Mean Recall | 0.7100 |
| Mean Redundancy Rate | 0.3120 |
| 兼容 total_score | 35 |
| 兼容 accuracy | 0.7000 |

## 多 λ 代理分数

| λ | 代理分数 |
| --- | --- |
| 0 | 0.7100 |
| 0.05 | 0.7044 |
| 0.1 | 0.6988 |
| 0.2 | 0.6876 |
| 0.3 | 0.6764 |
| 0.5 | 0.6540 |

## 按难度拆分表现

| 难度 | 任务数 | 有预测 | Full Cover | Full Cover Rate | Mean Recall | Mean Redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 15 | 15 | 11 | 0.7333 | 0.7333 | 0.3533 |
| extreme | 2 | 2 | 1 | 0.5000 | 0.5000 | 0.5000 |
| hard | 10 | 7 | 6 | 0.6000 | 0.6000 | 0.1333 |
| medium | 23 | 22 | 17 | 0.7391 | 0.7609 | 0.3463 |

## 失败原因与耗时分析

### 失败原因

| 失败原因 | 任务数 |
| --- | --- |
| Agent did not submit an answer within max_steps. | 4 |

### 运行时摘要

| 指标 | 值 |
| --- | --- |
| 可用耗时任务数 | 50 |
| 平均耗时（秒） | 39.064 |
| 中位耗时（秒） | 19.491 |
| P95 耗时（秒） | 135.999 |
| 最长耗时（秒） | 245.477 |
| 可用 step_count 任务数 | 50 |
| 平均 step_count | 23.96 |
| 最大 step_count | 64 |

## 最值得复盘的任务

| 任务 | 难度 | Recall | Redundancy | Full Cover | 失败/备注 | 耗时(秒) |
| --- | --- | --- | --- | --- | --- | --- |
| task_344 | hard | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 245.477 |
| task_396 | hard | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 147.472 |
| task_352 | hard | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 129.292 |
| task_173 | medium | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 98.432 |
| task_418 | extreme | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 141.487 |
| task_80 | easy | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 122.437 |
| task_180 | medium | 0.0000 | 1.0000 | no | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. | 65.014 |
| task_169 | medium | 0.0000 | 1.0000 | no | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. | 50.381 |

## 全量任务附录

| 任务 | 难度 | Gold列数 | 预测列数 | 覆盖Gold列数 | 冗余列数 | Recall | Redundancy | Full Cover | 耗时(秒) | 失败/备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_11 | easy | 3 | 3 | 3 | 0 | 1.0000 | 0.0000 | yes | 10.668 | - |
| task_145 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 13.815 | - |
| task_163 | medium | 2 | 2 | 0 | 2 | 0.0000 | 1.0000 | no | 14.964 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_169 | medium | 1 | 1 | 0 | 1 | 0.0000 | 1.0000 | no | 50.381 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_173 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | no | 98.432 | Agent did not submit an answer within max_steps. |
| task_180 | medium | 1 | 2 | 0 | 2 | 0.0000 | 1.0000 | no | 65.014 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_19 | easy | 2 | 2 | 2 | 0 | 1.0000 | 0.0000 | yes | 18.125 | - |
| task_194 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 10.800 | - |
| task_196 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 21.137 | - |
| task_199 | medium | 2 | 2 | 2 | 0 | 1.0000 | 0.0000 | yes | 62.168 | - |
| task_200 | medium | 1 | 1 | 0 | 1 | 0.0000 | 1.0000 | no | 39.770 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_214 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 14.925 | - |
| task_218 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 13.385 | - |
| task_22 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 16.259 | - |
| task_24 | easy | 1 | 2 | 1 | 1 | 1.0000 | 0.5000 | yes | 6.812 | All gold columns covered, with 1 extra prediction column(s). |
| task_243 | medium | 1 | 4 | 1 | 3 | 1.0000 | 0.7500 | yes | 9.031 | All gold columns covered, with 3 extra prediction column(s). |
| task_249 | medium | 2 | 2 | 1 | 1 | 0.5000 | 0.5000 | no | 28.854 | Covered 1/2 gold column(s), with 1 extra prediction column(s). |
| task_25 | easy | 1 | 2 | 0 | 2 | 0.0000 | 1.0000 | no | 17.155 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_250 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 37.683 | - |
| task_257 | medium | 2 | 3 | 2 | 1 | 1.0000 | 0.3333 | yes | 46.406 | All gold columns covered, with 1 extra prediction column(s). |
| task_259 | medium | 1 | 7 | 1 | 5 | 1.0000 | 0.7143 | yes | 12.324 | All gold columns covered, with 5 extra prediction column(s). |
| task_26 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 10.856 | - |
| task_261 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 12.735 | - |
| task_269 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 19.645 | - |
| task_27 | easy | 3 | 2 | 3 | 0 | 1.0000 | 0.0000 | yes | 10.521 | - |
| task_283 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 13.230 | - |
| task_287 | medium | 1 | 2 | 1 | 1 | 1.0000 | 0.5000 | yes | 19.350 | All gold columns covered, with 1 extra prediction column(s). |
| task_292 | medium | 1 | 3 | 1 | 2 | 1.0000 | 0.6667 | yes | 8.585 | All gold columns covered, with 2 extra prediction column(s). |
| task_303 | medium | 1 | 2 | 1 | 1 | 1.0000 | 0.5000 | yes | 26.909 | All gold columns covered, with 1 extra prediction column(s). |
| task_305 | medium | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 19.632 | - |
| task_330 | hard | 2 | 3 | 2 | 1 | 1.0000 | 0.3333 | yes | 14.132 | All gold columns covered, with 1 extra prediction column(s). |
| task_344 | hard | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | no | 245.477 | Agent did not submit an answer within max_steps. |
| task_349 | hard | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 11.471 | - |
| task_350 | hard | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 43.782 | - |
| task_352 | hard | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | no | 129.292 | Agent did not submit an answer within max_steps. |
| task_355 | hard | 3 | 2 | 3 | 0 | 1.0000 | 0.0000 | yes | 26.311 | - |
| task_379 | hard | 1 | 2 | 0 | 2 | 0.0000 | 1.0000 | no | 31.822 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_38 | easy | 1 | 10 | 1 | 8 | 1.0000 | 0.8000 | yes | 122.028 | All gold columns covered, with 8 extra prediction column(s). |
| task_396 | hard | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | no | 147.472 | Agent did not submit an answer within max_steps. |
| task_408 | hard | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 25.098 | - |
| task_415 | hard | 2 | 2 | 2 | 0 | 1.0000 | 0.0000 | yes | 12.613 | - |
| task_418 | extreme | 1 | 1 | 0 | 1 | 0.0000 | 1.0000 | no | 141.487 | No gold columns matched; 1 prediction column(s) were produced and 1 are redundant. |
| task_420 | extreme | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 42.190 | - |
| task_64 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 10.919 | - |
| task_67 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 8.944 | - |
| task_74 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 8.869 | - |
| task_75 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 15.702 | - |
| task_80 | easy | 1 | 2 | 0 | 2 | 0.0000 | 1.0000 | no | 122.437 | No gold columns matched; 2 prediction column(s) were produced and 2 are redundant. |
| task_86 | easy | 1 | 3 | 0 | 3 | 0.0000 | 1.0000 | no | 23.579 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
| task_89 | easy | 1 | 3 | 0 | 3 | 0.0000 | 1.0000 | no | 19.993 | No gold columns matched; 3 prediction column(s) were produced and 3 are redundant. |
