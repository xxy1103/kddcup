# Objective
将现有 Agent 架构改造为“两阶段数据理解”流程：首先在**没有具体问题**的情况下全局探索和分析数据（融合原有的 Perception 功能），生成全局的数据画像（Data Profile）；然后再引入用户问题，结合该数据画像生成精确的实体映射和查询契约（Data Understanding Handoff），最后交由主 Agent 开始执行。

# Key Files & Context
- `src/data_agent_baseline/agents/langgraph_runtime.py`：LangGraph 状态图的编排。
- `src/data_agent_baseline/inspectors/data_understanding_agent.py`：DataUnderstandingAgent 的核心逻辑。
- `src/data_agent_baseline/inspectors/perception.py`：现有的 Perception 逻辑（将被重构或融合）。
- `src/data_agent_baseline/agents/state.py`：AgentGraphState 定义。

# Implementation Steps

## 1. 改造 DataUnderstandingAgent 支持两阶段流程
在 `data_understanding_agent.py` 中重构：
- **阶段 1：全局数据探索 (Profile Generation)**
  - 新增方法 `explore_data_globally(context_dir)`。该阶段**不输入任务问题**。
  - 调用 `build_semantic_catalog`，提取 Schema、样本数据。
  - 整合并阅读 `knowledge.md` 等通用文档（取代并合并原有 `perception` 提取领域名词的逻辑）。
  - 利用大模型生成一份 `GlobalDataProfile`，归纳库中含有哪些表、表的核心含义以及显而易见的关联与质量特征。
- **阶段 2：问题导向理解 (Handoff Generation)**
  - 改造主入口，使其接收 `task` 和上一步生成的 `GlobalDataProfile`。
  - 在生成 `context_bundle` 和执行现有的 Guided Loop（Overview, Grounding, Fabric, Contract）时，将 `GlobalDataProfile` 注入到 Prompt 中，使模型基于全局认知来生成针对当前问题的约束与映射。

## 2. 重新编排 LangGraph 工作流
在 `langgraph_runtime.py` 中更新节点和连线：
- **移除** 独立的 `perceive_task` 节点。
- **新增节点 `global_data_exploration`**：执行阶段 1，将 `GlobalDataProfile` 写入图状态（Graph State）。
- **新增节点 `receive_problem`**：读取 `task.question` 并将其以 `HumanMessage` 的形式附加到 `messages` 中，正式引入问题。
- **重命名/修改节点 `problem_grounding`**（原 `understand_and_explore_data`）：执行阶段 2，生成并应用 `DataUnderstandingHandoff`。
- **更新连线逻辑**：`START -> init_state -> global_data_exploration -> receive_problem -> problem_grounding -> model_step -> ...`

## 3. 更新图状态 (State) 
在 `state.py` 中：
- 扩展 `AgentGraphState`，增加对 `global_data_profile` 和分阶段过程的存储支持。

## 4. Prompt 更新
- 为阶段 1 编写无问题的 Data Profiling Prompt。
- 修改阶段 2 和主 Agent 的 Prompt，确保它们能够正确利用阶段 1 的产物。

# Verification & Testing
- 运行现有的测试集或基准脚本进行单任务测试。
- 在 Trace（执行日志）中确认：
  1. 状态图首先进入探索节点，并输出不含问题的全局分析总结。
  2. 随后日志中记录了“问题”的出现。
  3. Grounding 和 Contract 成功基于前面的全局分析生成。
  4. 主 Agent 成功获取完整上下文并顺利求解。