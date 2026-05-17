from __future__ import annotations

import operator
from typing import Any
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from data_agent_baseline.benchmark.schema import AnswerTable


class AgentGraphState(TypedDict, total=False):
    task_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    empty_stop_retry_count: int
    # Counts answer-validation rejections so validation cannot loop forever.
    validation_retry_count: int
    answer_validation_history: Annotated[list[dict[str, Any]], operator.add]
    answer: AnswerTable | None
    failure_reason: str | None
    steps: Annotated[list[dict[str, Any]], operator.add]
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    temp_workspace: str | None
    started_at: str
    inspector: dict[str, Any] | None
    global_data_profile: str | None
    ambiguity_analysis: dict[str, Any] | None
    document_context: str | None
