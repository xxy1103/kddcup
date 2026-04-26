from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget
from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
from data_agent_baseline.inspectors.exchange import AgentEnvelope, AgentEnvelopeContent
from data_agent_baseline.inspectors.perception import build_perception_envelope
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
from data_agent_baseline.inspectors.semantic_index import build_semantic_index
from data_agent_baseline.inspectors.semantic_query import SemanticQueryTools
from data_agent_baseline.run.runner import _write_task_outputs
from data_agent_baseline.tools.registry import create_default_tool_registry


def _create_task(tmp_path: Path, question: str = "What's the finish time for the ranked second driver?") -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "results.csv").write_text(
        "raceId,driverId,rank,position,time\n1,10,2,1,1:30.000\n",
        encoding="utf-8",
    )
    (context_dir / "bad.json").write_text("{not json", encoding="utf-8")
    (context_dir / "data.json").write_text(
        json.dumps([{"raceId": 1, "name": "Chinese Grand Prix", "year": 2008}]),
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text("# Notes\nrank differs from position\n", encoding="utf-8")
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_cost_event_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_25_like"
    context_dir = task_dir / "context"
    (context_dir / "json").mkdir(parents=True)
    (context_dir / "csv").mkdir()
    (context_dir / "json" / "event.json").write_text(
        json.dumps(
            {
                "table": "event",
                "records": [
                    {"event_id": "event_a", "event_name": "November Speaker"},
                    {"event_id": "event_b", "event_name": "October Speaker"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "json" / "expense.json").write_text(
        json.dumps(
            {
                "table": "expense",
                "records": [
                    {"expense_id": "exp_a", "cost": 6.0, "link_to_budget": "budget_a"},
                    {"expense_id": "exp_b", "cost": 6.0, "link_to_budget": "budget_b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "csv" / "budget.csv").write_text(
        "budget_id,amount,spent,link_to_event\n"
        "budget_a,10,6,event_a\n"
        "budget_b,10,6,event_b\n",
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text(
        "# Financials\n"
        "- amount: budgeted amount, not actual cost.\n"
        "- spent: total expenditure associated with a budget.\n"
        "- cost: individual expense amount.\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id="task_25_like", difficulty="easy", question="Which event has the lowest cost?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class InvalidJsonSynthesisModel:
    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(content="not json")


class SequenceSynthesisModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("No scripted synthesis responses remaining.")
        return AIMessage(content=self.responses.pop(0))


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.invocations = []

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        del tools, tool_choice, parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(messages)
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        return self._responses.pop(0)


def test_perception_detects_answer_shape_and_high_risk_terms(tmp_path: Path) -> None:
    task = _create_task(
        tmp_path,
        "What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
    )

    envelope = build_perception_envelope(task)

    payload = envelope.content.payload
    assert payload["expected_answer_shape"]["row_shape"] == "single_row"
    assert payload["expected_answer_shape"]["column_hint"] == "finish time"
    assert "rank_position_ambiguity" in payload["high_risk_terms"]
    assert envelope.message_type == "perception_result"


def test_semantic_catalog_handles_supported_assets_and_bad_json(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(catalog_sample_rows=2, max_doc_chars=100, max_json_chars=100),
    )

    asset_paths = {asset["path"] for asset in catalog["assets"]}
    assert {"results.csv", "data.json", "knowledge.md", "sample.db", "bad.json"} <= asset_paths
    assert any(schema["kind"] == "csv" and schema["asset_path"] == "results.csv" for schema in catalog["schemas"])
    assert any(schema["kind"] == "sqlite" and schema["asset_path"] == "sample.db" for schema in catalog["schemas"])
    assert any(item["asset_path"] == "bad.json" for item in catalog["semantic_uncertainties"])
    assert any(item["field"] == "raceid" for item in catalog["relationships"])


def test_semantic_index_matches_query_tokens_and_risk_candidates(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = build_perception_envelope(task).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)

    assert index["query_index"]["time"]["fields"]
    assert index["risk_index"]["rank_position_ambiguity"]["rank"]
    assert index["risk_index"]["rank_position_ambiguity"]["position"]


def test_semantic_query_tools_ground_cost_event_and_join_path(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, max_join_hops=3)

    search = tools.search_semantic_index("cost event")
    assert any(item["field"] == "records.cost" for item in search["fields"])
    assert any(item["field"] == "records.event_name" for item in search["fields"])

    knowledge_hits = tools.lookup_knowledge("amount")
    assert any("budgeted amount" in item["snippet"] for item in knowledge_hits)

    paths = tools.find_join_paths("records.cost", "records.event_name")
    rendered = json.dumps(paths, ensure_ascii=False)
    assert "link_to_budget" in rendered
    assert "budget_id" in rendered
    assert "link_to_event" in rendered
    assert "event_id" in rendered


def test_agent_envelope_rejects_invalid_message_type() -> None:
    with pytest.raises(ValidationError):
        AgentEnvelope(
            task_id="task_demo",
            sender="a",
            recipient="b",
            message_type="unknown",
            content=AgentEnvelopeContent(),
        )


def test_data_understanding_agent_falls_back_when_synthesis_json_is_invalid(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = build_perception_envelope(task)
    agent = DataUnderstandingAgent(
        model=InvalidJsonSynthesisModel(),
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert result.semantic_catalog["assets"]
    assert "Data Understanding Brief" in result.summary


def test_data_understanding_agent_retries_synthesis_once_and_keeps_notes(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task)
    model = SequenceSynthesisModel(
        [
            "not json",
            json.dumps(
                {
                    "semantic_notes": ["json/expense.json.records.cost is the actual cost field."],
                    "uncertainties": ["Verify whether ties should be preserved."],
                }
            ),
        ]
    )
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 2
    assert result.synthesis is not None
    assert "recovered after retry" in str(result.synthesis_error)
    handoff = result.data_understanding_handoff
    assert "json/expense.json.records.cost is the actual cost field." in handoff["llm_notes"]


def test_data_understanding_agent_discards_synthesis_after_two_invalid_attempts(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task)
    model = SequenceSynthesisModel(["not json", "still not json"])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 2
    assert result.synthesis is None
    assert "discarded after retry" in str(result.synthesis_error)
    assert result.data_understanding_handoff["llm_notes"] == []


def test_data_understanding_agent_retries_unknown_field_reference_and_discards(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task)
    invalid_payload = json.dumps(
        {
            "semantic_notes": ["Use json/expense.json.records.missing_cost."],
            "uncertainties": [],
        }
    )
    model = SequenceSynthesisModel([invalid_payload, invalid_payload])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 2
    assert result.synthesis is None
    assert "outside whitelist" in str(result.synthesis_error)


def test_data_understanding_agent_allows_known_asset_references_in_notes(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task)
    payload = json.dumps(
        {
            "semantic_notes": [
                "Compare json/expense.json and json/event.json before choosing json/expense.json.records.cost."
            ],
            "uncertainties": ["json/expense.json may need joining to json/event.json."],
        }
    )
    model = SequenceSynthesisModel([payload])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 1
    assert result.synthesis is not None
    assert result.synthesis_error is None
    assert result.data_understanding_handoff["llm_notes"]


def test_task_25_like_handoff_has_grounding_join_path_and_answer_contract(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = build_perception_envelope(task)
    agent = DataUnderstandingAgent(
        model=None,
        config=DataInspectorConfig(mode="rules"),
    )

    result = agent.run(task, perception)
    handoff = result.data_understanding_handoff

    assert handoff["answer_contract"]["columns"] == ["event_name"]
    assert handoff["answer_contract"]["metric_operation"] == "min"
    assert handoff["answer_contract"]["row_policy"] == "preserve_all_ties"
    assert handoff["answer_contract"]["metric_field"] == "json/expense.json.records.cost"
    brief = handoff["brief_markdown"]
    assert "Do not use csv/budget.csv.amount" in brief
    assert "Do not use csv/budget.csv.spent" in brief
    assert "link_to_budget" in brief
    assert "link_to_event" in brief


def test_langgraph_agent_data_inspector_failure_does_not_block_answer(tmp_path: Path, monkeypatch) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["ok"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    def fail_perception(task):  # noqa: ANN001
        del task
        raise RuntimeError("synthetic inspector failure")

    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.build_perception_envelope", fail_perception)
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2, enable_data_inspector=True),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.inspector is not None
    assert "Perception failed" in result.inspector["error"]


def test_langgraph_agent_injects_full_data_understanding_handoff(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["event_name"], "rows": [["November Speaker"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(mode="rules"),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_request = model.invocations[0]
    injected_messages = [getattr(message, "content", "") for message in first_request]
    injected_text = "\n".join(str(content) for content in injected_messages)
    assert "Full data_understanding_handoff.json" in injected_text
    assert '"question_grounding"' in injected_text
    assert '"data_fabric"' in injected_text
    assert '"answer_contract"' in injected_text
    assert '"metric_field": "json/expense.json.records.cost"' in injected_text


def test_runner_writes_inspector_artifacts(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "none",
        "succeeded": False,
        "inspector": {
            "perception": {"ok": True},
            "semantic_catalog": {"assets": []},
            "semantic_index": {"mode": "keyword"},
            "data_understanding_handoff": {"brief_markdown": "ok"},
        },
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    assert (run_output_dir / "task_demo" / "perception.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_catalog.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_index.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_handoff.json").exists()
