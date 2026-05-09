from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget, load_app_config
from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
from data_agent_baseline.inspectors.prompts import build_global_profiling_prompt
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
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
        conn.execute("INSERT INTO races VALUES (1, 'Chinese Grand Prix')")
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


def test_semantic_catalog_handles_supported_assets_and_bad_json(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "broken.db").write_text("not a sqlite database", encoding="utf-8")

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(max_doc_chars=100),
    )

    asset_paths = {asset["path"] for asset in catalog["assets"]}
    assert {"results.csv", "data.json", "knowledge.md", "sample.db", "bad.json", "broken.db"} <= asset_paths
    assert any(schema["kind"] == "csv" and schema["asset_path"] == "results.csv" for schema in catalog["schemas"])
    assert any(schema["kind"] == "sqlite" and schema["asset_path"] == "sample.db" for schema in catalog["schemas"])
    sqlite_schema = next(schema for schema in catalog["schemas"] if schema["asset_path"] == "sample.db")
    races_table = next(table for table in sqlite_schema["tables"] if table["name"] == "races")
    race_id_field = next(field for field in races_table["fields"] if field["name"] == "raceId")
    name_field = next(field for field in races_table["fields"] if field["name"] == "name")
    assert race_id_field["distinct_values"] == ["1"]
    assert name_field["distinct_values"] == ["Chinese Grand Prix"]
    assert any(item["asset_path"] == "bad.json" for item in catalog["semantic_uncertainties"])
    assert any(item["asset_path"] == "broken.db" for item in catalog["semantic_uncertainties"])
    assert catalog["relationships"] == []


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

    def fail_explore(self, *, context_dir, task_id=""):
        raise RuntimeError("synthetic inspector failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert len(result.steps) > 0
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "global_data_exploration"
    assert first_step["ok"] is False


def test_langgraph_agent_records_global_exploration_failure_and_continues(tmp_path: Path, monkeypatch) -> None:
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
            ),
        ]
    )

    def fail_explore(self, *, context_dir, task_id=""):
        raise RuntimeError("synthetic profiling failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "global_data_exploration"
    assert first_step["ok"] is False
    assert "synthetic profiling failure" in first_step["tool_results"][0]["error"]


def test_langgraph_agent_receive_problem_injects_catalog(tmp_path: Path) -> None:
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
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_request = model.invocations[-1]
    injected_messages = [getattr(message, "content", "") for message in first_request]
    injected_text = "\n".join(str(content) for content in injected_messages)
    assert 'raw data catalog' in injected_text.lower() or "global data profile" in injected_text.lower()


def test_runner_writes_inspector_artifacts(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "none",
        "succeeded": False,
        "inspector": {
            "semantic_catalog": {"assets": []},
            "semantic_index": {"mode": "keyword"},
            "data_understanding_handoff": {"brief_markdown": "ok"},
            "handoff_status": "complete",
            "validation_errors": [],
            "inspector_steps": [{"phase": "deterministic_context"}],
            "global_data_profile": "# Test Profile\n\nSome profile content.",
        },
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    assert (run_output_dir / "task_demo" / "global_data_profile.md").exists()
    assert (run_output_dir / "task_demo" / "semantic_catalog.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_index.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_handoff.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_trace.json").exists()


def test_runner_writes_global_profile_from_top_level_result(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "stopped during grounding",
        "succeeded": False,
        "inspector": None,
        "global_data_profile": "## Global Data Profile\n\nRecovered from stage 1.",
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    profile_path = run_output_dir / "task_demo" / "global_data_profile.md"
    assert profile_path.read_text(encoding="utf-8") == "## Global Data Profile\n\nRecovered from stage 1."


def test_runner_preserves_global_profile_from_partial_trace(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    task_dir = run_output_dir / "task_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "trace.json").write_text(
        json.dumps(
            {
                "task_id": "task_demo",
                "answer": None,
                "steps": [{"node": "global_data_exploration"}],
                "failure_reason": None,
                "succeeded": False,
                "inspector": None,
                "partial": True,
                "global_data_profile": "## Global Data Profile\n\nPreserved from live trace.",
            }
        ),
        encoding="utf-8",
    )
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "Task failed before final state.",
        "succeeded": False,
        "inspector": None,
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    profile_path = task_dir / "global_data_profile.md"
    trace_payload = json.loads((task_dir / "trace.json").read_text(encoding="utf-8"))
    assert profile_path.read_text(encoding="utf-8") == "## Global Data Profile\n\nPreserved from live trace."
    assert trace_payload["global_data_profile"] == "## Global Data Profile\n\nPreserved from live trace."
    assert trace_payload["finalized_from_partial_trace"] is True


def test_explore_data_globally_returns_raw_catalog_json(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    profile = DataUnderstandingAgent(
        config=DataInspectorConfig(
            sample_budget=DataInspectorSampleBudget(max_doc_chars=20)
        ),
    ).explore_data_globally(context_dir=task.context_dir, task_id=task.task_id)

    payload = json.loads(profile)
    assert payload["phase"] == "global_data_profiling"
    assert "assets" in payload
    assert "schemas" in payload
    assert "task_id" in payload


def test_global_profiling_prompt_includes_full_knowledge_md() -> None:
    full_knowledge = "start\n" + ("domain rule " * 600) + "\nend"
    prompt = build_global_profiling_prompt(
        catalog={
            "task_id": "task_demo",
            "assets": [],
            "schemas": [],
            "relationships": [],
            "semantic_uncertainties": [],
        },
        knowledge_docs=[
            {
                "asset_path": "knowledge.md",
                "content": full_knowledge,
                "char_count": len(full_knowledge),
                "headings": ["Overview", "Domain Rules"],
            },
            {
                "asset_path": "doc/background.md",
                "content": "x" * 5000,
                "char_count": 5000,
                "headings": [],
            },
        ],
    )
    payload = json.loads(prompt)
    knowledge_doc, background_doc = payload["knowledge_documents"]

    assert knowledge_doc["asset_path"] == "knowledge.md"
    assert knowledge_doc["content"] == full_knowledge
    assert knowledge_doc["is_full_content"] is True
    assert knowledge_doc["headings"] == ["Overview", "Domain Rules"]
    assert background_doc["content"] == "x" * 5000
    assert background_doc["is_full_content"] is True
    assert background_doc["headings"] == []


def test_csv_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "id,operation,value\n"
        "1,VYBER,100\n"
        "2,VKLAD,200\n"
        "3,VYBER,300\n"
        "4,PREVOD,400\n"
        "5,VYBER,500\n"
        "6,VKLAD,600\n",
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "test.csv", budget)

    assert schema["row_count"] == 6
    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    op_field = next(f for f in schema["fields"] if f["name"] == "operation")
    val_field = next(f for f in schema["fields"] if f["name"] == "value")

    assert id_field["cardinality"] == 6
    assert id_field["distinct_values"] == ["1", "2", "3", "4", "5", "6"]

    assert op_field["cardinality"] == 3
    assert sorted(op_field["distinct_values"]) == ["PREVOD", "VKLAD", "VYBER"]
    assert val_field["cardinality"] == 6


def test_csv_schema_reports_cardinality_and_top_distinct_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "high_card.csv"
    rows = ["id,name"]
    for i in range(250):
        rows.append(f"{i},name_{i}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "high_card.csv", budget)

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    name_field = next(f for f in schema["fields"] if f["name"] == "name")

    assert id_field["cardinality"] == 250
    assert len(id_field["distinct_values"]) == 50
    assert name_field["cardinality"] == 250
    assert len(name_field["distinct_values"]) == 50


def test_sqlite_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE ops (id INTEGER, operation TEXT, value REAL)")
        conn.execute("INSERT INTO ops VALUES (1, 'VYBER', 100)")
        conn.execute("INSERT INTO ops VALUES (2, 'VKLAD', 200)")
        conn.execute("INSERT INTO ops VALUES (3, 'VYBER', 300)")
        conn.execute("INSERT INTO ops VALUES (4, 'PREVOD', 400)")
        conn.execute("INSERT INTO ops VALUES (5, 'VYBER', 500)")

    from data_agent_baseline.inspectors.semantic_catalog import _read_sqlite_schema

    budget = DataInspectorSampleBudget()
    schema = _read_sqlite_schema(db_path, "test.db", budget)

    table = schema["tables"][0]
    assert table["name"] == "ops"

    op_field = next(f for f in table["fields"] if f["name"] == "operation")
    assert op_field["cardinality"] == 3
    assert sorted(op_field["distinct_values"]) == ["PREVOD", "VKLAD", "VYBER"]

    id_field = next(f for f in table["fields"] if f["name"] == "id")
    assert id_field["cardinality"] == 5


def test_json_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(
        json.dumps([
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "A", "value": 30},
            {"category": "C", "value": 40},
        ]),
        encoding="utf-8",
    )

    from data_agent_baseline.inspectors.semantic_catalog import _read_json_schema

    budget = DataInspectorSampleBudget()
    schema = _read_json_schema(json_path, "data.json", budget)

    cat_field = next(f for f in schema["fields"] if f["name"] == "category")
    assert cat_field["cardinality"] == 3
    assert sorted(cat_field["distinct_values"]) == ["A", "B", "C"]

    val_field = next(f for f in schema["fields"] if f["name"] == "value")
    assert val_field["cardinality"] == 4


def test_global_profiling_prompt_includes_distinct_values() -> None:
    catalog = {
        "task_id": "task_demo",
        "assets": [],
        "schemas": [
            {
                "asset_path": "trans.csv",
                "kind": "csv",
                "row_count": 1000,
                "fields": [
                    {
                        "name": "operation",
                        "type": "string",
                        "missing_count": 0,
                        "cardinality": 5,
                        "distinct_values": ["PREVOD", "VKLAD", "VYBER", "VYBER_PREVOD", "VYBER_PREVOD_PLAT"],
                    },
                    {
                        "name": "amount",
                        "type": "number",
                        "missing_count": 0,
                        "cardinality": 999,
                        "distinct_values": ["100", "200", "300"],
                    },
                ],
            }
        ],
        "relationships": [],
        "semantic_uncertainties": [],
    }
    prompt = build_global_profiling_prompt(catalog=catalog, knowledge_docs=[])
    payload = json.loads(prompt)

    schema_summary = payload["schemas"][0]
    op_field = next(f for f in schema_summary["fields"] if f["name"] == "operation")
    assert op_field["cardinality"] == 5
    assert "VYBER_PREVOD_PLAT" in op_field["distinct_values"]

    amt_field = next(f for f in schema_summary["fields"] if f["name"] == "amount")
    assert amt_field["cardinality"] == 999
    assert amt_field["distinct_values"] == ["100", "200", "300"]


def test_csv_schema_includes_min_max_for_numeric_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "numeric.csv"
    csv_path.write_text(
        "id,amount,label\n"
        "1,100,sale\n"
        "2,200,refund\n"
        "3,150,sale\n",
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "numeric.csv", budget)

    amt_field = next(f for f in schema["fields"] if f["name"] == "amount")
    assert amt_field["min_value"] == 100.0
    assert amt_field["max_value"] == 200.0

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    assert id_field["min_value"] == 1.0
    assert id_field["max_value"] == 3.0

    lbl_field = next(f for f in schema["fields"] if f["name"] == "label")
    assert "min_value" not in lbl_field
    assert "max_value" not in lbl_field


def test_csv_schema_mixed_column_skips_min_max(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("id,note\n1,hello\n2,world\n", encoding="utf-8")
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "mixed.csv", budget)

    note_field = next(f for f in schema["fields"] if f["name"] == "note")
    assert "min_value" not in note_field

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    assert "min_value" in id_field


def test_sqlite_schema_includes_row_count_and_min_max(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, amt REAL, label TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 100.5, 'sale')")
        conn.execute("INSERT INTO t VALUES (2, 200.0, 'refund')")
        conn.execute("INSERT INTO t VALUES (3, 150.0, 'sale')")

    from data_agent_baseline.inspectors.semantic_catalog import _read_sqlite_schema

    budget = DataInspectorSampleBudget()
    schema = _read_sqlite_schema(db_path, "test.db", budget)

    table = schema["tables"][0]
    assert table["row_count"] == 3

    amt_field = next(f for f in table["fields"] if f["name"] == "amt")
    assert amt_field["min_value"] == 100.5
    assert amt_field["max_value"] == 200.0

    id_field = next(f for f in table["fields"] if f["name"] == "id")
    assert id_field["min_value"] == 1.0
    assert id_field["max_value"] == 3.0

    lbl_field = next(f for f in table["fields"] if f["name"] == "label")
    assert "min_value" not in lbl_field


def test_json_schema_includes_min_max_for_numeric_fields(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(
        json.dumps([
            {"score": 95.5, "grade": "A"},
            {"score": 72.0, "grade": "B"},
            {"score": 88.0, "grade": "A"},
        ]),
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_json_schema

    budget = DataInspectorSampleBudget()
    schema = _read_json_schema(json_path, "data.json", budget)

    score_field = next(f for f in schema["fields"] if f["name"] == "score")
    assert score_field["min_value"] == 72.0
    assert score_field["max_value"] == 95.5

    grade_field = next(f for f in schema["fields"] if f["name"] == "grade")
    assert "min_value" not in grade_field


def test_data_inspector_config_parses_sample_budget(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "dataset:\n  root_path: data/public/input\n"
        "agent:\n  model: test\n  api_base: http://localhost/v1\n  api_key: ''\n  api_key_env: TEST\n"
        "  max_steps: 16\n  temperature: 0.0\n  enable_data_inspector: true\n"
        "  model_request_timeout_seconds: 120\n"
        "data_inspector:\n"
        "  sample_budget:\n"
        "    catalog_top_distinct_values: 100\n"
        "    max_doc_chars: 5000\n"
        "run:\n  output_dir: artifacts/runs\n  max_workers: 4\n  task_timeout_seconds: 600\n",
        encoding="utf-8",
    )
    config = load_app_config(yaml_path)
    assert config.data_inspector.sample_budget.catalog_top_distinct_values == 100
    assert config.data_inspector.sample_budget.max_doc_chars == 5000


def test_explore_data_globally_returns_json_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    agent = DataUnderstandingAgent(config=DataInspectorConfig())
    profile = agent.explore_data_globally(
        context_dir=task.context_dir,
        task_id=task.task_id,
    )
    assert profile.startswith("{")
    payload = json.loads(profile)
    assert payload["phase"] == "global_data_profiling"
    assert "assets" in payload
    assert "schemas" in payload


def test_receive_problem_injects_catalog_message(tmp_path: Path) -> None:
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
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )
    result = agent.run(task)
    assert result.succeeded is True


def test_master_switch_still_disables_both_nodes(tmp_path: Path) -> None:
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
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=False,
            data_inspector=DataInspectorConfig(),
        ),
    )
    result = agent.run(task)
    assert result.succeeded is True
    assert result.global_data_profile is None
    assert result.inspector is None
