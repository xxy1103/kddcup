from __future__ import annotations

from datetime import date
import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget, load_app_config
from data_agent_baseline.inspectors.data_understanding_agent import (
    ContractDraft,
    DataUnderstandingAgent,
    FabricDraft,
    GuidedDataUnderstandingLoop,
    GroundingDraft,
    OverviewDraft,
    RepairDraft,
    ToolRequest,
    _apply_repair_draft,
    _compact_tool_result,
    _validate_contract_draft,
)
from data_agent_baseline.inspectors.exchange import AgentEnvelope, AgentEnvelopeContent
from data_agent_baseline.inspectors.perception import PerceptionBuildError, build_perception_envelope
from data_agent_baseline.inspectors.prompts import (
    build_global_profiling_prompt,
    build_guided_phase_prompt,
    build_guided_retry_prompt,
)
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


def _create_driver_records_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_86_like"
    context_dir = task_dir / "context"
    (context_dir / "json").mkdir(parents=True)
    (context_dir / "csv").mkdir()
    (context_dir / "json" / "drivers.json").write_text(
        json.dumps(
            {
                "table": "drivers",
                "records": [
                    {"driverId": 1, "forename": "Lewis", "surname": "Hamilton", "number": 44},
                    {"driverId": 62, "forename": "Alex", "surname": "Yoong", "number": 17},
                ],
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "csv" / "driverStandings.csv").write_text(
        "driverStandingsId,raceId,driverId,points,position,positionText,wins\n"
        "1,10,62,0,12,12,0\n",
        encoding="utf-8",
    )
    (context_dir / "csv" / "races.csv").write_text(
        "raceId,year,round,circuitId,name,date,time,url\n"
        "10,2002,1,1,Australian Grand Prix,2002-03-03,00:00:00,http://example.test/race\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id="task_86_like",
            difficulty="easy",
            question="Which race was Alex Yoong in when he was in track number less than 20?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_patient_exam_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_11_like"
    context_dir = task_dir / "context"
    (context_dir / "json").mkdir(parents=True)
    (context_dir / "json" / "clinical.json").write_text(
        json.dumps(
            {
                "Patient": [
                    {"ID": "P001", "SEX": "F", "Diagnosis": "case"},
                    {"ID": "P002", "SEX": "M", "Diagnosis": "control"},
                ],
                "Examination": [
                    {"ID": "P001", "Diagnosis": "exam-positive"},
                    {"ID": "P003", "Diagnosis": "exam-positive"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id="task_11_like",
            difficulty="medium",
            question="List ID, SEX, and Diagnosis for patients with an exam-positive examination record.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_sat_frpm_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_199_like"
    context_dir = task_dir / "context"
    (context_dir / "db").mkdir(parents=True)
    (context_dir / "csv").mkdir()
    with sqlite3.connect(context_dir / "db" / "satscores.db") as conn:
        conn.execute(
            "CREATE TABLE satscores (cds TEXT, rtype TEXT, sname TEXT, AvgScrMath REAL)"
        )
        conn.execute("INSERT INTO satscores VALUES ('1', 'S', 'Riverside High', 510)")
        conn.execute("INSERT INTO satscores VALUES ('2', 'D', NULL, 500)")
    (context_dir / "csv" / "frpm.csv").write_text(
        "CDSCode,District Name,County Name,School Name,Charter Funding Type\n"
        "1,Riverside Unified,Riverside,Riverside High,Directly funded\n"
        "2,Riverside Unified,Riverside,,\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id="task_199_like",
            difficulty="medium",
            question=(
                "List sname and Charter Funding Type for schools from Riverside-related school districts "
                "with available SAT math scores."
            ),
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class InvalidJsonSynthesisModel:
    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(content="not json")


class SequenceSynthesisModel:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("No scripted synthesis responses remaining.")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return AIMessage(content=response)


def _perception_response(
    task: PublicTask,
    *,
    entities: list[str] | None = None,
    metrics: list[str] | None = None,
    filter_phrases: list[str] | None = None,
    row_shape: str = "single_row",
    column_hint: str = "unknown",
    high_risk_terms: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "question": task.question,
            "difficulty": task.difficulty,
            "entities": entities or ["driver"],
            "metrics": metrics or [],
            "filter_phrases": filter_phrases or [],
            "expected_answer_shape": {
                "row_shape": row_shape,
                "column_hint": column_hint,
                "only_requested_columns": True,
            },
            "high_risk_terms": high_risk_terms or [],
        }
    )


def _build_test_perception(
    task: PublicTask,
    *,
    entities: list[str] | None = None,
    metrics: list[str] | None = None,
    filter_phrases: list[str] | None = None,
    row_shape: str = "single_row",
    column_hint: str = "unknown",
    high_risk_terms: list[str] | None = None,
) -> AgentEnvelope:
    model = SequenceSynthesisModel(
        [
            _perception_response(
                task,
                entities=entities,
                metrics=metrics,
                filter_phrases=filter_phrases,
                row_shape=row_shape,
                column_hint=column_hint,
                high_risk_terms=high_risk_terms,
            )
        ]
    )
    return build_perception_envelope(task, model)


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


TEST_GLOBAL_DATA_PROFILE = (
    "## Global Data Profile\n\n"
    "- Assets, schemas, entity semantics, and relationships were already profiled by global_data_exploration."
)


def test_data_inspector_config_parses_profile_guided_fast_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data_inspector:\n"
        "  profile_guided_fast_path: false\n",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert DataInspectorConfig().profile_guided_fast_path is True
    assert config.data_inspector.profile_guided_fast_path is False


def test_perception_model_builds_answer_shape_and_high_risk_terms(tmp_path: Path) -> None:
    task = _create_task(
        tmp_path,
        "What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
    )
    model = SequenceSynthesisModel(
        [
            _perception_response(
                task,
                entities=["driver", "Chinese Grand Prix", "2008"],
                metrics=["finish time"],
                filter_phrases=["driver who ranked second", "in 2008's Chinese Grand Prix"],
                row_shape="single_row",
                column_hint="finish time",
                high_risk_terms=["rank_position_ambiguity"],
            )
        ]
    )

    envelope = build_perception_envelope(task, model)

    payload = envelope.content.payload
    assert payload["expected_answer_shape"]["row_shape"] == "single_row"
    assert payload["expected_answer_shape"]["column_hint"] == "finish time"
    assert "rank_position_ambiguity" in payload["high_risk_terms"]
    assert envelope.message_type == "perception_result"
    assert model.call_count == 1


def test_perception_model_retries_invalid_json_once(tmp_path: Path) -> None:
    task = _create_task(tmp_path, "List facilities with an average inspection score above 90.")
    model = SequenceSynthesisModel(
        [
            "not json",
            _perception_response(
                task,
                entities=["facilities"],
                metrics=["average inspection score"],
                filter_phrases=["with an average inspection score above 90"],
                row_shape="multiple_rows",
                column_hint="facilities",
                high_risk_terms=["aggregation_grain"],
            ),
        ]
    )

    envelope = build_perception_envelope(task, model)

    assert model.call_count == 2
    assert envelope.content.payload["filter_phrases"] == ["with an average inspection score above 90"]
    assert envelope.content.payload["high_risk_terms"] == ["aggregation_grain"]


def test_perception_model_retries_request_error_with_backoff(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    task = _create_task(tmp_path, "List facilities with an average inspection score above 90.")
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = SequenceSynthesisModel(
        [
            RuntimeError("temporary request failure"),
            _perception_response(
                task,
                entities=["facilities"],
                metrics=["average inspection score"],
                filter_phrases=["with an average inspection score above 90"],
                row_shape="multiple_rows",
                column_hint="facilities",
                high_risk_terms=["aggregation_grain"],
            ),
        ]
    )

    envelope = build_perception_envelope(task, model)

    assert model.call_count == 2
    assert sleep_delays == [15]
    assert envelope.content.payload["high_risk_terms"] == ["aggregation_grain"]


def test_perception_request_error_exhaustion_does_not_trigger_format_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path, "List facilities with an average inspection score above 90.")
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = SequenceSynthesisModel(
        [
            RuntimeError("temporary request failure 1"),
            RuntimeError("temporary request failure 2"),
            RuntimeError("temporary request failure 3"),
            RuntimeError("temporary request failure 4"),
            RuntimeError("temporary request failure 5"),
        ]
    )

    with pytest.raises(PerceptionBuildError) as exc_info:
        build_perception_envelope(task, model)

    assert model.call_count == 5
    assert sleep_delays == [15, 30, 45, 60]
    assert len(exc_info.value.attempts) == 1
    assert "Perception request failed" in str(exc_info.value)


def test_perception_model_fails_after_retry_without_rules_fallback(tmp_path: Path) -> None:
    task = _create_task(tmp_path, "List facilities with an average inspection score above 90.")
    model = SequenceSynthesisModel(["not json", "still not json"])

    with pytest.raises(PerceptionBuildError) as exc_info:
        build_perception_envelope(task, model)

    assert model.call_count == 2
    assert len(exc_info.value.attempts) == 2


def test_perception_preserves_complete_multiword_filters_and_risks(tmp_path: Path) -> None:
    task = _create_task(
        tmp_path,
        "List names and categories of facilities from coastal city departments where the average inspection score across sites exceeds 90.",
    )
    envelope = _build_test_perception(
        task,
        entities=["facilities", "coastal city departments", "sites"],
        metrics=["average inspection score"],
        filter_phrases=[
            "from coastal city departments",
            "where the average inspection score across sites exceeds 90",
        ],
        row_shape="multiple_rows",
        column_hint="names and categories",
        high_risk_terms=["geographic_scope_ambiguity", "aggregation_grain"],
    )

    payload = envelope.content.payload
    assert "from coastal city departments" in payload["filter_phrases"]
    assert "coastal city departments" in payload["entities"]
    assert "aggregation_grain" in payload["high_risk_terms"]


def test_semantic_catalog_handles_supported_assets_and_bad_json(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "broken.db").write_text("not a sqlite database", encoding="utf-8")

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(catalog_sample_rows=2, max_doc_chars=100, max_json_chars=100),
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
    assert races_table["sample_rows"] == [["1", "Chinese Grand Prix"]]
    assert any(item["asset_path"] == "bad.json" for item in catalog["semantic_uncertainties"])
    assert any(item["asset_path"] == "broken.db" for item in catalog["semantic_uncertainties"])
    assert catalog["relationships"] == []


def test_semantic_index_matches_query_tokens_and_risk_candidates(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = _build_test_perception(
        task,
        metrics=["finish time"],
        filter_phrases=["driver who ranked second"],
        column_hint="finish time",
        high_risk_terms=["rank_position_ambiguity"],
    ).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)

    assert index["query_index"]["time"]["fields"]
    assert index["risk_index"]["rank_position_ambiguity"]["rank"]
    assert index["risk_index"]["rank_position_ambiguity"]["position"]


def test_semantic_query_tools_ground_cost_event_and_join_path(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event",
    ).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, )

    search = tools.search_semantic_index("cost event")
    assert any(item["field"] == "records.cost" for item in search["fields"])
    assert any(item["field"] == "records.event_name" for item in search["fields"])

    knowledge_hits = tools.lookup_knowledge("amount")
    assert any("budgeted amount" in item["snippet"] for item in knowledge_hits)


def test_semantic_query_tools_resolves_sqlite_table_schema_refs(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = _build_test_perception(task).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, )

    schema = tools.get_asset_schema("sample.db.races")

    assert schema is not None
    assert schema["asset_path"] == "sample.db"
    assert [table["name"] for table in schema["tables"]] == ["races"]


def test_probe_tools_query_sqlite_tables_and_full_refs(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = _build_test_perception(task).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(
        catalog=catalog,
        semantic_index=index,
        limit=5,
        context_dir=task.context_dir,
    )

    bare = tools.execute_probe_query("SELECT name FROM races WHERE raceId = 1")
    full_ref = tools.execute_probe_query("SELECT name FROM sample.db.races WHERE raceId = 1")

    assert bare["ok"] is True
    assert bare["rows"] == [["Chinese Grand Prix"]]
    assert full_ref["ok"] is True
    assert full_ref["rows"] == [["Chinese Grand Prix"]]
    assert full_ref["normalized_sql"] == 'SELECT name FROM "races" WHERE raceId = 1'


def test_probe_sqlite_table_name_conflicts_require_full_ref_alias(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "races.csv").write_text("name\nCSV Race\n", encoding="utf-8")
    perception = _build_test_perception(task).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(
        catalog=catalog,
        semantic_index=index,
        limit=5,
        context_dir=task.context_dir,
    )

    bare = tools.execute_probe_query("SELECT name FROM races")
    full_ref = tools.execute_probe_query("SELECT name FROM sample.db.races WHERE raceId = 1")

    assert bare["ok"] is True
    assert bare["rows"] == [["CSV Race"]]
    assert full_ref["ok"] is True
    assert full_ref["rows"] == [["Chinese Grand Prix"]]
    assert full_ref["normalized_sql"] == 'SELECT name FROM "sample__races" WHERE raceId = 1'


def test_probe_tools_expand_json_records_and_normalize_asset_refs(tmp_path: Path) -> None:
    task = _create_driver_records_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["Alex Yoong", "race"],
        metrics=["track number"],
        filter_phrases=["track number less than 20"],
    ).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(
        catalog=catalog,
        semantic_index=index,
        limit=5,
        context_dir=task.context_dir,
    )

    path_style = tools.execute_probe_query(
        "SELECT d.records.forename, d.records.surname, d.records.driverId, d.records.number "
        "FROM json/drivers.json.records d "
        "WHERE LOWER(d.records.forename) = 'alex' AND LOWER(d.records.surname) = 'yoong'"
    )
    flat_style = tools.execute_probe_query(
        "SELECT records.forename, records.surname, records.driverId, records.number "
        "FROM drivers "
        "WHERE LOWER(records.forename) = 'alex' AND LOWER(records.surname) = 'yoong'"
    )
    join_style = tools.execute_probe_query(
        "SELECT r.name FROM csv/driverStandings.csv ds "
        "JOIN csv/races.csv r ON ds.raceId = r.raceId "
        "WHERE ds.driverId IN ("
        "SELECT records.driverId FROM json/drivers.json.records "
        "WHERE LOWER(records.forename) = 'alex' AND LOWER(records.surname) = 'yoong'"
        ")"
    )
    distinct = tools.get_column_distinct_values("drivers", "records.number")
    distinct_by_asset_path = tools.get_column_distinct_values("json/drivers.json", "records.number")

    assert path_style["ok"] is True
    assert path_style["rows"] == [["Alex", "Yoong", 62, 17]]
    assert path_style["normalized_sql"]
    assert flat_style["ok"] is True
    assert flat_style["rows"] == [["Alex", "Yoong", 62, 17]]
    assert join_style["ok"] is True
    assert join_style["rows"] == [["Australian Grand Prix"]]
    assert distinct["ok"] is True
    assert any(item["value"] == 17 for item in distinct["values"])
    assert distinct_by_asset_path["ok"] is True
    assert any(item["value"] == 17 for item in distinct_by_asset_path["values"])


def test_probe_tools_expand_large_json_records(tmp_path: Path) -> None:
    json_path = tmp_path / "posts.json"
    json_path.write_text(
        json.dumps(
            {
                "table": "posts",
                "records": [
                    {
                        "Id": 257,
                        "Title": "Computer Game Datasets",
                        "ViewCount": 12345,
                        "Body": "x" * (17 * 1024 * 1024),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    catalog = {
        "schemas": [
            {
                "asset_path": "posts.json",
                "kind": "json",
                "fields": [
                    {"name": "records.Id"},
                    {"name": "records.Title"},
                    {"name": "records.ViewCount"},
                    {"name": "records.Body"},
                ],
            }
        ]
    }
    tools = SemanticQueryTools(
        catalog=catalog,
        semantic_index={},
        limit=5,
        context_dir=tmp_path,
    )

    result = tools.execute_probe_query(
        "SELECT Id, Title, ViewCount FROM posts WHERE Title = 'Computer Game Datasets'"
    )

    assert result["ok"] is True
    assert result["rows"] == [[257, "Computer Game Datasets", 12345]]


def test_lookup_knowledge_searches_full_document_beyond_preview(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    knowledge_path = task.context_dir / "knowledge.md"
    knowledge_path.write_text(
        "# Notes\n"
        "rank differs from position\n"
        + ("filler line\n" * 30)
        + "Late rule: severe thrombosis means Thrombosis = 2.\n",
        encoding="utf-8",
    )
    perception = _build_test_perception(
        task,
        metrics=["finish time"],
        filter_phrases=["ranked second"],
        column_hint="finish time",
        high_risk_terms=["rank_position_ambiguity"],
    ).content.payload
    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(catalog_sample_rows=2, max_doc_chars=40, max_json_chars=100),
    )
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, )

    knowledge_hits = tools.lookup_knowledge("severe thrombosis")

    assert any("Thrombosis = 2" in item["snippet"] for item in knowledge_hits)


def test_agent_envelope_rejects_invalid_message_type() -> None:
    with pytest.raises(ValidationError):
        AgentEnvelope(
            task_id="task_demo",
            sender="a",
            recipient="b",
            message_type="unknown",
            content=AgentEnvelopeContent(),
        )


def test_data_understanding_agent_falls_back_when_guided_phase_json_is_invalid(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    perception = _build_test_perception(
        task,
        metrics=["finish time"],
        filter_phrases=["ranked second"],
        column_hint="finish time",
        high_risk_terms=["rank_position_ambiguity"],
    )
    agent = DataUnderstandingAgent(
        model=InvalidJsonSynthesisModel(),
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception, global_data_profile=TEST_GLOBAL_DATA_PROFILE)

    assert result.handoff_status == "fallback"
    assert result.validation_errors
    assert result.semantic_catalog["assets"]
    assert "Data Understanding Brief" in result.summary


def _guided_cost_event_responses() -> list[str]:
    return [
        json.dumps(
            {
                "task_intent": "Return the event name with the lowest actual expense cost.",
                "concepts": [
                    {"term": "event", "role": "answer_entity"},
                    {"term": "cost", "role": "metric"},
                    {"term": "lowest", "role": "operation"},
                ],
                "ambiguity_targets": ["cost versus budget amount/spent"],
                "tool_requests": [
                    {"tool": "search_semantic_index", "args": {"query": "cost event"}},
                ],
            }
        ),
        json.dumps(
            {
                "grounded_concepts": [
                    {
                        "term": "event",
                        "role": "answer_entity",
                        "accepted_fields": [
                            {
                                "field_ref": "json/event.json.records.event_name",
                                "confidence": "high",
                                "reason": "Event name is the requested output entity.",
                            }
                        ],
                        "rejected_fields": [],
                    },
                    {
                        "term": "cost",
                        "role": "metric",
                        "accepted_fields": [
                            {
                                "field_ref": "json/expense.json.records.cost",
                                "confidence": "high",
                                "reason": "Cost is the individual expense amount.",
                            }
                        ],
                        "rejected_fields": [
                            {"field_ref": "csv/budget.csv.amount", "reason": "Budgeted amount, not actual expense cost."},
                            {"field_ref": "csv/budget.csv.spent", "reason": "Budget aggregate, not individual expense cost."},
                        ],
                    },
                    {"term": "lowest", "role": "operation", "accepted_fields": [], "rejected_fields": []},
                ],
                "remaining_uncertainties": [],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "join_paths": [
                    {
                        "purpose": "Link each expense to its event through budget.",
                        "path": [
                            {
                                "from_field": "json/expense.json.records.link_to_budget",
                                "to_field": "csv/budget.csv.budget_id",
                            },
                            {
                                "from_field": "csv/budget.csv.link_to_event",
                                "to_field": "json/event.json.records.event_id",
                            },
                        ],
                        "confidence": "high",
                    }
                ],
                "data_grain": "one row per tied minimum-cost event",
                "relationship_risks": [],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "answer_columns": [
                    {
                        "name": "event_name",
                        "source_field": "json/event.json.records.event_name",
                        "reason": "Final answer should submit the event name header.",
                    }
                ],
                "filters": [],
                "group_by": [],
                "metric_operation": "min",
                "metric_fields": ["json/expense.json.records.cost"],
                "row_policy": "preserve_all_ties",
                "distinct_policy": "preserve",
                "output_grain": "event",
                "row_source": "json/expense.json",
                "join_policy": "inner",
                "enrichment_fields": ["json/event.json.records.event_name"],
                "remaining_uncertainties": [],
            }
        ),
    ]


def _guided_final_response_without_tool_requests(response: str) -> str:
    payload = json.loads(response)
    payload["tool_requests"] = []
    return json.dumps(payload)


def test_guided_retry_prompt_forces_repair_hint_replacement() -> None:
    previous_error = (
        "Draft referenced fields outside whitelist: ['json/Examination.json.records.SEX']; "
        "repair_hints: [\"json/Examination.json.records.SEX is not available; "
        "use one of ['json/Patient.json.records.SEX'] if it matches the requested concept, "
        "and join from the filter table when needed.\"]"
    )

    prompt = build_guided_retry_prompt(
        phase="grounding",
        previous_error=previous_error,
        question="List patient sex for severe thrombosis cases.",
        allowed_field_refs=["json/Patient.json.records.SEX"],
        working_memory={"overview": None},
        tool_observations=[],
    )

    assert "choose exactly one replacement" in prompt
    assert "If there is only one candidate, use it" in prompt
    assert "Never repeat X anywhere" in prompt
    assert "Do not drop the concept" in prompt


def test_guided_phase_prompt_includes_full_global_data_profile() -> None:
    global_data_profile = "## Global Data Profile\n\n" + ("profile detail " * 700)
    prompt = build_guided_phase_prompt(
        phase="grounding",
        question="Which driver has number 44?",
        perception_payload={},
        context_bundle={},
        allowed_field_refs=["csv/qualifying.csv.number"],
        working_memory={"overview": None},
        tool_observations=[],
        previous_reasoning="r" * 7000,
        global_data_profile=global_data_profile,
    )
    payload = json.loads(prompt)

    assert payload["global_data_profile"] == global_data_profile
    assert len(payload["global_data_profile"]) > 6000
    assert payload["previous_reasoning"] == "r" * 6000


def _guided_patient_exam_responses() -> list[str]:
    return [
        json.dumps(
            {
                "task_intent": "Return Patient-level ID, SEX, and Diagnosis for patients with a matching exam record.",
                "concepts": [
                    {"term": "patients", "role": "answer_entity"},
                    {"term": "exam-positive examination record", "role": "filter"},
                ],
                "ambiguity_targets": ["Patient.Diagnosis versus Examination.Diagnosis"],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "grounded_concepts": [
                    {
                        "term": "patient output fields",
                        "role": "answer_entity",
                        "accepted_fields": [
                            {
                                "field_ref": "json/clinical.json.Patient.ID",
                                "confidence": "high",
                                "reason": "ID is requested from the Patient entity.",
                            },
                            {
                                "field_ref": "json/clinical.json.Patient.SEX",
                                "confidence": "high",
                                "reason": "SEX is requested from the Patient entity.",
                            },
                            {
                                "field_ref": "json/clinical.json.Patient.Diagnosis",
                                "confidence": "high",
                                "reason": "Diagnosis is requested as a Patient-level output attribute.",
                            },
                        ],
                        "rejected_fields": [],
                    },
                    {
                        "term": "exam-positive examination record",
                        "role": "filter",
                        "accepted_fields": [
                            {
                                "field_ref": "json/clinical.json.Examination.Diagnosis",
                                "confidence": "high",
                                "reason": "This field identifies the qualifying examination record.",
                            }
                        ],
                        "rejected_fields": [],
                    },
                ],
                "remaining_uncertainties": [],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "join_paths": [
                    {
                        "purpose": "Keep Patient rows that have a matching Examination row.",
                        "path": [
                            {
                                "from_field": "json/clinical.json.Patient.ID",
                                "to_field": "json/clinical.json.Examination.ID",
                            }
                        ],
                        "confidence": "high",
                    }
                ],
                "data_grain": "one row per qualifying Patient record",
                "relationship_risks": [],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "answer_columns": [
                    {
                        "name": "ID",
                        "source_field": "json/clinical.json.Patient.ID",
                        "reason": "Submit the Patient ID header exactly as requested.",
                    },
                    {
                        "name": "SEX",
                        "source_field": "json/clinical.json.Patient.SEX",
                        "reason": "Submit the Patient SEX header exactly as requested.",
                    },
                    {
                        "name": "Diagnosis",
                        "source_field": "json/clinical.json.Patient.Diagnosis",
                        "reason": "Use Patient Diagnosis for the requested output column.",
                    },
                ],
                "filters": ["json/clinical.json.Examination.Diagnosis = 'exam-positive'"],
                "group_by": [],
                "metric_operation": "lookup",
                "metric_fields": [],
                "row_policy": "multiple",
                "distinct_policy": "preserve",
                "output_grain": "Patient",
                "row_source": "json/clinical.json.Patient",
                "join_policy": "inner",
                "enrichment_fields": [],
                "remaining_uncertainties": [],
            }
        ),
    ]


def _guided_sat_frpm_responses() -> list[str]:
    return [
        json.dumps(
            {
                "task_intent": "Return SAT school rows and enrich them with charter funding type from FRPM.",
                "concepts": [
                    {"term": "sname", "role": "answer_entity"},
                    {"term": "Charter Funding Type", "role": "answer_entity"},
                    {"term": "Riverside-related school districts", "role": "filter"},
                    {"term": "available SAT math scores", "role": "filter"},
                ],
                "ambiguity_targets": ["District Name versus County Name", "SAT row source versus FRPM roster expansion"],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "grounded_concepts": [
                    {
                        "term": "sname",
                        "role": "answer_entity",
                        "accepted_fields": [
                            {
                                "field_ref": "db/satscores.db.satscores.sname",
                                "confidence": "high",
                                "reason": "The final header requested is sname from SAT school rows.",
                            }
                        ],
                        "rejected_fields": [
                            {
                                "field_ref": "csv/frpm.csv.School Name",
                                "reason": "FRPM school name is metadata for enrichment, not the requested SAT sname header.",
                            }
                        ],
                    },
                    {
                        "term": "Charter Funding Type",
                        "role": "answer_entity",
                        "accepted_fields": [
                            {
                                "field_ref": "csv/frpm.csv.Charter Funding Type",
                                "confidence": "high",
                                "reason": "This FRPM field supplies the requested funding attribute.",
                            }
                        ],
                        "rejected_fields": [],
                    },
                    {
                        "term": "Riverside-related school districts",
                        "role": "filter",
                        "accepted_fields": [
                            {
                                "field_ref": "csv/frpm.csv.District Name",
                                "confidence": "high",
                                "reason": "The phrase modifies school districts, so District Name is the filter level.",
                            }
                        ],
                        "rejected_fields": [
                            {
                                "field_ref": "csv/frpm.csv.County Name",
                                "reason": "County is a different geography level from school district.",
                            }
                        ],
                    },
                    {
                        "term": "available SAT math scores",
                        "role": "filter",
                        "accepted_fields": [
                            {
                                "field_ref": "db/satscores.db.satscores.AvgScrMath",
                                "confidence": "high",
                                "reason": "This field determines SAT math-score availability.",
                            },
                            {
                                "field_ref": "db/satscores.db.satscores.rtype",
                                "confidence": "high",
                                "reason": "School-level SAT rows have rtype = 'S'.",
                            },
                        ],
                        "rejected_fields": [],
                    },
                ],
                "remaining_uncertainties": [],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "join_paths": [
                    {
                        "purpose": "Join SAT school rows to FRPM attributes by CDS code.",
                        "path": [
                            {
                                "from_field": "db/satscores.db.satscores.cds",
                                "to_field": "csv/frpm.csv.CDSCode",
                            }
                        ],
                        "confidence": "high",
                    }
                ],
                "data_grain": "one row per qualifying SAT school record",
                "relationship_risks": ["Do not expand SAT rows to all FRPM schools in the district."],
                "tool_requests": [],
            }
        ),
        json.dumps(
            {
                "answer_columns": [
                    {
                        "name": "sname",
                        "source_field": "db/satscores.db.satscores.sname",
                        "reason": "Final output must keep the requested SAT column header.",
                    },
                    {
                        "name": "Charter Funding Type",
                        "source_field": "csv/frpm.csv.Charter Funding Type",
                        "reason": "This enrichment attribute is requested as the second output column.",
                    },
                ],
                "filters": [
                    "csv/frpm.csv.District Name contains Riverside",
                    "db/satscores.db.satscores.rtype = 'S'",
                    "db/satscores.db.satscores.sname IS NOT NULL",
                    "db/satscores.db.satscores.AvgScrMath IS NOT NULL",
                ],
                "group_by": [],
                "metric_operation": "lookup",
                "metric_fields": ["db/satscores.db.satscores.AvgScrMath"],
                "row_policy": "multiple",
                "distinct_policy": "preserve",
                "output_grain": "SAT school record",
                "row_source": "db/satscores.db.satscores",
                "join_policy": "inner",
                "enrichment_fields": ["csv/frpm.csv.Charter Funding Type"],
                "remaining_uncertainties": [],
            }
        ),
    ]


def test_data_understanding_agent_runs_staged_loop_and_promotes_fields_to_handoff(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    responses = _guided_cost_event_responses()
    model = SequenceSynthesisModel([responses[0], _guided_final_response_without_tool_requests(responses[0]), *responses[1:]])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
    assert any(step["phase"] == "overview" and step["accepted_draft"] for step in result.inspector_steps)
    assert result.handoff_status == "complete"
    handoff = result.data_understanding_handoff
    assert "columns" not in handoff["answer_contract"]
    assert handoff["answer_contract"]["answer_columns"][0]["name"] == "event_name"
    assert handoff["answer_contract"]["answer_columns"][0]["source_field"] == "json/event.json.records.event_name"
    assert handoff["answer_contract"]["metric_fields"] == ["json/expense.json.records.cost"]
    assert handoff["answer_contract"]["row_source"] == "json/expense.json"
    assert handoff["answer_contract"]["join_policy"] == "inner"
    assert handoff["answer_contract"]["row_policy"] == "preserve_all_ties"
    assert any(step["phase"] == "overview_tools" for step in result.inspector_steps)


def test_profile_guided_fast_path_requires_global_profile(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    model = SequenceSynthesisModel(_guided_cost_event_responses())
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception, global_data_profile="Global profiling failed: synthetic failure")

    assert model.call_count == 0
    assert result.handoff_status == "partial"
    assert any("global_data_profile is required" in error for error in result.validation_errors)
    assert any(step["phase"] == "profile_missing" for step in result.inspector_steps)
    assert not any(step["phase"] == "overview" for step in result.inspector_steps)


def test_profile_guided_fast_path_skips_overview_and_single_asset_fabric(tmp_path: Path) -> None:
    task = _create_task(tmp_path, "List the time values.")
    perception = _build_test_perception(
        task,
        entities=["time"],
        metrics=[],
        row_shape="multiple_rows",
        column_hint="time",
    )
    grounding = json.dumps(
        {
            "grounded_concepts": [
                {
                    "term": "time",
                    "role": "answer_entity",
                    "accepted_fields": [
                        {
                            "field_ref": "results.csv.time",
                            "confidence": "high",
                            "reason": "The requested output column is time from results.csv.",
                        }
                    ],
                    "rejected_fields": [],
                }
            ],
            "remaining_uncertainties": [],
            "tool_requests": [],
        }
    )
    contract = json.dumps(
        {
            "answer_columns": [
                {"name": "time", "source_field": "results.csv.time", "reason": "Return time values."}
            ],
            "filters": [],
            "group_by": [],
            "metric_operation": "lookup",
            "metric_fields": [],
            "row_policy": "multiple",
            "distinct_policy": "preserve",
            "output_grain": "results row",
            "row_source": "results.csv",
            "join_policy": "unknown",
            "enrichment_fields": [],
            "remaining_uncertainties": [],
        }
    )
    model = SequenceSynthesisModel([grounding, contract])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=4),
    )

    result = agent.run(task, perception, global_data_profile=TEST_GLOBAL_DATA_PROFILE)
    phases = [step["phase"] for step in result.inspector_steps]

    assert model.call_count == 2
    assert result.handoff_status == "complete"
    assert "overview_skipped" in phases
    assert "fabric_deterministic" in phases
    assert "overview" not in phases
    assert "fabric" not in phases


def test_profile_guided_fast_path_keeps_llm_fabric_for_cross_asset_ambiguity(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    responses = _guided_cost_event_responses()
    model = SequenceSynthesisModel(responses[1:])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception, global_data_profile=TEST_GLOBAL_DATA_PROFILE)
    phases = [step["phase"] for step in result.inspector_steps]

    assert model.call_count == 3
    assert result.handoff_status == "complete"
    assert "overview_skipped" in phases
    assert "fabric" in phases
    assert "fabric_deterministic" not in phases
    assert "overview" not in phases


def test_guided_loop_executes_all_requested_semantic_tools(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(task, entities=["event"], metrics=["cost"])
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    semantic_index = build_semantic_index(
        question=task.question,
        catalog=catalog,
        perception_payload=perception.content.payload,
    )
    query_tools = SemanticQueryTools(catalog=catalog, semantic_index=semantic_index, limit=5, )
    loop = GuidedDataUnderstandingLoop(model=None, query_tools=query_tools, max_steps=5, max_phase_retries=0)
    requests = [
        ToolRequest(tool="get_asset_schema", args={"asset_path": "json/expense.json"}),
        ToolRequest(tool="get_asset_schema", args={"asset_path": "json/event.json"}),
        ToolRequest(tool="lookup_knowledge", args={"term": "cost"}),
        ToolRequest(tool="search_semantic_index", args={"query": "event cost"}),
    ]
    steps: list[dict[str, object]] = []

    observations = loop._execute_tool_requests(requests, steps, "overview")

    assert len(observations) == 4
    assert [result["tool"] for result in observations] == [request.tool for request in requests]
    assert len(steps) == 1
    assert [request["tool"] for request in steps[0]["tool_requests"]] == [request.tool for request in requests]


def test_guided_loop_marks_failed_probe_observation_not_ok() -> None:
    query_tools = SemanticQueryTools(catalog={}, semantic_index={}, limit=5, )
    loop = GuidedDataUnderstandingLoop(model=None, query_tools=query_tools, max_steps=5, max_phase_retries=0)
    requests = [ToolRequest(tool="execute_probe_query", args={"sql": "SELECT 1"})]
    steps: list[dict[str, object]] = []

    observations = loop._execute_tool_requests(requests, steps, "overview")

    assert observations[0]["ok"] is False
    assert observations[0]["content"]["ok"] is False
    assert steps[0]["tool_results"][0]["ok"] is False


def test_compact_tool_result_returns_json_safe_date_values() -> None:
    compacted = _compact_tool_result(
        {
            "ok": True,
            "columns": ["date_received"],
            "rows": [[date(2019, 10, 17)]],
            "row_count": 1,
        }
    )

    assert compacted["rows"] == [["2019-10-17"]]
    assert compacted["row_count"] == 1
    json.dumps({"tool_observations": [{"content": compacted}]}, ensure_ascii=False)


def test_guided_final_phase_rejects_unexecuted_tool_requests(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception_payload = _build_test_perception(task, entities=["event"], metrics=["cost"]).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    semantic_index = build_semantic_index(
        question=task.question,
        catalog=catalog,
        perception_payload=perception_payload,
    )
    query_tools = SemanticQueryTools(
        catalog=catalog,
        semantic_index=semantic_index,
        limit=5,
        context_dir=task.context_dir,
    )
    probe = {
        "task_intent": "Find the event.",
        "concepts": [{"term": "event", "role": "answer_entity"}],
        "ambiguity_targets": ["Need field evidence."],
        "tool_requests": [{"tool": "search_semantic_index", "args": {"query": "event cost"}}],
    }
    bad_final = {**probe, "tool_requests": [{"tool": "lookup_knowledge", "args": {"term": "cost"}}]}
    good_final = {**probe, "tool_requests": []}
    model = SequenceSynthesisModel([json.dumps(probe), json.dumps(bad_final), json.dumps(good_final)])
    loop = GuidedDataUnderstandingLoop(model=model, query_tools=query_tools, max_steps=5, max_phase_retries=1)
    steps: list[dict[str, object]] = []

    draft = loop._run_phase_with_tool_refinement(
        phase="overview",
        draft_model=OverviewDraft,
        task=task,
        perception_payload=perception_payload,
        context_bundle=query_tools.build_context_bundle(task.question),
        field_whitelist=sorted(query_tools.known_field_refs()),
        working_memory={"overview": None},
        tool_observations=[],
        steps=steps,
    )

    assert model.call_count == 3
    assert draft.tool_requests == []
    assert any(
        step.get("validation_error") and "final phase must not include tool_requests" in str(step["validation_error"])
        for step in steps
    )


def test_guided_handoff_keeps_patient_output_columns_and_row_source(tmp_path: Path) -> None:
    task = _create_patient_exam_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["patients", "examination record"],
        metrics=[],
        filter_phrases=["with an exam-positive examination record"],
        row_shape="multiple_rows",
        column_hint="ID, SEX, Diagnosis",
        high_risk_terms=["field_semantics", "join_key"],
    )
    model = SequenceSynthesisModel(_guided_patient_exam_responses())
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert result.handoff_status == "complete"
    contract = result.data_understanding_handoff["answer_contract"]
    assert [column["name"] for column in contract["answer_columns"]] == ["ID", "SEX", "Diagnosis"]
    assert contract["answer_columns"][2]["source_field"] == "json/clinical.json.Patient.Diagnosis"
    assert contract["row_source"] == "json/clinical.json.Patient"
    assert contract["join_policy"] == "inner"
    assert "json/clinical.json.Examination.Diagnosis" in contract["filters"][0]


def test_guided_grounding_retry_repairs_single_candidate_field_hint(tmp_path: Path) -> None:
    task = _create_patient_exam_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["patients", "examination record"],
        metrics=[],
        filter_phrases=["with an exam-positive examination record"],
        row_shape="multiple_rows",
        column_hint="ID, SEX, Diagnosis",
        high_risk_terms=["field_semantics", "join_key"],
    )
    responses = _guided_patient_exam_responses()
    bad_grounding = json.loads(responses[1])
    bad_grounding["grounded_concepts"][0]["accepted_fields"][1]["field_ref"] = "json/clinical.json.Examination.SEX"
    model = SequenceSynthesisModel([responses[0], json.dumps(bad_grounding), *responses[1:]])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, max_phase_retries=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
    assert result.handoff_status == "complete"
    assert any(
        step["phase"] == "grounding" and step.get("validation_error") and "outside whitelist" in step["validation_error"]
        for step in result.inspector_steps
    )
    contract = result.data_understanding_handoff["answer_contract"]
    assert contract["answer_columns"][1]["source_field"] == "json/clinical.json.Patient.SEX"


def test_guided_handoff_uses_sat_rows_and_frpm_enrichment(tmp_path: Path) -> None:
    task = _create_sat_frpm_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["schools", "Riverside-related school districts"],
        metrics=["SAT math scores"],
        filter_phrases=["from Riverside-related school districts", "with available SAT math scores"],
        row_shape="multiple_rows",
        column_hint="sname and Charter Funding Type",
        high_risk_terms=["geographic_scope", "row_source", "join_key"],
    )
    model = SequenceSynthesisModel(_guided_sat_frpm_responses())
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert result.handoff_status == "complete"
    contract = result.data_understanding_handoff["answer_contract"]
    assert [column["name"] for column in contract["answer_columns"]] == ["sname", "Charter Funding Type"]
    assert contract["answer_columns"][0]["source_field"] == "db/satscores.db.satscores.sname"
    assert contract["row_source"] == "db/satscores.db.satscores"
    assert contract["join_policy"] == "inner"
    assert contract["enrichment_fields"] == ["csv/frpm.csv.Charter Funding Type"]
    assert any("rtype = 'S'" in filter_text for filter_text in contract["filters"])
    assert any("AvgScrMath IS NOT NULL" in filter_text for filter_text in contract["filters"])


def test_contract_allows_rejected_field_when_accepted_as_filter_support() -> None:
    field_whitelist = [
        "csv/yearmonth.csv.Consumption",
        "db/transactions_1k.db.transactions_1k.Price",
        "db/transactions_1k.db.transactions_1k.Amount",
    ]
    grounding = GroundingDraft(
        grounded_concepts=[
            {
                "term": "consumption status",
                "role": "metric",
                "accepted_fields": [
                    {
                        "field_ref": "csv/yearmonth.csv.Consumption",
                        "confidence": "high",
                        "reason": "Monthly consumption is the requested output metric.",
                    }
                ],
                "rejected_fields": [
                    {
                        "field_ref": "db/transactions_1k.db.transactions_1k.Amount",
                        "reason": "Amount is not the monthly consumption output metric.",
                    }
                ],
            },
            {
                "term": "paid more than 29.00 per unit",
                "role": "filter",
                "accepted_fields": [
                    {
                        "field_ref": "db/transactions_1k.db.transactions_1k.Price",
                        "confidence": "high",
                        "reason": "Price is the numerator for the per-unit filter.",
                    },
                    {
                        "field_ref": "db/transactions_1k.db.transactions_1k.Amount",
                        "confidence": "high",
                        "reason": "Amount is the denominator for the per-unit filter.",
                    },
                ],
                "rejected_fields": [],
            },
        ]
    )
    contract = ContractDraft(
        answer_columns=[
            {
                "name": "Consumption",
                "source_field": "csv/yearmonth.csv.Consumption",
                "reason": "Return the requested consumption value.",
            }
        ],
        filters=[
            "db/transactions_1k.db.transactions_1k.Price / db/transactions_1k.db.transactions_1k.Amount > 29"
        ],
        metric_operation="lookup",
        metric_fields=["csv/yearmonth.csv.Consumption"],
        row_policy="multiple",
        distinct_policy="deduplicate",
        output_grain="customer-month",
        row_source="csv/yearmonth.csv",
        join_policy="inner",
    )

    _validate_contract_draft(contract, grounding, field_whitelist)


def test_apply_repair_draft_preserves_top_level_uncertainties_in_contract_patch() -> None:
    country_field = "json/gasstations.json.records.Country"
    grounding = GroundingDraft(
        grounded_concepts=[
            {
                "term": "countries",
                "role": "answer_entity",
                "accepted_fields": [
                    {
                        "field_ref": country_field,
                        "confidence": "high",
                        "reason": "Country is the requested output.",
                    }
                ],
                "rejected_fields": [],
            }
        ]
    )
    repair = RepairDraft(
        contract_patch=ContractDraft(
            answer_columns=[
                {
                    "name": "country",
                    "source_field": country_field,
                    "reason": "Return country values.",
                }
            ],
            remaining_uncertainties=["Existing contract uncertainty."],
        ),
        remaining_uncertainties=["Probe query still failed."],
    )

    _, _, updated_contract = _apply_repair_draft(
        grounding,
        FabricDraft(),
        ContractDraft(),
        repair,
        [country_field],
    )

    assert updated_contract.remaining_uncertainties == [
        "Existing contract uncertainty.",
        "Probe query still failed.",
    ]


def test_contract_still_rejects_fields_never_accepted() -> None:
    field_whitelist = [
        "json/expense.json.records.cost",
        "csv/budget.csv.amount",
    ]
    grounding = GroundingDraft(
        grounded_concepts=[
            {
                "term": "cost",
                "role": "metric",
                "accepted_fields": [
                    {
                        "field_ref": "json/expense.json.records.cost",
                        "confidence": "high",
                        "reason": "Actual expense cost.",
                    }
                ],
                "rejected_fields": [
                    {
                        "field_ref": "csv/budget.csv.amount",
                        "reason": "Budgeted amount is not actual expense cost.",
                    }
                ],
            }
        ]
    )
    contract = ContractDraft(
        answer_columns=[{"name": "cost", "source_field": "json/expense.json.records.cost", "reason": "actual cost"}],
        metric_fields=["csv/budget.csv.amount"],
    )

    with pytest.raises(ValueError, match="Contract uses rejected fields"):
        _validate_contract_draft(contract, grounding, field_whitelist)


def test_data_understanding_agent_retries_phase_json_once(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    model = SequenceSynthesisModel(["not json", *_guided_cost_event_responses()])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, max_phase_retries=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
    assert result.handoff_status == "complete"
    assert any(step["phase"] == "overview" and step["validation_error"] for step in result.inspector_steps)


def test_data_understanding_agent_marks_contract_uncertainties_partial(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    responses = _guided_cost_event_responses()
    uncertain_contract = json.loads(responses[-1])
    uncertain_contract["remaining_uncertainties"] = ["Need to verify the filter condition against actual rows."]
    responses[-1] = json.dumps(uncertain_contract)
    model = SequenceSynthesisModel(responses[1:])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception, global_data_profile=TEST_GLOBAL_DATA_PROFILE)

    assert model.call_count == 3
    assert result.handoff_status == "partial"
    assert any("remaining_uncertainties" in error for error in result.validation_errors)
    assert any(step["phase"] == "overview_skipped" for step in result.inspector_steps)
    assert not any(step["phase"] == "overview" for step in result.inspector_steps)
    assert any(step["phase"] == "repair_skipped" for step in result.inspector_steps)


def test_data_understanding_agent_retries_phase_request_error_with_backoff(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = SequenceSynthesisModel([RuntimeError("temporary request failure"), *_guided_cost_event_responses()])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=4, max_phase_retries=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
    assert sleep_delays == [15]
    assert result.handoff_status == "complete"


def test_data_understanding_agent_rejects_unknown_field_and_falls_back(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    responses = _guided_cost_event_responses()
    bad_grounding = json.dumps(
        {
            "grounded_concepts": [
                {
                    "term": "cost",
                    "role": "metric",
                    "accepted_fields": [
                        {"field_ref": "json/expense.json.records.missing_cost", "confidence": "high", "reason": "bad"}
                    ],
                    "rejected_fields": [],
                }
            ],
            "remaining_uncertainties": [],
            "tool_requests": [],
        }
    )
    model = SequenceSynthesisModel(
        [responses[0], _guided_final_response_without_tool_requests(responses[0]), bad_grounding, bad_grounding]
    )
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, max_phase_retries=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 4
    assert result.handoff_status == "fallback"
    assert any("outside whitelist" in error for error in result.validation_errors)


def test_contract_failure_fallback_preserves_guided_grounding_and_fabric(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    responses = _guided_cost_event_responses()
    bad_contract = json.loads(responses[3])
    bad_contract["metric_fields"] = ["csv/budget.csv.amount"]
    bad_contract = json.dumps(bad_contract)
    model = SequenceSynthesisModel(
        [
            responses[0],
            _guided_final_response_without_tool_requests(responses[0]),
            responses[1],
            responses[2],
            bad_contract,
            bad_contract,
        ]
    )
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=6, max_phase_retries=1, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert result.handoff_status == "fallback"
    assert any("Contract uses rejected fields" in error for error in result.validation_errors)
    handoff = result.data_understanding_handoff
    assert any(concept["term"] == "cost" for concept in handoff["question_grounding"]["concepts"])
    assert handoff["data_fabric"]["join_paths"]
    assert handoff["answer_contract"]["answer_columns"] == []


def test_empty_guided_handoff_triggers_repair(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    empty_contract = json.dumps(
        {
            "answer_columns": [],
            "filters": [],
            "group_by": [],
            "metric_operation": "unknown",
            "metric_fields": [],
            "row_policy": "unknown",
            "distinct_policy": "unknown",
            "output_grain": "",
            "row_source": "",
            "join_policy": "unknown",
            "enrichment_fields": [],
            "remaining_uncertainties": [],
        }
    )
    repair = json.dumps(
        {
            "contract_patch": {
                "answer_columns": [
                    {
                        "name": "event_name",
                        "source_field": "json/event.json.records.event_name",
                        "reason": "Final answer should submit the event name header.",
                    }
                ],
                "filters": [],
                "group_by": [],
                "metric_operation": "min",
                "metric_fields": ["json/expense.json.records.cost"],
                "row_policy": "preserve_all_ties",
                "distinct_policy": "preserve",
                "output_grain": "event",
                "row_source": "json/expense.json",
                "join_policy": "inner",
                "enrichment_fields": ["json/event.json.records.event_name"],
                "remaining_uncertainties": [],
            },
            "remaining_uncertainties": [],
        }
    )
    responses = _guided_cost_event_responses()
    model = SequenceSynthesisModel(
        [responses[0], _guided_final_response_without_tool_requests(responses[0]), responses[1], responses[2], empty_contract, repair]
    )
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=6, profile_guided_fast_path=False),
    )

    result = agent.run(task, perception)

    assert model.call_count == 6
    assert result.handoff_status == "complete"
    assert result.data_understanding_handoff["answer_contract"]["answer_columns"][0]["name"] == "event_name"
    assert any(step["phase"] == "repair_or_critique" for step in result.inspector_steps)


def test_task_25_like_handoff_has_grounding_join_path_and_answer_contract(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    perception = _build_test_perception(
        task,
        entities=["event"],
        metrics=["lowest cost"],
        column_hint="event_name",
        high_risk_terms=["metric_operation_ambiguity"],
    )
    agent = DataUnderstandingAgent(
        model=None,
        config=DataInspectorConfig(mode="rules"),
    )

    result = agent.run(task, perception)
    handoff = result.data_understanding_handoff

    assert handoff["answer_contract"]["answer_columns"][0]["name"] == "event_name"
    assert handoff["answer_contract"]["answer_columns"][0]["source_field"] == "json/event.json.records.event_name"
    assert handoff["answer_contract"]["metric_operation"] == "min"
    assert handoff["answer_contract"]["row_policy"] == "preserve_all_ties"
    assert handoff["answer_contract"]["metric_field"] == "json/expense.json.records.cost"
    brief = handoff["brief_markdown"]
    assert "Do not use csv/budget.csv.amount" in brief
    assert "Do not use csv/budget.csv.spent" in brief
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
            data_inspector=DataInspectorConfig(mode="rules"),
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
            data_inspector=DataInspectorConfig(mode="rules"),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "global_data_exploration"
    assert first_step["ok"] is False
    assert "synthetic profiling failure" in first_step["tool_results"][0]["error"]


def test_langgraph_agent_injects_full_data_understanding_handoff(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content=_perception_response(
                    task,
                    entities=["event"],
                    metrics=["lowest cost"],
                    column_hint="event_name",
                    high_risk_terms=["metric_operation_ambiguity"],
                )
            ),
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
    # problem_grounding is the 2nd step (after global_data_exploration)
    grounding_step = result.steps[1].to_dict()
    assert grounding_step["node"] == "problem_grounding"
    first_request = model.invocations[-1]
    injected_messages = [getattr(message, "content", "") for message in first_request]
    injected_text = "\n".join(str(content) for content in injected_messages)
    assert "Full data_understanding_handoff.json" in injected_text
    assert '"question_grounding"' in injected_text
    assert '"data_fabric"' in injected_text
    assert '"answer_contract"' in injected_text
    assert '"answer_columns"' in injected_text
    assert '"row_source"' in injected_text
    assert '"metric_field": "json/expense.json.records.cost"' in injected_text
    assert "trusted guidance" in injected_text
    assert "hypotheses from a separate data understanding agent" not in injected_text
    assert "Verify important claims" not in injected_text


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


def test_rule_based_global_profile_is_self_contained(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    profile = DataUnderstandingAgent(
        model=None,
        config=DataInspectorConfig(
            sample_budget=DataInspectorSampleBudget(catalog_sample_rows=1, max_doc_chars=20, max_json_chars=20)
        ),
    ).explore_data_globally(context_dir=task.context_dir, task_id=task.task_id)

    assert profile.startswith("## Global Data Profile")
    assert "### Assets" in profile
    assert "### Schemas" in profile
    assert "### Knowledge Documents" in profile
    assert "- Headings:" in profile


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
            },
            {
                "asset_path": "doc/background.md",
                "content": "x" * 5000,
                "char_count": 5000,
            },
        ],
    )
    payload = json.loads(prompt)
    knowledge_doc, background_doc = payload["knowledge_documents"]

    assert knowledge_doc["asset_path"] == "knowledge.md"
    assert knowledge_doc["content"] == full_knowledge
    assert knowledge_doc["is_full_content"] is True
    assert background_doc["content"] == "x" * 5000
    assert background_doc["is_full_content"] is True


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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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
                "sample_rows": [["1", "VYBER", "100"]],
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
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

    budget = DataInspectorSampleBudget(catalog_sample_rows=2)
    schema = _read_json_schema(json_path, "data.json", budget)

    score_field = next(f for f in schema["fields"] if f["name"] == "score")
    assert score_field["min_value"] == 72.0
    assert score_field["max_value"] == 95.5

    grade_field = next(f for f in schema["fields"] if f["name"] == "grade")
    assert "min_value" not in grade_field


def test_rule_based_profile_includes_row_count_and_min_max() -> None:
    from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent

    catalog: dict[str, Any] = {
        "task_id": "task_demo",
        "assets": [],
        "schemas": [
            {
                "asset_path": "trans.csv",
                "kind": "csv",
                "row_count": 1000000,
                "fields": [
                    {
                        "name": "amount",
                        "type": "number",
                        "missing_count": 0,
                        "cardinality": 999,
                        "distinct_values": ["100", "200", "300"],
                        "min_value": 10.0,
                        "max_value": 50000.0,
                    },
                    {
                        "name": "date",
                        "type": "string",
                        "missing_count": 0,
                        "cardinality": 365,
                        "distinct_values": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    },
                ],
                "sample_rows": [],
            },
            {
                "asset_path": "lookup.db",
                "kind": "sqlite",
                "tables": [
                    {
                        "name": "status_codes",
                        "row_count": 10,
                        "fields": [
                            {
                                "name": "code",
                                "type": "string",
                                "missing_count": 0,
                                "cardinality": 3,
                                "distinct_values": ["A", "B", "C"],
                            },
                        ],
                        "sample_rows": [["A"], ["B"]],
                    },
                ],
            },
        ],
        "relationships": [],
        "semantic_uncertainties": [],
    }
    profile = DataUnderstandingAgent._build_rule_based_profile(catalog, knowledge_docs=[])

    assert "1000000 rows" in profile
    assert "10 rows" in profile
    assert "range=[10.0, 50000.0]" in profile
    assert "distinct_values=[\"A\", \"B\", \"C\"]" in profile
