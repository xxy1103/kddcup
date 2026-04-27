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
from data_agent_baseline.inspectors.perception import PerceptionBuildError, build_perception_envelope
from data_agent_baseline.inspectors.prompts import build_guided_phase_prompt
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
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("No scripted synthesis responses remaining.")
        return AIMessage(content=self.responses.pop(0))


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
    assert race_id_field["sample_values"] == ["1"]
    assert name_field["sample_values"] == ["Chinese Grand Prix"]
    assert races_table["sample_rows"] == [["1", "Chinese Grand Prix"]]
    assert any(item["asset_path"] == "bad.json" for item in catalog["semantic_uncertainties"])
    assert any(item["asset_path"] == "broken.db" for item in catalog["semantic_uncertainties"])
    assert any(item["field"] == "raceid" for item in catalog["relationships"])


def test_semantic_catalog_classifies_document_knowledge_items(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text(
        "# Knowledge Guide\n\n"
        "This guide introduces the patient dataset.\n\n"
        "## Core Entities & Fields\n\n"
        "### Patient\n"
        "- **Diagnosis (text):** The disease diagnosed in the patient.\n\n"
        "## Constraints & Conventions\n\n"
        "- Thrombosis = 2 indicates severe cases.\n\n"
        "## Exemplar Use Cases\n\n"
        "### Use Case 2: Identify Patients with Severe Thrombosis\n"
        "```sql\n"
        "SELECT DISTINCT ID, SEX, Diagnosis FROM Examination WHERE Thrombosis = 2\n"
        "```\n",
        encoding="utf-8",
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    document_schema = next(schema for schema in catalog["schemas"] if schema["kind"] == "document")
    knowledge_items = document_schema["knowledge_items"]
    assert any(
        item["evidence_type"] == "schema_definition"
        and "Diagnosis" in item["text"]
        and item["section_path"] == ["Knowledge Guide", "Core Entities & Fields", "Patient"]
        for item in knowledge_items
    )
    assert any(
        item["evidence_type"] == "business_rule" and "Thrombosis = 2 indicates severe cases" in item["text"]
        for item in knowledge_items
    )
    assert any(
        item["evidence_type"] == "exemplar_sql" and "SELECT DISTINCT ID" in item["text"]
        for item in knowledge_items
    )
    assert any(
        item["evidence_type"] == "free_text_note" and "introduces the patient dataset" in item["text"]
        for item in knowledge_items
    )


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
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, max_join_hops=3)

    knowledge_hits = tools.lookup_knowledge("severe thrombosis")

    assert any("Thrombosis = 2" in item["snippet"] for item in knowledge_hits)


def test_lookup_knowledge_returns_typed_hits_from_knowledge_items(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text(
        "# Knowledge Guide\n\n"
        "## Core Entities & Fields\n\n"
        "### Patient\n"
        "- **Diagnosis (text):** The disease diagnosed in the patient.\n\n"
        "## Constraints & Conventions\n\n"
        "- Thrombosis = 2 indicates severe thrombosis cases.\n\n"
        "## Exemplar Use Cases\n\n"
        "### Use Case 2: Identify Patients with Severe Thrombosis\n"
        "```sql\n"
        "SELECT DISTINCT ID, SEX, Diagnosis FROM Examination WHERE Thrombosis = 2\n"
        "```\n",
        encoding="utf-8",
    )
    perception = _build_test_perception(
        task,
        metrics=["diagnosis"],
        filter_phrases=["severe thrombosis"],
        column_hint="Diagnosis",
        high_risk_terms=["entity_level_ambiguity"],
    ).content.payload
    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    index = build_semantic_index(question=task.question, catalog=catalog, perception_payload=perception)
    tools = SemanticQueryTools(catalog=catalog, semantic_index=index, limit=5, max_join_hops=3)

    diagnosis_hits = tools.lookup_knowledge("diagnosis")
    severe_hits = tools.lookup_knowledge("severe thrombosis")

    assert any(
        hit["evidence_type"] == "schema_definition"
        and hit["section_path"] == ["Knowledge Guide", "Core Entities & Fields", "Patient"]
        and "disease diagnosed in the patient" in hit["snippet"]
        for hit in diagnosis_hits
    )
    assert any(hit.get("evidence_type") in {"business_rule", "exemplar_sql"} for hit in severe_hits)
    assert any("Thrombosis = 2" in hit["snippet"] for hit in severe_hits)


def test_guided_phase_prompt_includes_knowledge_evidence_policy() -> None:
    context_bundle = {
        "query": "List patient diagnoses.",
        "knowledge_hits": [
            {
                "asset_path": "knowledge.md",
                "line_start": 10,
                "line_end": 10,
                "section_path": ["Knowledge Guide", "Core Entities & Fields", "Patient"],
                "evidence_type": "schema_definition",
                "snippet": "Diagnosis: The disease diagnosed in the patient.",
            }
        ],
    }

    prompt = build_guided_phase_prompt(
        phase="grounding",
        question="List patient diagnoses.",
        perception_payload={},
        context_bundle=context_bundle,
        allowed_field_refs=["json/Patient.json.records.Diagnosis"],
        working_memory={},
        tool_observations=[],
    )
    payload = json.loads(prompt)

    assert "knowledge_evidence_policy" in payload
    assert "exemplar_sql knowledge" in payload["knowledge_evidence_policy"]["evidence_priority"]
    assert payload["initial_context_bundle"]["knowledge_hits"][0]["evidence_type"] == "schema_definition"


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
    perception = _build_test_perception(
        task,
        metrics=["finish time"],
        filter_phrases=["ranked second"],
        column_hint="finish time",
        high_risk_terms=["rank_position_ambiguity"],
    )
    agent = DataUnderstandingAgent(
        model=InvalidJsonSynthesisModel(),
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=1),
    )

    result = agent.run(task, perception)

    assert result.synthesis is None
    assert result.synthesis_error is not None
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
                    {
                        "tool": "find_join_paths",
                        "args": {
                            "source": "json/expense.json.records.cost",
                            "target": "json/event.json.records.event_name",
                        },
                    },
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
                "row_filters": [],
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
                "row_filters": ["Patient row has a matching Examination row with Diagnosis = 'exam-positive'"],
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
                "filters": ["csv/frpm.csv.District Name contains Riverside"],
                "row_filters": [
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
    model = SequenceSynthesisModel(_guided_cost_event_responses())
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception)

    assert model.call_count == 4
    assert result.synthesis is not None
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
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception)

    assert result.handoff_status == "complete"
    contract = result.data_understanding_handoff["answer_contract"]
    assert [column["name"] for column in contract["answer_columns"]] == ["ID", "SEX", "Diagnosis"]
    assert contract["answer_columns"][2]["source_field"] == "json/clinical.json.Patient.Diagnosis"
    assert contract["row_source"] == "json/clinical.json.Patient"
    assert contract["join_policy"] == "inner"
    assert "json/clinical.json.Examination.Diagnosis" in contract["filters"][0]


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
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception)

    assert result.handoff_status == "complete"
    contract = result.data_understanding_handoff["answer_contract"]
    assert [column["name"] for column in contract["answer_columns"]] == ["sname", "Charter Funding Type"]
    assert contract["answer_columns"][0]["source_field"] == "db/satscores.db.satscores.sname"
    assert contract["row_source"] == "db/satscores.db.satscores"
    assert contract["join_policy"] == "inner"
    assert contract["enrichment_fields"] == ["csv/frpm.csv.Charter Funding Type"]
    assert any("rtype = 'S'" in row_filter for row_filter in contract["row_filters"])
    assert any("AvgScrMath IS NOT NULL" in row_filter for row_filter in contract["row_filters"])


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
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, max_phase_retries=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
    assert result.handoff_status == "complete"
    assert any(step["phase"] == "overview" and step["validation_error"] for step in result.inspector_steps)


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
    model = SequenceSynthesisModel([responses[0], bad_grounding, bad_grounding])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5, max_phase_retries=1),
    )

    result = agent.run(task, perception)

    assert model.call_count == 3
    assert result.handoff_status == "fallback"
    assert result.synthesis is None
    assert "outside whitelist" in str(result.synthesis_error)


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
            "row_filters": [],
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
                "row_filters": [],
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
    model = SequenceSynthesisModel([responses[0], responses[1], responses[2], empty_contract, repair])
    agent = DataUnderstandingAgent(
        model=model,
        config=DataInspectorConfig(mode="hybrid", max_agent_steps=5),
    )

    result = agent.run(task, perception)

    assert model.call_count == 5
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

    def fail_perception(task, model):  # noqa: ANN001
        del task, model
        raise RuntimeError("synthetic inspector failure")

    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.invoke_perception_agent", fail_perception)
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2, enable_data_inspector=True),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.inspector is not None
    assert "Perception failed" in result.inspector["error"]


def test_langgraph_agent_records_perception_retry_failure_and_continues(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="not json"),
            AIMessage(content="still not json"),
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
        config=LangGraphAgentConfig(max_steps=2, enable_data_inspector=True),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.inspector is not None
    assert "Perception failed" in result.inspector["error"]
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "perceive_task"
    assert first_step["ok"] is False
    assert len(first_step["model_response"]["attempts"]) == 2


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
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "perceive_task"
    assert first_step["model_request"] is not None
    assert first_step["model_response"] is not None
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
            "perception": {"ok": True},
            "semantic_catalog": {"assets": []},
            "semantic_index": {"mode": "keyword"},
            "data_understanding_handoff": {"brief_markdown": "ok"},
            "handoff_status": "complete",
            "validation_errors": [],
            "inspector_steps": [{"phase": "deterministic_context"}],
        },
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    assert (run_output_dir / "task_demo" / "perception.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_catalog.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_index.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_handoff.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_trace.json").exists()
