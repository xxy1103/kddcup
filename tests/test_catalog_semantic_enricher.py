from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.config import load_app_config
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.inspectors.catalog_semantic_enricher import (
    _build_knowledge_preview,
    _count_fields,
    _merge_enrichments,
    _try_parse_json,
    _validate_enriched_catalog,
    enrich_lightweight_catalog_with_knowledge,
)
from data_agent_baseline.inspectors.prompts import build_lightweight_catalog


def _create_test_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "results.csv").write_text(
        "raceId,driverId,rank,position,time\n1,10,2,1,1:30.000\n",
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text("# Notes\nrank differs from position\n", encoding="utf-8")
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'Chinese Grand Prix')")
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question="What's the finish time?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _make_lightweight_catalog() -> dict:
    catalog = {
        "task_id": "test_task",
        "assets": [
            {"asset_path": "data.csv", "kind": "csv"},
            {"asset_path": "sample.db", "kind": "sqlite"},
        ],
        "schemas": [
            {
                "asset_path": "data.csv",
                "kind": "csv",
                "fields": [
                    {"name": "raceId", "type": "INTEGER"},
                    {"name": "driverId", "type": "INTEGER"},
                    {"name": "position", "type": "INTEGER"},
                ],
            },
            {
                "asset_path": "sample.db",
                "kind": "sqlite",
                "tables": [
                    {
                        "name": "results",
                        "fields": [
                            {"name": "raceId", "type": "INTEGER"},
                            {"name": "driverId", "type": "INTEGER"},
                        ],
                    }
                ],
            },
        ],
        "relationships": [
            {
                "from": "races.raceId",
                "to": "results.raceId",
                "type": "foreign_key",
                "cardinality": "1:N",
                "confidence": 0.95,
            }
        ],
        "knowledge_documents": [
            {
                "asset_path": "knowledge.md",
                "content": "raceId: unique race identifier.\ndriverId: unique driver identifier.",
                "token_count": 20,
                "is_full_content": True,
            }
        ],
    }
    return catalog


def _make_enriched_response(original: dict) -> dict:
    enriched = json.loads(json.dumps(original, ensure_ascii=False))
    for schema in enriched["schemas"]:
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                for field in table["fields"]:
                    field["description"] = f"Description for {field['name']}"
        else:
            for field in schema.get("fields", []):
                field["description"] = f"Description for {field['name']}"
    return enriched


# ── _count_fields ──────────────────────────────────────────────


def test_count_fields_flat_schemas() -> None:
    schemas = [
        {"kind": "csv", "fields": [{"name": "a"}, {"name": "b"}]},
        {"kind": "json", "fields": [{"name": "c"}]},
    ]
    assert _count_fields(schemas) == 3


def test_count_fields_sqlite() -> None:
    schemas = [
        {
            "kind": "sqlite",
            "tables": [
                {"name": "t1", "fields": [{"name": "a"}, {"name": "b"}]},
                {"name": "t2", "fields": [{"name": "c"}]},
            ],
        }
    ]
    assert _count_fields(schemas) == 3


def test_count_fields_mixed() -> None:
    schemas = [
        {"kind": "csv", "fields": [{"name": "x"}]},
        {
            "kind": "sqlite",
            "tables": [
                {"name": "t1", "fields": [{"name": "a"}, {"name": "b"}]},
            ],
        },
    ]
    assert _count_fields(schemas) == 3


# ── _try_parse_json ────────────────────────────────────────────


def test_try_parse_json_valid() -> None:
    assert _try_parse_json('{"a": 1}') == {"a": 1}


def test_try_parse_json_with_fences() -> None:
    assert _try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_try_parse_json_invalid() -> None:
    assert _try_parse_json("not json") is None


def test_try_parse_json_array_not_dict() -> None:
    assert _try_parse_json("[1, 2, 3]") is None


# ── _build_knowledge_preview ───────────────────────────────────


def test_build_knowledge_preview() -> None:
    docs = [
        {"asset_path": "a.md", "content": "hello"},
        {"asset_path": "b.md", "content": "world"},
    ]
    result = _build_knowledge_preview(docs)
    assert "### a.md" in result
    assert "hello" in result
    assert "### b.md" in result
    assert "world" in result


# ── _validate_enriched_catalog ─────────────────────────────────


def test_validate_accepts_valid_enrichment() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    assert _validate_enriched_catalog(original, enriched) == []


def test_validate_rejects_missing_top_level_key() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    del enriched["schemas"]
    errors = _validate_enriched_catalog(original, enriched)
    assert any("schemas" in e for e in errors)


def test_validate_rejects_wrong_schema_count() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"].pop()
    errors = _validate_enriched_catalog(original, enriched)
    assert any("Schema count" in e for e in errors)


def test_validate_rejects_wrong_field_count() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"].pop()
    errors = _validate_enriched_catalog(original, enriched)
    assert any("field count mismatch" in e for e in errors)


def test_validate_rejects_changed_field_name() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["name"] = "changed"
    errors = _validate_enriched_catalog(original, enriched)
    assert any("name changed" in e for e in errors)


def test_validate_rejects_changed_field_type() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["type"] = "TEXT"
    errors = _validate_enriched_catalog(original, enriched)
    assert any("type changed" in e for e in errors)


def test_validate_rejects_unexpected_field_key() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["semantic_role"] = "id"
    errors = _validate_enriched_catalog(original, enriched)
    assert any("unexpected keys" in e for e in errors)


def test_validate_rejects_empty_description() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["description"] = ""
    errors = _validate_enriched_catalog(original, enriched)
    assert any("description is missing or empty" in e for e in errors)


def test_validate_rejects_missing_description() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    del enriched["schemas"][0]["fields"][0]["description"]
    errors = _validate_enriched_catalog(original, enriched)
    assert any("description is missing or empty" in e for e in errors)


def test_validate_rejects_non_string_note() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["note"] = 123
    errors = _validate_enriched_catalog(original, enriched)
    assert any("note must be a string" in e for e in errors)


def test_validate_accepts_note_omitted() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    # _make_enriched_response adds description only, no note — this is valid
    assert _validate_enriched_catalog(original, enriched) == []


def test_validate_rejects_wrong_sqlite_table_count() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][1]["tables"].pop()
    errors = _validate_enriched_catalog(original, enriched)
    assert any("table count mismatch" in e for e in errors)


def test_validate_rejects_wrong_sqlite_field_count() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][1]["tables"][0]["fields"].pop()
    errors = _validate_enriched_catalog(original, enriched)
    assert any("field count mismatch" in e for e in errors)


# ── _merge_enrichments ─────────────────────────────────────────


def test_merge_preserves_original_structure() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    merged = _merge_enrichments(original, enriched)
    assert merged["task_id"] == original["task_id"]
    assert len(merged["schemas"]) == len(original["schemas"])
    assert merged["relationships"] == original["relationships"]


def test_merge_adds_descriptions_to_flat_fields() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    merged = _merge_enrichments(original, enriched)
    field = merged["schemas"][0]["fields"][0]
    assert field["description"] == "Description for raceId"


def test_merge_adds_descriptions_to_sqlite_fields() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    merged = _merge_enrichments(original, enriched)
    field = merged["schemas"][1]["tables"][0]["fields"][0]
    assert field["description"] == "Description for raceId"


def test_merge_adds_note_when_present() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["note"] = "Primary key"
    merged = _merge_enrichments(original, enriched)
    assert merged["schemas"][0]["fields"][0]["note"] == "Primary key"


def test_merge_does_not_add_empty_note() -> None:
    original = _make_lightweight_catalog()
    enriched = _make_enriched_response(original)
    enriched["schemas"][0]["fields"][0]["note"] = "   "
    merged = _merge_enrichments(original, enriched)
    assert "note" not in merged["schemas"][0]["fields"][0]


# ── enrich_lightweight_catalog_with_knowledge integration ──────


def test_enrich_returns_original_on_no_fields() -> None:
    class DummyModel:
        def invoke(self, messages):  # noqa: ANN001
            del messages
            raise RuntimeError("should not be called")

    original = json.dumps({"task_id": "t", "assets": [], "schemas": [], "relationships": [], "knowledge_documents": []})
    result = enrich_lightweight_catalog_with_knowledge(
        model=DummyModel(),
        lightweight_catalog=original,
    )
    assert result == original


def test_enrich_returns_original_on_llm_failure(tmp_path: Path) -> None:
    from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
    from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget

    task = _create_test_task(tmp_path)
    agent = DataUnderstandingAgent(
        DataInspectorConfig(sample_budget=DataInspectorSampleBudget(max_doc_tokens=500))
    )
    lightweight, catalog = agent.explore_data_globally(
        context_dir=task.context_dir, task_id=task.task_id
    )

    class FailingModel:
        def invoke(self, messages):  # noqa: ANN001
            del messages
            raise RuntimeError("simulated failure")

    result = enrich_lightweight_catalog_with_knowledge(
        model=FailingModel(),
        lightweight_catalog=lightweight,
    )
    assert result == lightweight


def test_enrich_returns_original_on_non_json_response(tmp_path: Path) -> None:
    from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
    from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget
    from langchain_core.messages import AIMessage

    task = _create_test_task(tmp_path)
    agent = DataUnderstandingAgent(
        DataInspectorConfig(sample_budget=DataInspectorSampleBudget(max_doc_tokens=500))
    )
    lightweight, catalog = agent.explore_data_globally(
        context_dir=task.context_dir, task_id=task.task_id
    )

    class GarbageModel:
        def invoke(self, messages):  # noqa: ANN001
            del messages
            return AIMessage(content="not json at all")

    result = enrich_lightweight_catalog_with_knowledge(
        model=GarbageModel(),
        lightweight_catalog=lightweight,
    )
    assert result == lightweight


def test_enrich_success(tmp_path: Path) -> None:
    from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
    from data_agent_baseline.config import DataInspectorConfig, DataInspectorSampleBudget
    from langchain_core.messages import AIMessage

    task = _create_test_task(tmp_path)
    agent = DataUnderstandingAgent(
        DataInspectorConfig(sample_budget=DataInspectorSampleBudget(max_doc_tokens=500))
    )
    lightweight, catalog = agent.explore_data_globally(
        context_dir=task.context_dir, task_id=task.task_id
    )

    original_parsed = json.loads(lightweight)
    enriched = json.loads(json.dumps(original_parsed, ensure_ascii=False))

    def enrich_fields(schemas):
        for schema in schemas:
            if schema.get("kind") == "sqlite":
                for table in schema.get("tables", []):
                    for field in table["fields"]:
                        field["description"] = f"Business meaning of {field['name']}"
            else:
                for field in schema.get("fields", []):
                    field["description"] = f"Business meaning of {field['name']}"

    enrich_fields(enriched["schemas"])

    class GoodModel:
        def invoke(self, messages):  # noqa: ANN001
            del messages
            return AIMessage(content=json.dumps(enriched, ensure_ascii=False))

    result = enrich_lightweight_catalog_with_knowledge(
        model=GoodModel(),
        lightweight_catalog=lightweight,
    )
    assert result != lightweight
    result_parsed = json.loads(result)
    first_field = result_parsed["schemas"][0]["fields"][0]
    assert "description" in first_field
    assert first_field["name"] == original_parsed["schemas"][0]["fields"][0]["name"]


# ── Config parse ───────────────────────────────────────────────


def test_config_default_disables_enrichment() -> None:
    from data_agent_baseline.config import DataInspectorConfig
    assert DataInspectorConfig().enable_semantic_enrichment is False


def test_config_parse_enable_semantic_enrichment(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data_inspector:\n  enable_semantic_enrichment: true\n",
        encoding="utf-8",
    )
    cfg = load_app_config(config_path)
    assert cfg.data_inspector.enable_semantic_enrichment is True


def test_config_parse_explicit_false(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data_inspector:\n  enable_semantic_enrichment: false\n",
        encoding="utf-8",
    )
    cfg = load_app_config(config_path)
    assert cfg.data_inspector.enable_semantic_enrichment is False
