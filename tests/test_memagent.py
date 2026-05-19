from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.memagent_etl import (
    default_store_id,
    parse_markdown_blocks,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRuntimeContext, create_default_tool_registry


def _etl_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_etl"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    return PublicTask(
        record=TaskRecord(task_id="task_etl", difficulty="easy", question="Extract docs."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class FakeRuleModel:
    def invoke(self, prompt: str):  # noqa: ANN001
        if '"table": "Patient"' in prompt:
            payload = {
                "table": "Patient",
                "primary_key": ["ID"],
                "columns": ["ID", "SEX", "Birthday", "Description", "First_Date", "Admission", "Diagnosis"],
                "key_patterns": [r"\bPatient ID\s*(\d+)"],
                "field_patterns": {
                    "SEX": [r"\bSEX:\s*([A-Za-z]+)"],
                    "Birthday": [r"\bBirthday:\s*([0-9-]+)"],
                    "Diagnosis": [r"\bDiagnosis:\s*([^.;\n]+)"],
                    "First_Date": [r"\bFirst visit:\s*(None|NaN|[0-9-]+)"],
                },
                "field_types": {"ID": "int"},
            }
        elif '"table": "Laboratory"' in prompt:
            payload = {
                "table": "Laboratory",
                "primary_key": ["ID", "Date"],
                "columns": ["ID", "Date", "GOT", "GPT", "LDH", "ALP", "TBIL", "TP", "ALB", "UA", "UN", "CRE"],
                "key_patterns": [r"\bPatient ID\s*(\d+)"],
                "date_patterns": [r"\b((?:20\d{2}|19\d{2})-\d{1,2}-\d{1,2})\b"],
                "field_patterns": {
                    "GOT": [r"\bGOT:\s*(None|NaN|-?\d+(?:\.\d+)?)"],
                    "CRE": [
                        r"\bCRE\b.*?\bcorrected\b.{0,30}?\bto\s+(None|NaN|-?\d+(?:\.\d+)?)\b",
                        r"\bCRE:\s*(None|NaN|-?\d+(?:\.\d+)?)",
                        r"\bCRE\s+initially\s+(None|NaN|-?\d+(?:\.\d+)?)",
                    ],
                },
                "field_types": {"ID": "int", "GOT": "float", "CRE": "float"},
            }
        elif '"table": "superhero"' in prompt:
            payload = {
                "table": "Superhero",
                "primary_key": ["id"],
                "columns": [
                    "id",
                    "superhero_name",
                    "full_name",
                    "gender_id",
                    "eye_colour_id",
                    "hair_colour_id",
                    "skin_colour_id",
                    "race_id",
                    "publisher_id",
                    "alignment_id",
                    "height_cm",
                    "weight_kg",
                ],
                "key_patterns": [
                    r"\bEntry\s*(\d+)",
                    r"\bcataloged with reference code\s*(\d+)",
                    r"\btracked with identifier\s*(\d+)",
                    r"\bregistered at ID\s*(\d+)",
                    r"\bregistered under ID\s*(\d+)",
                    r"\bregistered at ID\s*(\d+)",
                ],
                "field_patterns": {
                    "superhero_name": [r"\bcodename:\s*([^.;,\n]+)"],
                    "full_name": [r"\bfull name:\s*([^.;,\n]+)"],
                    "height_cm": [
                        r"\bcorrected to\s+(None|NaN|-?\d+(?:\.\d+)?)\s*centimeters",
                        r"\bheight\b[^.]{0,120}?\b(None|NaN|-?\d+(?:\.\d+)?)\s*centimeters",
                        r"\bheight\b[^.]{0,80}?as\s+(None|NaN|-?\d+(?:\.\d+)?)\b",
                    ],
                    "weight_kg": [
                        r"\bweight\b[^.]{0,120}?\b(None|NaN|-?\d+(?:\.\d+)?)\s*(?:kilograms|kg)",
                        r"\bweight\b[^.]{0,80}?as\s+(None|NaN|-?\d+(?:\.\d+)?)\b",
                    ],
                    "publisher_id": [
                        r"\bpublisher affiliation (?:is |is logged as |is logged with the code |is recorded as |is recorded )?(\d+)",
                        r"\bregistered with publisher\s*(\d+)",
                    ],
                    "gender_id": [r"\bgender id is\s*(\d+)"],
                },
                "paired_fields": [
                    {
                        "fields": ["height_cm", "weight_kg"],
                        "pattern": r"placeholder data of\s+(None|NaN|-?\d+(?:\.\d+)?)\s+for both height and weight",
                    }
                ],
                "field_types": {
                    "id": "int",
                    "height_cm": "float",
                    "weight_kg": "float",
                    "publisher_id": "int",
                    "gender_id": "int",
                },
            }
        else:
            payload = {
                "table": "custom",
                "primary_key": ["id"],
                "columns": ["id", "amount"],
                "key_patterns": [r"Record key (\d+)"],
                "field_patterns": {"amount": [r"amount is (\d+(?:\.\d+)?)"]},
                "field_types": {"id": "int", "amount": "float"},
            }
        return type("Msg", (), {"content": json.dumps(payload)})()


def test_markdown_block_parser_tracks_headings_lines_and_phase() -> None:
    text = (
        "# Patient Facts\n"
        "Patient ID 1. SEX: F.\n\n"
        "## Corrected Labs\n"
        "Patient ID 1 date 2020-01-01 CRE corrected to 1.2.\n"
    )

    blocks = parse_markdown_blocks(text)

    assert len(blocks) == 2
    assert blocks[0].heading == "Patient Facts"
    assert blocks[0].start_line == 2
    assert blocks[0].end_line == 2
    assert blocks[1].heading == "Corrected Labs"
    assert blocks[1].section == "Patient Facts"
    assert blocks[1].phase == "Corrected Labs"


def test_default_store_id_is_stable_and_uses_doc_stems() -> None:
    left = default_store_id("task_1", ["doc/Patient.md", "doc/Laboratory.md"])
    right = default_store_id("task_1", ["doc/Patient.md", "doc/Laboratory.md"])

    assert left == right
    assert left.startswith("task_1_Patient_Laboratory_")


def test_memagent_is_registered_as_etl_tool_only() -> None:
    registry = create_default_tool_registry()

    assert "memagent" in registry.specs
    assert "query_memagent_sql" in registry.specs
    assert "list_memagent_tables" in registry.specs
    assert "read_memagent_unresolved" in registry.specs
    assert "SQLite ETL" in registry.specs["memagent"].description
    assert "regex/Python extraction patterns" not in registry.specs["memagent"].description


def test_memagent_rejects_removed_pattern_mode(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "Patient.md").write_text("Patient ID 1001. SEX: F.", encoding="utf-8")
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "memagent",
        {"path": "Patient.md", "mode": "pattern"},
    )

    assert result.ok is False
    assert "legacy pattern analyzer has been removed" in result.content["error"]


def test_memagent_extract_tables_requires_model(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "Patient.md").write_text("Patient ID 1001. SEX: F.", encoding="utf-8")
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(runtime_context, "memagent", {"path": "Patient.md"})

    assert result.ok is False
    assert "requires model access" in result.content["error"]


def test_memagent_extracts_patient_laboratory_store_and_queries(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "Patient.md").write_text(
        "\n\n".join(
            [
                "Patient ID 1001. SEX: F. Birthday: 1960-02-03. Diagnosis: hepatitis.",
                "Patient ID 1002. SEX: M. Birthday: 1980-01-01. First visit: None.",
                "This is narrative background with no extractable identifier.",
            ]
        ),
        encoding="utf-8",
    )
    (task.context_dir / "Laboratory.md").write_text(
        "\n\n".join(
            [
                "Patient ID 1001 had laboratory date 2020-05-01 with CRE: 1.2 and GOT: 40.",
                "Patient ID 1001 had laboratory date 2020-05-02 with CRE initially 2.5 but corrected to 1.8.",
                "Laboratory date 2020-05-03 CRE: 1.1 without a patient key.",
            ]
        ),
        encoding="utf-8",
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=FakeRuleModel(),
    )

    result = registry.execute(
        runtime_context,
        "memagent",
        {"paths": ["Patient.md", "Laboratory.md"], "goal": "extract patient and labs"},
    )

    assert result.ok is True
    store_id = result.content["store_id"]
    assert store_id in runtime_context.memagent_stores
    assert {table["name"] for table in result.content["tables"]} == {"Patient", "Laboratory"}

    query = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {
            "store_id": store_id,
            "sql": (
                "SELECT p.ID, p.SEX, l.Date, l.CRE "
                "FROM Patient p JOIN Laboratory l ON p.ID = l.ID "
                "WHERE l.CRE > 1.5 ORDER BY l.Date"
            ),
        },
    )
    assert query.ok is True
    assert query.content["columns"] == ["ID", "SEX", "Date", "CRE"]
    assert query.content["rows"] == [[1001, "F", "2020-05-02", 1.8]]

    unresolved = registry.execute(
        runtime_context,
        "read_memagent_unresolved",
        {"store_id": store_id, "status": "partial", "limit": 10},
    )
    assert unresolved.ok is True
    assert any("without a patient key" in row["text"] for row in unresolved.content["rows"])

    rejected = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {"store_id": store_id, "sql": "DELETE FROM Patient"},
    )
    assert rejected.ok is False
    assert "Only read-only SQL" in rejected.content["error"]


def test_memagent_extracts_superhero_wide_table(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "superhero.md").write_text(
        "\n\n".join(
            [
                "Entry 1 codename: Alpha. full name: Alice A. height is recorded as 170 centimeters and publisher affiliation is logged with the code 13.",
                "Entry 2 codename: Beta. height was initially 160 centimeters but corrected to 180 centimeters. publisher affiliation is logged as 4.",
                "Entry 2 weight is recorded as 72 kg and gender id is 2.",
            ]
        ),
        encoding="utf-8",
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=FakeRuleModel(),
    )

    result = registry.execute(runtime_context, "memagent", {"path": "superhero.md"})

    assert result.ok is True
    store_id = result.content["store_id"]
    listed = registry.execute(runtime_context, "list_memagent_tables", {"store_id": store_id})
    assert listed.ok is True
    superhero = next(t for t in listed.content["tables"] if t["name"] == "Superhero")
    assert "height_cm" in superhero["columns"]

    query = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {
            "store_id": store_id,
            "sql": "SELECT id, height_cm, publisher_id, weight_kg FROM Superhero ORDER BY id",
        },
    )
    assert query.ok is True
    assert query.content["rows"] == [[1, 170, 13, None], [2, 180, 4, 72]]


def test_memagent_superhero_handles_reference_code_and_placeholder_height(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "superhero.md").write_text(
        "\n\n".join(
            [
                "The Asgardian queen Frigga, cataloged with reference code 278, has a regal stature. Her height is 180.0 centimeters. Her weight is 167.0 kilograms.",
                "The Asgardian queen Frigga, tracked with identifier 278, is registered with publisher 13.",
                "The Starfleet captain Jean-Luc Picard, tracked with identifier 369, has placeholder data of 0.0 for both height and weight, which is inaccurate. His service record lists his height as 175 centimeters.",
                "The Starfleet captain Jean-Luc Picard, registered at ID 369, has publisher affiliation is 20.",
            ]
        ),
        encoding="utf-8",
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=FakeRuleModel(),
    )

    result = registry.execute(runtime_context, "memagent", {"path": "superhero.md"})
    assert result.ok is True
    store_id = result.content["store_id"]

    query = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {
            "store_id": store_id,
            "sql": "SELECT id, height_cm, weight_kg, publisher_id FROM Superhero ORDER BY id",
        },
    )

    assert query.ok is True
    assert query.content["rows"] == [[278, 180, 167, 13], [369, 0, 0, 20]]


def test_memagent_unknown_store_id_is_rejected(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {"store_id": "missing", "sql": "SELECT 1"},
    )

    assert result.ok is False
    assert result.content["known_store_ids"] == []


def test_memagent_uses_model_generated_rule_json_for_regexes(tmp_path: Path) -> None:
    task = _etl_task(tmp_path)
    (task.context_dir / "custom.md").write_text(
        "Record key 10 carries amount is 7.5 units.",
        encoding="utf-8",
    )

    class FakeRuleModel:
        def invoke(self, prompt: str):  # noqa: ANN001
            assert "Record key 10" in prompt
            return type(
                "Msg",
                (),
                {
                    "content": (
                        '{"table":"custom","primary_key":["id"],'
                        '"columns":["id","amount"],'
                        '"key_patterns":["Record key (\\\\d+)"],'
                        '"field_patterns":{"amount":["amount is (\\\\d+(?:\\\\.\\\\d+)?)"]},'
                        '"field_types":{"id":"int","amount":"float"}}'
                    )
                },
            )()

    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=FakeRuleModel(),
    )

    result = registry.execute(runtime_context, "memagent", {"path": "custom.md"})
    assert result.ok is True
    store_id = result.content["store_id"]
    assert result.content["diagnostics"]["rules"][0]["source"] == "llm"

    query = registry.execute(
        runtime_context,
        "query_memagent_sql",
        {"store_id": store_id, "sql": "SELECT id, amount FROM custom"},
    )

    assert query.ok is True
    assert query.content["rows"] == [[10, 7.5]]
