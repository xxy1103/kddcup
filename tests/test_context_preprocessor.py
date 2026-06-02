from __future__ import annotations

import json
from pathlib import Path

import fitz

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.run.context_preprocessor import (
    _coalesce_markdown_lines,
    pdf_to_markdown,
    prepare_task_context,
)
from data_agent_baseline.tools.filesystem import list_context_tree, read_doc_preview
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


def _write_pdf(path: Path, lines: list[tuple[float, str]], *, toc: list[list[object]] | None = None) -> None:
    document = fitz.open()
    page = document.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=12)
    if toc:
        document.set_toc(toc)
    document.save(path)
    document.close()


def _task_with_context(task_dir: Path) -> PublicTask:
    context_dir = task_dir / "context"
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="question"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_pdf_to_markdown_uses_pdf_toc_as_headings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_toc.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "Main Report"),
            (96, "The first paragraph continues"),
            (120, "without punctuation"),
            (168, "Detailed Section"),
            (192, "Section content."),
        ],
        toc=[[1, "Main Report", 1], [2, "Detailed Section", 1]],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert "# Main Report" in markdown
    assert "## Detailed Section" in markdown
    assert "The first paragraph continues without punctuation" in markdown


def test_pdf_to_markdown_does_not_infer_headings_without_toc(tmp_path: Path) -> None:
    pdf_path = tmp_path / "without_toc.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "Capital Market Audit Summary"),
            (96, "Body text."),
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert "Capital Market Audit Summary" in markdown
    assert "#" not in markdown


def test_pdf_to_markdown_repairs_cjk_hard_line_wraps(tmp_path: Path) -> None:
    del tmp_path

    markdown = _coalesce_markdown_lines(
        [
            "其适用对象明确为城市信",
            "用社。此举旨在调节该类机构。",
        ]
    )

    assert "城市信用社" in markdown


def test_prepare_task_context_converts_pdfs_and_preserves_md_collisions(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True)
    (context_dir / "doc" / "report.md").write_text("# Existing\n", encoding="utf-8")
    _write_pdf(context_dir / "doc" / "report.pdf", [(72, "PDF body.")])
    task = _task_with_context(task_dir)

    preprocessed = prepare_task_context(task, tmp_path / "output" / "task_1")

    manifest = json.loads(
        (tmp_path / "output" / "task_1" / "context_preprocessing_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_by_visible_path = {entry["visible_path"]: entry for entry in manifest["entries"]}
    assert manifest_by_visible_path["doc/report.md"]["action"] == "source"
    assert manifest_by_visible_path["doc/report_pdf.md"]["action"] == "pdf_to_markdown"
    assert manifest_by_visible_path["doc/report_pdf.md"]["generated"] is True
    assert not (tmp_path / "output" / "task_1" / "context").exists()
    assert (preprocessed.generated_context_dir / "doc" / "report_pdf.md").exists()
    assert not (preprocessed.generated_context_dir / "doc" / "report.md").exists()
    assert preprocessed.task.context_dir == context_dir
    listed_paths = {
        entry["path"]
        for entry in list_context_tree(preprocessed.task)["entries"]
        if entry["kind"] == "file"
    }
    assert listed_paths == {"doc/report.md", "doc/report_pdf.md"}
    assert read_doc_preview(preprocessed.task, "doc/report.md")["preview"] == "# Existing\n"


def test_task_context_workspace_materializes_overlay_without_pdfs(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "doc").mkdir()
    (context_dir / "csv" / "data.csv").write_text("id,value\n1,A\n", encoding="utf-8")
    _write_pdf(context_dir / "doc" / "report.pdf", [(72, "PDF body.")])
    task = _task_with_context(task_dir)
    preprocessed = prepare_task_context(task, tmp_path / "output" / "task_1")

    workspace = TaskContextWorkspace(
        source_root=preprocessed.task.context_dir,
        context_view=preprocessed.context_view,
    )
    workspace_root = workspace.materialize()

    assert (workspace_root / "csv" / "data.csv").read_text(encoding="utf-8") == "id,value\n1,A\n"
    assert (workspace_root / "doc" / "report.md").exists()
    assert not (workspace_root / "doc" / "report.pdf").exists()
    workspace.cleanup()
    assert workspace.path is None
    assert not workspace_root.exists()
