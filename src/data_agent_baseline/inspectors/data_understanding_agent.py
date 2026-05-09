from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.prompts import build_global_profiling_prompt
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog


class DataUnderstandingAgent:
    def __init__(self, config: DataInspectorConfig) -> None:
        self.config = config

    def explore_data_globally(self, *, context_dir: Path, task_id: str = "") -> str:
        catalog = build_semantic_catalog(
            PublicTask(
                record=type("TaskRecord", (), {"task_id": task_id, "difficulty": "", "question": ""})(),
                assets=type("TaskAssets", (), {"task_dir": context_dir, "context_dir": context_dir})(),
            ),
            budget=self.config.sample_budget,
        )
        knowledge_docs: list[dict[str, Any]] = []
        for schema in catalog.get("schemas", []):
            if schema.get("kind") == "document":
                doc_content = schema.get("content") or schema.get("preview") or ""
                if doc_content.strip():
                    knowledge_docs.append(
                        {
                            "asset_path": schema.get("asset_path", ""),
                            "content": doc_content,
                            "char_count": schema.get("char_count", len(doc_content)),
                            "headings": schema.get("headings", []),
                        }
                    )

        return build_global_profiling_prompt(catalog=catalog, knowledge_docs=knowledge_docs)
