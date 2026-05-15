from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog


class DataUnderstandingAgent:
    def __init__(self, config: DataInspectorConfig) -> None:
        self.config = config

    def explore_data_globally(self, *, context_dir: Path, task_id: str = "") -> tuple[str, dict[str, Any]]:
        catalog = build_semantic_catalog(
            PublicTask(
                record=type("TaskRecord", (), {"task_id": task_id, "difficulty": "", "question": ""})(),
                assets=type("TaskAssets", (), {"task_dir": context_dir, "context_dir": context_dir})(),
            ),
            budget=self.config.sample_budget,
        )
        return json.dumps(catalog, ensure_ascii=False, indent=2), catalog
