from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.multimodal import _video_summary_context_text
from data_agent_baseline.agents.prompt import build_system_prompt
from data_agent_baseline.agents.prompt2 import build_system_prompt_v2
from data_agent_baseline.agents.prompt3 import (
    SYSTEM_PROMPT_V3_ZH_REFERENCE,
    build_system_prompt_v3,
)
from data_agent_baseline.benchmark.schema import ContextAsset


def test_prompt_v3_uses_the_unified_four_phase_protocol() -> None:
    prompt = build_system_prompt_v3()

    assert "## Unified task execution protocol and source routing" in prompt
    assert "### Phase 1 — task contract and video-rule confirmation" in prompt
    assert "### Phase 4 — validate and reproducibly submit the answer" in prompt
    assert "the injected video summary is a locator only, never final evidence" in prompt
    assert "call 'read_doc' on the relevant timeline" in prompt
    assert "call 'read_context_image' on every stable frame" in prompt
    assert "call 'record_visual_evidence' for that same frame" in prompt
    assert "All values returned by 'extract_structured_doc' are already normalized to base unit 1" in prompt
    assert "100 million is returned as 100000000 yuan and 1% as 0.01" in prompt
    assert "## 统一任务执行协议与来源路由" in SYSTEM_PROMPT_V3_ZH_REFERENCE


def test_prompt_v3_distinguishes_entity_sets_from_source_record_sets() -> None:
    prompt = build_system_prompt_v3()

    assert "explicit record-level wording takes precedence over generic" in prompt.lower()
    assert "First classify the requested result as an entity set or a source record set" in prompt
    assert "complete set in the final output" in prompt
    assert "Never deduplicate a source record set" in prompt
    assert "实体集合或来源记录集合" in SYSTEM_PROMPT_V3_ZH_REFERENCE
    assert "完整集合" in SYSTEM_PROMPT_V3_ZH_REFERENCE
    assert "绝不得对来源记录集合去重" in SYSTEM_PROMPT_V3_ZH_REFERENCE


def test_video_summary_injection_is_locator_only_and_overrides_legacy_text(tmp_path: Path) -> None:
    summary_path = tmp_path / "briefing_video_summary.md"
    summary_path.write_text(
        "Explicit facts may be used directly as observed video evidence.\nFrame 1 contains a threshold.",
        encoding="utf-8",
    )

    content = _video_summary_context_text(
        summaries=[
            ContextAsset(
                visible_path="video/briefing_video_summary.md",
                physical_path=summary_path,
                action="video_summary",
            )
        ]
    )

    assert "locator and planning aid only, never final video evidence" in content
    assert "call `read_doc` on the relevant video timeline" in content
    assert "`read_context_image` on every stable frame" in content
    assert "Policy override (takes precedence" in content
    assert content.index("may be used directly") < content.index("Policy override")
    assert "must be ignored" in content


def test_v1_v2_video_prompts_do_not_authorize_summary_as_evidence() -> None:
    for prompt in (build_system_prompt(), build_system_prompt_v2()):
        assert "Treat explicit facts in the summary as observed video evidence" not in prompt
        assert "locator and planning aid" in prompt
        assert "read_context_image" in prompt
