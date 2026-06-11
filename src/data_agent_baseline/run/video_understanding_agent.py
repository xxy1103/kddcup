from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.benchmark.schema import (
    ContextAsset,
    ContextView,
    PublicTask,
    TaskAssets,
)
from data_agent_baseline.model_retry import invoke_model_with_retries
from data_agent_baseline.run.context_preprocessor import PreprocessedContext


VIDEO_SUMMARY_ACTION = "video_summary"

VIDEO_UNDERSTANDING_SYSTEM_PROMPT = """
You are a video evidence summarization agent for a data-analysis benchmark.
Your job is to summarize only the supplied video timeline and stable-frame images.
Do not solve the user's data task and do not use or infer structured-data results.

Produce a concise Markdown document with:
- A short overview of what the video shows.
- Key time ranges and what each contributes.
- Visible UI/table/value/filter/year/join clues seen in the images.
- An "Exact Extracted Values" section for important visible values that might be
  submitted as answers.
- Audible criteria and transcript clues from the timeline.
- ASR/OCR uncertainty or contradictions.
- Evidence references using timestamps and stable-frame paths.

The summary will be injected into another agent's initial context. Explicit
non-uncertain facts in it may be used directly as observed video evidence, so state
uncertainties clearly and preserve paths and timestamps that the main agent can inspect
later when needed.

For the "Exact Extracted Values" section:
- Copy answer-like visible values exactly as displayed in the image, including Chinese
  text, punctuation, separators, spaces, hyphens, parentheses, and line breaks.
- Do not translate, romanize, normalize, reformat, or replace separators. For example,
  keep a visible phone list separator such as `、` if that is what appears on screen;
  do not change it to `,`.
- For each value, include the stable-frame path that supports it.
- If the exact displayed punctuation or text is unclear, mark that value as uncertain
  instead of presenting a normalized best guess.
""".strip()


@dataclass(frozen=True, slots=True)
class VideoSummaryResult:
    source_path: str
    timeline_path: str
    summary_asset: ContextAsset
    frame_paths: tuple[str, ...]
    status: str
    debug_summary_path: Path | None = None
    error: str | None = None


def _video_timeline_assets(task: PublicTask) -> list[ContextAsset]:
    return [
        asset
        for asset in (task.assets.context_view.assets if task.assets.context_view else ())
        if asset.action in {"video_timeline", "video_preprocessing_failed"}
    ]


def _stable_frame_assets_for_source(task: PublicTask, source_path: str | None) -> list[ContextAsset]:
    return [
        asset
        for asset in (task.assets.context_view.assets if task.assets.context_view else ())
        if asset.action == "video_stable_frame" and asset.source_path == source_path
    ]


def _summary_visible_path(timeline_asset: ContextAsset) -> str:
    source_path = PurePosixPath(timeline_asset.source_path or timeline_asset.visible_path)
    parent = "" if str(source_path.parent) == "." else source_path.parent.as_posix()
    stem = source_path.stem
    if timeline_asset.source_path is None:
        timeline_path = PurePosixPath(timeline_asset.visible_path)
        stem = timeline_path.stem.removesuffix("_timeline")
        parent = "" if str(timeline_path.parent) == "." else timeline_path.parent.as_posix()
    filename = f"{stem}_video_summary.md"
    return f"{parent}/{filename}" if parent else filename


def _video_artifact_dir_name(source_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", source_path).strip("_") or "video"


def _debug_summary_path(task_output_dir: Path, timeline_asset: ContextAsset) -> Path:
    source_path = timeline_asset.source_path or timeline_asset.visible_path
    return task_output_dir / "video_understanding" / _video_artifact_dir_name(source_path) / "summary.md"


def _image_part(asset: ContextAsset) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(asset.physical_path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(asset.physical_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_b64}",
            "detail": "high",
        },
    }


def _render_user_content(
    *,
    timeline_asset: ContextAsset,
    timeline_text: str,
    frame_assets: list[ContextAsset],
) -> list[dict[str, Any]]:
    frame_list = "\n".join(f"- {asset.visible_path}" for asset in frame_assets) or "- none"
    text = (
        "Summarize the following preprocessed video evidence for the main data agent.\n\n"
        f"Source video: `{timeline_asset.source_path or 'unknown'}`\n"
        f"Timeline document path: `{timeline_asset.visible_path}`\n"
        "Stable-frame paths in chronological order:\n"
        f"{frame_list}\n\n"
        "Timeline document content:\n"
        "```markdown\n"
        f"{timeline_text.strip()}\n"
        "```\n\n"
        "The stable-frame images are attached after this text in the same order as listed."
    )
    return [{"type": "text", "text": text}, *[_image_part(asset) for asset in frame_assets]]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def _success_summary_text(
    *,
    raw_summary: str,
    timeline_path: str,
    frame_paths: tuple[str, ...],
) -> str:
    del frame_paths
    return (
        "# Video Understanding Summary\n\n"
        "This document was generated by the pre-main video understanding agent. "
        "Explicit non-uncertain facts may be used directly as observed video evidence. "
        "Final answers should preserve exact visible values, punctuation, and separators "
        "from the summary's exact-value entries; if formatting is unclear, inspect the "
        "referenced stable frame.\n\n"
        f"- Original timeline: `{timeline_path}`\n"
        "## Summary\n\n"
        f"{raw_summary.strip()}\n"
    )


def _failure_summary_text(
    *,
    error: str,
    timeline_path: str,
    frame_paths: tuple[str, ...],
) -> str:
    frame_list = "\n".join(f"- `{path}`" for path in frame_paths) or "- none"
    return (
        "# Video Understanding Summary Unavailable\n\n"
        "The pre-main video understanding agent failed, so this summary is not reliable.\n\n"
        f"- Error: {error}\n"
        f"- Original timeline to read: `{timeline_path}`\n"
        "- Stable frames to inspect with `read_context_image`:\n"
        f"{frame_list}\n\n"
        "Main agent instruction: do not rely on this failed summary. You must read the "
        "timeline document with `read_doc` and inspect the relevant stable-frame images "
        "with `read_context_image` to understand the video evidence before using it."
    )


def summarize_video_asset(
    *,
    model: Any,
    task: PublicTask,
    timeline_asset: ContextAsset,
    output_path: Path,
    debug_output_path: Path | None,
    timeout_seconds: int | None,
) -> VideoSummaryResult:
    frame_assets = _stable_frame_assets_for_source(task, timeline_asset.source_path)
    frame_paths = tuple(asset.visible_path for asset in frame_assets)
    timeline_text = timeline_asset.physical_path.read_text(encoding="utf-8", errors="replace")

    try:
        messages = [
            SystemMessage(content=VIDEO_UNDERSTANDING_SYSTEM_PROMPT),
            HumanMessage(
                content=_render_user_content(
                    timeline_asset=timeline_asset,
                    timeline_text=timeline_text,
                    frame_assets=frame_assets,
                )
            ),
        ]
        response = invoke_model_with_retries(
            model,
            messages,
            timeout_seconds=timeout_seconds,
        )
        raw_summary = _message_text(response)
        if not raw_summary:
            raise RuntimeError("Video understanding model returned an empty summary.")
        summary_text = _success_summary_text(
            raw_summary=raw_summary,
            timeline_path=timeline_asset.visible_path,
            frame_paths=frame_paths,
        )
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        summary_text = _failure_summary_text(
            error=error,
            timeline_path=timeline_asset.visible_path,
            frame_paths=frame_paths,
        )
        status = "failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text + "\n", encoding="utf-8")
    if debug_output_path is not None:
        debug_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, debug_output_path)
    summary_asset = ContextAsset(
        visible_path=_summary_visible_path(timeline_asset),
        physical_path=output_path,
        source_path=timeline_asset.source_path,
        action=VIDEO_SUMMARY_ACTION,
        generated=True,
    )
    return VideoSummaryResult(
        source_path=timeline_asset.source_path or timeline_asset.visible_path,
        timeline_path=timeline_asset.visible_path,
        summary_asset=summary_asset,
        frame_paths=frame_paths,
        status=status,
        debug_summary_path=debug_output_path,
        error=error,
    )


def _manifest_entry(result: VideoSummaryResult) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "source_path": result.source_path,
        "visible_path": result.summary_asset.visible_path,
        "physical_path": str(result.summary_asset.physical_path),
        "action": result.summary_asset.action,
        "generated": True,
        "video_understanding": {
            "status": result.status,
            "timeline_path": result.timeline_path,
            "frame_count": len(result.frame_paths),
            "frame_paths": list(result.frame_paths),
        },
    }
    if result.debug_summary_path is not None:
        entry["video_understanding"]["debug_summary_path"] = str(result.debug_summary_path)
    if result.error is not None:
        entry["video_understanding"]["error"] = result.error
    return entry


def _update_context_manifest(task_output_dir: Path, results: list[VideoSummaryResult]) -> None:
    manifest_path = task_output_dir / "context_preprocessing_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("entries") or [])
    existing_summary_paths = {
        entry.get("visible_path")
        for entry in entries
        if isinstance(entry, dict) and entry.get("action") == VIDEO_SUMMARY_ACTION
    }
    for result in results:
        if result.summary_asset.visible_path not in existing_summary_paths:
            entries.append(_manifest_entry(result))
    manifest["entries"] = entries
    manifest["video_understanding"] = {
        "summary_count": len(results),
        "summaries": [_manifest_entry(result)["video_understanding"] | {
            "visible_path": result.summary_asset.visible_path,
            "source_path": result.source_path,
        } for result in results],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_video_understanding_summaries(
    *,
    preprocessed_context: PreprocessedContext,
    task_output_dir: Path,
    model: Any,
    timeout_seconds: int | None,
) -> PreprocessedContext:
    task = preprocessed_context.task
    context_view = task.assets.context_view
    if context_view is None:
        return preprocessed_context

    timelines = _video_timeline_assets(task)
    if not timelines:
        return preprocessed_context

    results: list[VideoSummaryResult] = []
    for timeline_asset in timelines:
        summary_visible_path = _summary_visible_path(timeline_asset)
        output_path = context_view.generated_context_dir / summary_visible_path
        results.append(
            summarize_video_asset(
                model=model,
                task=task,
                timeline_asset=timeline_asset,
                output_path=output_path,
                debug_output_path=_debug_summary_path(task_output_dir, timeline_asset),
                timeout_seconds=timeout_seconds,
            )
        )

    if not results:
        return preprocessed_context

    existing_assets = [
        asset for asset in context_view.assets if asset.action != VIDEO_SUMMARY_ACTION
    ]
    next_context_view = ContextView(
        source_context_dir=context_view.source_context_dir,
        generated_context_dir=context_view.generated_context_dir,
        assets=tuple(
            sorted(
                [*existing_assets, *(result.summary_asset for result in results)],
                key=lambda asset: asset.visible_path,
            )
        ),
    )
    next_task = PublicTask(
        record=task.record,
        assets=TaskAssets(
            task_dir=task.task_dir,
            context_dir=task.context_dir,
            context_view=next_context_view,
        ),
    )
    _update_context_manifest(task_output_dir, results)
    return PreprocessedContext(
        task=next_task,
        context_view=next_context_view,
        generated_context_dir=preprocessed_context.generated_context_dir,
        manifest=preprocessed_context.manifest,
    )
