from __future__ import annotations

from typing import Any

from data_agent_baseline.benchmark.context_view import iter_context_file_assets
from data_agent_baseline.benchmark.schema import ContextAsset, PublicTask


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _video_timeline_assets(task: PublicTask) -> list[ContextAsset]:
    return [
        asset
        for asset in iter_context_file_assets(task)
        if asset.action in {"video_timeline", "video_preprocessing_failed"}
    ]


def _video_summary_assets(task: PublicTask) -> list[ContextAsset]:
    return [
        asset
        for asset in iter_context_file_assets(task)
        if asset.action == "video_summary"
    ]


def _stable_frame_assets(task: PublicTask) -> list[ContextAsset]:
    return [
        asset
        for asset in iter_context_file_assets(task)
        if asset.action == "video_stable_frame"
        and asset.physical_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _video_context_text(
    *,
    timelines: list[ContextAsset],
    stable_frames: list[ContextAsset],
) -> str:
    timeline_blocks = []
    for asset in timelines:
        timeline_text = asset.physical_path.read_text(encoding="utf-8", errors="replace")
        timeline_blocks.append(f"## `{asset.visible_path}`\n\n{timeline_text.strip()}")
    frame_lines = "\n".join(
        f"- Image {index}: `{asset.visible_path}`"
        for index, asset in enumerate(stable_frames, start=1)
    )
    if not frame_lines:
        frame_lines = "- No stable-frame images were generated."
    return (
        "<video_context>\n"
        "Original task videos were preprocessed before this model request. "
        "The raw video files are intentionally not attached. Timeline text is included "
        "below. Stable-frame images are listed by path; call `read_context_image` when "
        "you need to inspect a specific frame visually.\n\n"
        "Timeline document content:\n"
        f"{chr(10).join(timeline_blocks)}\n\n"
        "Stable-frame image path(s), in chronological order:\n"
        f"{frame_lines}"
        "\n"
        "</video_context>"
    )


def _video_summary_context_text(*, summaries: list[ContextAsset]) -> str:
    summary_blocks = []
    for asset in summaries:
        summary_text = asset.physical_path.read_text(encoding="utf-8", errors="replace")
        summary_blocks.append(f"## `{asset.visible_path}`\n\n{summary_text.strip()}")
    return (
        "<video_context>\n"
        "Original task videos were preprocessed before this model request. "
        "The raw video files are intentionally not attached. A pre-main video "
        "understanding agent summary is included below. Treat explicit, non-uncertain "
        "facts in this summary as observed video evidence that can be used directly. "
        "When submitting values from the summary, preserve exact visible punctuation, "
        "separators, spaces, hyphens, and Chinese text; prefer any exact-value entries. "
        "Call `read_doc` on the original timeline and `read_context_image` on relevant "
        "stable-frame images only when the summary says it failed, marks a needed fact "
        "as uncertain, omits a needed fact, conflicts with other observed evidence, or "
        "the task explicitly requires checking the original visual/audio evidence.\n\n"
        "Video summary document content:\n"
        f"{chr(10).join(summary_blocks)}"
        "\n"
        "</video_context>"
    )


def build_initial_user_content(
    task: PublicTask,
    text: str,
    *,
    max_attached_frames: int = 16,
) -> str | list[dict[str, Any]]:
    summaries = _video_summary_assets(task)
    if summaries:
        return f"{text}\n\n" + _video_summary_context_text(summaries=summaries)

    timelines = _video_timeline_assets(task)
    if not timelines:
        return text

    stable_frames = _stable_frame_assets(task)
    listed_frames = stable_frames[:max_attached_frames] if max_attached_frames > 0 else []
    full_text = (
        f"{text}\n\n"
        + _video_context_text(
            timelines=timelines,
            stable_frames=listed_frames,
        )
    )
    omitted_count = max(len(stable_frames) - len(listed_frames), 0)
    if omitted_count:
        full_text += f"\n{omitted_count} additional stable-frame image path(s) were omitted.\n"
    return full_text
