from __future__ import annotations

import base64
import mimetypes
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


def _stable_frame_assets(task: PublicTask) -> list[ContextAsset]:
    return [
        asset
        for asset in iter_context_file_assets(task)
        if asset.action == "video_stable_frame"
        and asset.physical_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _image_part(asset: ContextAsset) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(asset.physical_path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(asset.physical_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
    }


def _video_context_text(
    *,
    timelines: list[ContextAsset],
    attached_frames: list[ContextAsset],
    omitted_frame_count: int,
) -> str:
    timeline_lines = "\n".join(f"- `{asset.visible_path}`" for asset in timelines)
    frame_lines = "\n".join(
        f"- Image {index}: `{asset.visible_path}`"
        for index, asset in enumerate(attached_frames, start=1)
    )
    if not frame_lines:
        frame_lines = "- No stable-frame images are attached."
    omitted_line = (
        f"\n{omitted_frame_count} additional stable-frame image(s) were generated but not attached."
        if omitted_frame_count > 0
        else ""
    )
    return (
        "<video_context>\n"
        "Original task videos were preprocessed before this model request. "
        "The raw video files are intentionally not attached. Use the timeline document(s), "
        "their ASR transcript sections, and the attached stable-frame images together. "
        "Call `read_doc` on the timeline document path if the transcript is not already "
        "included in the provided catalog.\n\n"
        "Timeline document(s):\n"
        f"{timeline_lines}\n\n"
        "Attached stable-frame image(s), in chronological order:\n"
        f"{frame_lines}"
        f"{omitted_line}\n"
        "</video_context>"
    )


def build_initial_user_content(
    task: PublicTask,
    text: str,
    *,
    max_attached_frames: int = 16,
) -> str | list[dict[str, Any]]:
    timelines = _video_timeline_assets(task)
    if not timelines:
        return text

    stable_frames = _stable_frame_assets(task)
    attached_frames = stable_frames[:max_attached_frames]
    omitted_frame_count = max(len(stable_frames) - len(attached_frames), 0)
    full_text = (
        f"{text}\n\n"
        + _video_context_text(
            timelines=timelines,
            attached_frames=attached_frames,
            omitted_frame_count=omitted_frame_count,
        )
    )

    if not attached_frames:
        return full_text

    return [
        {"type": "text", "text": full_text},
        *[_image_part(asset) for asset in attached_frames],
    ]
