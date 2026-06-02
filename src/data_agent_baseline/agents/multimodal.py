from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"}


def find_first_video(context_dir: Path) -> Path | None:
    if not context_dir.exists():
        return None
    for path in sorted(context_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def build_initial_user_content(task: PublicTask, text: str) -> str | list[dict[str, Any]]:
    video_path = find_first_video(task.context_dir)
    if video_path is None:
        return text

    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    try:
        relative_video_path = video_path.relative_to(task.context_dir)
    except ValueError:
        relative_video_path = video_path.name

    return [
        {
            "type": "text",
            "text": (
                f"{text}\n\n"
                "<video_context>\n"
                f"A video file from the task context is attached: {relative_video_path}.\n"
                "Observe the attached video directly and combine it with tool evidence when solving.\n"
                "</video_context>"
            ),
        },
        {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{video_b64}"},
        },
    ]
