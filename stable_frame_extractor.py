#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if PROJECT_SRC.exists():
    sys.path.insert(0, str(PROJECT_SRC))


def main() -> None:
    from data_agent_baseline.run.video_preprocessor import extract_stable_frames

    parser = argparse.ArgumentParser(
        description="Extract stable screens from video using visual frame difference only."
    )
    parser.add_argument("video", type=Path, help="Input video path, e.g. input.mp4")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("stable_frames"),
        help="Output directory",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="How many frames per second to sample for analysis",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.025,
        help="Transition threshold; higher = fewer cuts",
    )
    parser.add_argument(
        "--pixel-delta",
        type=int,
        default=25,
        help="Pixel difference threshold used before counting changed pixels",
    )
    parser.add_argument(
        "--min-stable-duration",
        type=float,
        default=1.0,
        help="Minimum stable interval duration to save",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=320,
        help="Width for low-res difference calculation",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Do not remove near-duplicate stable screens",
    )
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=4,
        help="dHash Hamming distance for duplicate filtering",
    )
    args = parser.parse_args()

    manifest = extract_stable_frames(
        video_path=args.video,
        output_dir=args.output,
        sample_fps=args.sample_fps,
        diff_threshold=args.diff_threshold,
        pixel_delta=args.pixel_delta,
        min_stable_duration=args.min_stable_duration,
        resize_width=args.resize_width,
        dedup=not args.no_dedup,
        hash_threshold=args.hash_threshold,
    )
    print(
        json.dumps(
            {
                "duration_sec": manifest["duration_sec"],
                "sample_count": manifest["sample_count"],
                "stable_segment_count": manifest["stable_segment_count"],
                "saved_image_count": manifest["saved_image_count"],
                "output_dir": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
