"""Command-line interface for the smart white cane learning prototype."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from smart_cane import __version__
from smart_cane.app import run_demo, run_live
from smart_cane.config import RuntimeConfig, parse_source
from smart_cane.perception.camera import CameraError
from smart_cane.perception.detector import DetectorError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-cane",
        description="Mac-based obstacle-awareness learning prototype.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run deterministic sample detections without a camera or model",
    )
    parser.add_argument(
        "--source",
        default="0",
        metavar="CAMERA_OR_VIDEO",
        help="camera index (default: 0) or video path",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model name or weight path")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="minimum detector confidence from 0 to 1 (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device such as cpu or mps; automatic when omitted",
    )
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable optional macOS speech alerts",
    )
    parser.add_argument("--voice", default=None, help="optional macOS say voice name")
    parser.add_argument(
        "--audio-cooldown",
        type=float,
        default=2.5,
        metavar="SECONDS",
        help="minimum time between spoken alerts (default: 2.5)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="process frames without opening an OpenCV window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after N frames; useful for repeatable smoke tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = RuntimeConfig(
            source=parse_source(args.source),
            model=args.model,
            confidence=args.confidence,
            device=args.device,
            audio_enabled=args.audio,
            audio_cooldown_seconds=args.audio_cooldown,
            voice=args.voice,
            display=not args.no_display,
            max_frames=args.max_frames,
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        if args.demo:
            run_demo(config)
            return 0

        summary = run_live(config)
    except (CameraError, DetectorError, RuntimeError) as error:
        print(f"smart-cane: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130

    print(
        f"Processed {summary.frames} frames, {summary.detections} detections, "
        f"{summary.alerts} spoken alerts at {summary.average_fps:.1f} average FPS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
