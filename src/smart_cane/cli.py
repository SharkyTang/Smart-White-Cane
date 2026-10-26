"""Command-line interface for the smart white cane learning prototype."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from smart_cane import __version__
from smart_cane.app import run_demo, run_live
from smart_cane.config import DEFAULT_MODEL, RuntimeConfig, parse_source
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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model name or weight path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="minimum detector confidence from 0 to 1 (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device such as cpu or mps; measured CPU default when omitted",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        metavar="PIXELS",
        help="YOLO square inference size (default: 640)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        metavar="N",
        help="leading frames excluded from steady-state metrics (default: 5)",
    )
    parser.add_argument(
        "--confirmation-window",
        type=int,
        default=5,
        metavar="N",
        help="temporal confirmation window in frames (default: 5)",
    )
    parser.add_argument(
        "--minimum-confirmations",
        type=int,
        default=3,
        metavar="N",
        help="observations required inside the window (default: 3)",
    )
    parser.add_argument(
        "--high-risk-confirmations",
        type=int,
        default=2,
        metavar="N",
        help="consecutive large-box observations for fast confirmation (default: 2)",
    )
    parser.add_argument(
        "--high-risk-area-ratio",
        type=float,
        default=0.18,
        metavar="RATIO",
        help="shared large-target and high-risk area threshold (default: 0.18)",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=3,
        metavar="N",
        help="consecutive misses before a stable target is cleared (default: 3)",
    )
    parser.add_argument(
        "--direction-hysteresis",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="left/center/right boundary hysteresis (default: 0.05)",
    )
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable optional macOS speech alerts",
    )
    parser.add_argument("--voice", default=None, help="optional macOS say voice name")
    parser.add_argument(
        "--alert-cooldown",
        "--audio-cooldown",
        dest="alert_cooldown",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="repeat cooldown for the same tracked warning (default: 3.0)",
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
            image_size=args.image_size,
            warmup_frames=args.warmup_frames,
            confirmation_window=args.confirmation_window,
            minimum_confirmations=args.minimum_confirmations,
            high_risk_area_ratio=args.high_risk_area_ratio,
            high_risk_consecutive_frames=args.high_risk_confirmations,
            max_missing_frames=args.max_missing_frames,
            direction_hysteresis=args.direction_hysteresis,
            alert_cooldown_seconds=args.alert_cooldown,
            audio_enabled=args.audio,
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

    inference = summary.metrics.stage("inference") if summary.metrics is not None else None
    print(
        f"Processed {summary.frames} frames on {summary.device}: "
        f"{summary.detections} raw / {summary.stable_detections} stable detections, "
        f"{summary.alerts} alert events / {summary.spoken_alerts} spoken."
    )
    if inference is not None:
        if summary.metrics.steady_frames:
            print(
                f"Cold inference {summary.cold_inference_seconds * 1000.0:.1f} ms; "
                f"steady {summary.steady_average_fps:.1f} FPS after "
                f"{summary.metrics.warmup_frames} warmup frames; inference "
                f"mean/median/p95 {inference.mean_ms:.1f}/{inference.median_ms:.1f}/"
                f"{inference.p95_ms:.1f} ms."
            )
        else:
            print(
                f"Cold inference {summary.cold_inference_seconds * 1000.0:.1f} ms; "
                "no steady-state samples (run more frames than the warmup count)."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
