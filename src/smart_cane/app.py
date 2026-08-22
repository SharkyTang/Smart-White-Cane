"""Application orchestration for live and simulated smart-cane loops."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smart_cane.config import RuntimeConfig
from smart_cane.decision.risk_engine import RiskAssessment, RiskEngine, RiskRules
from smart_cane.feedback.audio import AudioFeedback
from smart_cane.models import Detection
from smart_cane.perception.camera import CameraReadError, CameraSource
from smart_cane.perception.detector import YoloDetector


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames: int
    detections: int
    alerts: int
    elapsed_seconds: float

    @property
    def average_fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def demo_detections() -> tuple[Detection, ...]:
    """Deterministic detections that exercise every direction and risk level."""

    return (
        Detection("chair", 0.92, (210, 80, 470, 450), 640, 480),
        Detection("person", 0.88, (18, 100, 150, 410), 640, 480),
        Detection("bottle", 0.76, (535, 250, 575, 390), 640, 480),
    )


def run_demo(config: RuntimeConfig) -> tuple[RiskAssessment, ...]:
    """Run the domain pipeline without camera, OpenCV, a model, or downloads."""

    engine = RiskEngine(RiskRules(minimum_confidence=config.confidence))
    assessments = engine.assess_all(demo_detections())
    urgent = engine.most_urgent(demo_detections())
    feedback = AudioFeedback(
        enabled=config.audio_enabled,
        cooldown_seconds=config.audio_cooldown_seconds,
        voice=config.voice,
    )

    print("Smart White Cane Lab — deterministic demo")
    print("label       confidence  direction  risk    alert  area")
    for item in assessments:
        print(
            f"{item.label:<11} {item.confidence:>10.2f}  {item.direction:<9} "
            f"{item.level:<7} {str(item.should_alert):<5}  {item.area_ratio:>5.1%}"
        )
    if urgent is not None:
        print(f"Most urgent: {urgent.message}")
        if urgent.should_alert:
            feedback.notify(urgent.message)
    return assessments


def run_live(config: RuntimeConfig) -> RunSummary:
    """Run camera/video input through YOLO, risk rules, display, and audio."""

    cv2 = _load_cv2()
    backend = None
    if sys.platform == "darwin" and isinstance(config.source, int):
        backend = getattr(cv2, "CAP_AVFOUNDATION", None)

    source = CameraSource(config.source, backend=backend, cv2_module=cv2)
    detector = YoloDetector(
        config.model,
        confidence=config.confidence,
        device=config.device,
    )
    engine = RiskEngine(RiskRules(minimum_confidence=config.confidence))
    feedback = AudioFeedback(
        enabled=config.audio_enabled,
        cooldown_seconds=config.audio_cooldown_seconds,
        voice=config.voice,
    )

    frames = detections_seen = alerts = 0
    started_at = time.monotonic()
    last_frame_at = started_at
    fps = 0.0

    try:
        with source:
            while config.max_frames is None or frames < config.max_frames:
                try:
                    frame = source.read()
                except CameraReadError:
                    break

                detections = detector(frame)
                assessments = engine.assess_all(detections)
                urgent = engine.most_urgent(detections)

                frames += 1
                detections_seen += len(detections)
                now = time.monotonic()
                instantaneous_fps = 1.0 / max(now - last_frame_at, 1e-9)
                fps = instantaneous_fps if fps == 0 else (0.9 * fps + 0.1 * instantaneous_fps)
                last_frame_at = now

                if urgent is not None and urgent.should_alert:
                    if feedback.notify(urgent.message):
                        alerts += 1

                if config.display:
                    _draw_overlay(cv2, frame, detections, assessments, fps)
                    cv2.imshow("Smart White Cane Lab", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
    finally:
        source.release()
        if config.display:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 0.0)
    return RunSummary(frames, detections_seen, alerts, elapsed)


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is not installed; run `uv sync` first") from error
    return cv2


def _draw_overlay(
    cv2: Any,
    frame: Any,
    detections: Sequence[Detection],
    assessments: Sequence[RiskAssessment],
    fps: float,
) -> None:
    colors = {
        "low": (160, 160, 160),
        "medium": (0, 170, 255),
        "high": (30, 30, 230),
    }
    for detection, assessment in zip(detections, assessments, strict=True):
        x_min, y_min, x_max, y_max = (int(value) for value in detection.bbox)
        color = colors[assessment.level]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        text = (
            f"{detection.label} {detection.confidence:.2f} | "
            f"{assessment.direction} {assessment.level}"
        )
        cv2.putText(
            frame,
            text,
            (max(0, x_min), max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    status = f"FPS {fps:.1f} | objects {len(detections)} | q / Esc to quit"
    cv2.putText(
        frame,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
