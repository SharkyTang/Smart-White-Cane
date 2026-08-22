"""Application orchestration for live and simulated smart-cane loops."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from smart_cane.config import RuntimeConfig
from smart_cane.decision.risk_engine import (
    AreaThresholds,
    RiskAssessment,
    RiskEngine,
    RiskRules,
)
from smart_cane.decision.temporal_filter import (
    StableDetection,
    TemporalFilter,
    TemporalFilterConfig,
)
from smart_cane.feedback.alert_policy import AlertDecision, AlertPolicy
from smart_cane.feedback.audio import AudioFeedback
from smart_cane.metrics import MetricsCollector, MetricsSummary
from smart_cane.models import Detection
from smart_cane.perception.camera import CameraReadError, CameraSource
from smart_cane.perception.detector import YoloDetector


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Counts and timings produced by one live run.

    ``detections`` counts raw YOLO boxes. ``stable_detections`` counts the
    confirmed targets passed to the risk engine. ``alerts`` counts policy
    events even when audio is disabled, while ``spoken_alerts`` counts speech
    processes that actually launched.
    """

    frames: int
    detections: int
    alerts: int
    elapsed_seconds: float
    stable_detections: int = 0
    spoken_alerts: int = 0
    cold_inference_seconds: float = 0.0
    device: str = "unknown"
    metrics: MetricsSummary | None = None

    @property
    def average_fps(self) -> float:
        """Whole-run FPS, including lazy model load and warmup."""

        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def steady_average_fps(self) -> float:
        """Steady-state FPS after the configured warmup frames."""

        return self.metrics.steady_average_fps if self.metrics is not None else 0.0


@dataclass(frozen=True, slots=True)
class FrameDecision:
    """Stable detections, risk assessments, and at most one alert per frame."""

    stable_detections: tuple[StableDetection, ...]
    assessments: tuple[RiskAssessment, ...]
    alert: AlertDecision | None
    alert_track_key: str | None = None


def demo_detections() -> tuple[Detection, ...]:
    """Deterministic detections that exercise every direction and risk level."""

    return (
        Detection("chair", 0.92, (210, 80, 470, 450), 640, 480),
        Detection("person", 0.88, (18, 100, 150, 410), 640, 480),
        Detection("bottle", 0.76, (535, 250, 575, 390), 640, 480),
    )


def run_demo(config: RuntimeConfig) -> tuple[RiskAssessment, ...]:
    """Run the stable domain pipeline without camera, OpenCV, or downloads."""

    engine = _build_risk_engine(config)
    temporal_filter = _build_temporal_filter(config)
    stable: tuple[StableDetection, ...] = ()
    for _ in range(config.minimum_confirmations):
        stable = temporal_filter.update(demo_detections())

    assessments = tuple(
        engine.assess(item.detection, direction_override=item.direction) for item in stable
    )
    urgent_index = _most_urgent_index(assessments)
    urgent = assessments[urgent_index] if urgent_index is not None else None
    feedback = AudioFeedback(enabled=config.audio_enabled, cooldown_seconds=0.0, voice=config.voice)

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


def evaluate_detections(
    detections: Iterable[Detection],
    *,
    temporal_filter: TemporalFilter,
    risk_engine: RiskEngine,
    alert_policy: AlertPolicy,
    commit_alert: bool = True,
) -> FrameDecision:
    """Apply temporal stability, risk rules, and alert deduplication to one frame."""

    stable = temporal_filter.update(detections)
    assessments = tuple(
        risk_engine.assess(item.detection, direction_override=item.direction) for item in stable
    )
    alert_policy.retain_active(item.track_key for item in stable)

    alert = None
    alert_track_key = None
    for index in _urgency_order(assessments):
        track_key = stable[index].track_key
        decision = alert_policy.propose(track_key, assessments[index])
        if alert is None:
            alert = decision
            alert_track_key = track_key
        if decision.should_emit:
            alert = decision
            alert_track_key = track_key
            break
    if commit_alert and alert is not None and alert.should_emit:
        assert alert_track_key is not None
        alert_policy.commit(alert_track_key, alert)
    return FrameDecision(stable, assessments, alert, alert_track_key)


def run_live(config: RuntimeConfig) -> RunSummary:
    """Run camera/video input through YOLO, stability, risk, display, and audio."""

    cv2 = _load_cv2()
    backend = None
    if sys.platform == "darwin" and isinstance(config.source, int):
        backend = getattr(cv2, "CAP_AVFOUNDATION", None)

    source = CameraSource(config.source, backend=backend, cv2_module=cv2)
    detector = YoloDetector(
        config.model,
        confidence=config.confidence,
        device=config.device,
        image_size=config.image_size,
    )
    if (
        isinstance(config.device, str)
        and config.device.casefold() == "mps"
        and detector.device == "cpu"
    ):
        print("smart-cane: MPS is unavailable; using CPU instead.", file=sys.stderr)

    engine = _build_risk_engine(config)
    temporal_filter = _build_temporal_filter(config)
    alert_policy = AlertPolicy(config.alert_cooldown_seconds)
    feedback = AudioFeedback(enabled=config.audio_enabled, cooldown_seconds=0.0, voice=config.voice)
    metrics = MetricsCollector(warmup_frames=config.warmup_frames)

    frames = raw_seen = stable_seen = alert_events = spoken_alerts = 0
    cold_inference_seconds = 0.0
    started_at = time.monotonic()
    display_fps = 0.0

    try:
        with source:
            pending_frame_started = time.perf_counter()
            capture_started = time.perf_counter()
            pending_frame = source.read()
            pending_capture_seconds = time.perf_counter() - capture_started

            pending_detections: list[Detection] | None = None
            pending_inference_seconds = 0.0
            if pending_frame is not None:
                inference_started = time.perf_counter()
                pending_detections = detector.warmup(pending_frame)
                pending_inference_seconds = time.perf_counter() - inference_started
                cold_inference_seconds = pending_inference_seconds

            while pending_frame is not None and pending_detections is not None:
                frame = pending_frame
                detections = pending_detections
                frame_started = pending_frame_started
                capture_seconds = pending_capture_seconds
                inference_seconds = pending_inference_seconds
                pending_frame = pending_detections = None

                decision_started = time.perf_counter()
                frame_decision = evaluate_detections(
                    detections,
                    temporal_filter=temporal_filter,
                    risk_engine=engine,
                    alert_policy=alert_policy,
                    commit_alert=False,
                )
                decision_seconds = time.perf_counter() - decision_started

                frames += 1
                raw_seen += len(detections)
                stable_seen += len(frame_decision.stable_detections)

                alert = frame_decision.alert
                if alert is not None and alert.should_emit:
                    force = alert.reason in {"risk_upgraded", "direction_changed"}
                    spoken = config.audio_enabled and feedback.notify(
                        alert.message,
                        force=force,
                    )
                    delivered = not config.audio_enabled or spoken
                    if delivered:
                        assert frame_decision.alert_track_key is not None
                        alert_policy.commit(frame_decision.alert_track_key, alert)
                        alert_events += 1
                    if spoken:
                        spoken_alerts += 1

                render_started = time.perf_counter()
                should_stop = False
                if config.display:
                    _draw_overlay(
                        cv2,
                        frame,
                        frame_decision.stable_detections,
                        frame_decision.assessments,
                        display_fps,
                        inference_seconds * 1000.0,
                        raw_count=len(detections),
                    )
                    cv2.imshow("Smart White Cane Lab", frame)
                    key = cv2.waitKey(1) & 0xFF
                    should_stop = key in (ord("q"), 27)
                render_seconds = time.perf_counter() - render_started
                total_seconds = time.perf_counter() - frame_started

                metrics.add(
                    capture_seconds=capture_seconds,
                    inference_seconds=inference_seconds,
                    decision_seconds=decision_seconds,
                    render_seconds=render_seconds,
                    total_seconds=total_seconds,
                )
                if frames > config.warmup_frames:
                    instantaneous_fps = 1.0 / max(total_seconds, 1e-9)
                    display_fps = (
                        instantaneous_fps
                        if frames == config.warmup_frames + 1 or display_fps == 0.0
                        else (0.9 * display_fps + 0.1 * instantaneous_fps)
                    )
                if should_stop or (config.max_frames is not None and frames >= config.max_frames):
                    break

                pending_frame_started = time.perf_counter()
                capture_started = time.perf_counter()
                try:
                    pending_frame = source.read()
                except CameraReadError:
                    break
                pending_capture_seconds = time.perf_counter() - capture_started

                inference_started = time.perf_counter()
                pending_detections = detector(pending_frame)
                pending_inference_seconds = time.perf_counter() - inference_started
    finally:
        source.release()
        if config.display:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 0.0)
    return RunSummary(
        frames=frames,
        detections=raw_seen,
        alerts=alert_events,
        elapsed_seconds=elapsed,
        stable_detections=stable_seen,
        spoken_alerts=spoken_alerts,
        cold_inference_seconds=cold_inference_seconds,
        device=str(detector.device),
        metrics=metrics.summary(),
    )


def _build_temporal_filter(config: RuntimeConfig) -> TemporalFilter:
    return TemporalFilter(
        TemporalFilterConfig(
            confirmation_window=config.confirmation_window,
            minimum_confirmations=config.minimum_confirmations,
            high_risk_area_ratio=config.high_risk_area_ratio,
            high_risk_consecutive_frames=config.high_risk_consecutive_frames,
            max_missing_frames=config.max_missing_frames,
            direction_hysteresis=config.direction_hysteresis,
        )
    )


def _build_risk_engine(config: RuntimeConfig) -> RiskEngine:
    return RiskEngine(
        RiskRules(
            minimum_confidence=config.confidence,
            default_area_thresholds=AreaThresholds(high=config.high_risk_area_ratio),
        )
    )


def _most_urgent_index(assessments: Sequence[RiskAssessment]) -> int | None:
    order = _urgency_order(assessments)
    return order[0] if order else None


def _urgency_order(assessments: Sequence[RiskAssessment]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(assessments)),
            key=lambda index: (
                assessments[index].rank,
                assessments[index].direction == "center",
                assessments[index].area_ratio,
                assessments[index].confidence,
            ),
            reverse=True,
        )
    )


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is not installed; run `uv sync` first") from error
    return cv2


def _draw_overlay(
    cv2: Any,
    frame: Any,
    stable_detections: Sequence[StableDetection],
    assessments: Sequence[RiskAssessment],
    fps: float,
    inference_ms: float,
    *,
    raw_count: int,
) -> None:
    colors = {
        "low": (160, 160, 160),
        "medium": (0, 170, 255),
        "high": (30, 30, 230),
    }
    for stable, assessment in zip(stable_detections, assessments, strict=True):
        detection = stable.detection
        x_min, y_min, x_max, y_max = (int(value) for value in detection.bbox)
        color = colors[assessment.level]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        stale = f" stale:{stable.missed_frames}" if stable.missed_frames else ""
        text = (
            f"{detection.label} {detection.confidence:.2f} | "
            f"{assessment.direction} {assessment.level} | {stable.track_key}{stale}"
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

    fps_status = f"FPS {fps:.1f}" if fps > 0.0 else "FPS warming up"
    status = (
        f"{fps_status} | infer {inference_ms:.1f} ms | "
        f"stable {len(stable_detections)} / raw {raw_count} | q / Esc to quit"
    )
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
