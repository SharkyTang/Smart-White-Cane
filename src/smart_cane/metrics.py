"""Deterministic, dependency-free performance metrics for the live frame loop.

Durations are recorded in seconds because that is what ``time.monotonic`` and
``time.perf_counter`` return.  Summaries expose milliseconds for readability.
The configured leading warmup frames are counted but excluded from throughput
and stage distributions so model initialisation does not skew steady-state
results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import fmean, median
from types import MappingProxyType


STAGE_NAMES = ("capture", "inference", "decision", "render", "total")


def _validated_duration(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite, non-negative number of seconds")
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite, non-negative number of seconds") from error
    if not isfinite(duration) or duration < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number of seconds")
    return duration


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Timing measurements for one processed frame, expressed in seconds."""

    capture_seconds: float
    inference_seconds: float
    decision_seconds: float
    render_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        for field_name in (
            "capture_seconds",
            "inference_seconds",
            "decision_seconds",
            "render_seconds",
            "total_seconds",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_duration(field_name, getattr(self, field_name)),
            )


@dataclass(frozen=True, slots=True)
class StageStatistics:
    """Steady-state distribution for one stage, expressed in milliseconds."""

    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Aggregate counts, throughput, and steady-state stage distributions."""

    total_frames: int
    warmup_frames: int
    steady_frames: int
    steady_average_fps: float
    stages: Mapping[str, StageStatistics]

    def stage(self, name: str) -> StageStatistics:
        """Return statistics for a named stage such as ``"inference"``."""

        try:
            return self.stages[name]
        except KeyError as error:
            expected = ", ".join(STAGE_NAMES)
            raise KeyError(f"unknown stage {name!r}; expected one of: {expected}") from error


class MetricsCollector:
    """Collect per-frame timings and summarize steady-state performance."""

    def __init__(self, warmup_frames: int = 0) -> None:
        if (
            isinstance(warmup_frames, bool)
            or not isinstance(warmup_frames, int)
            or warmup_frames < 0
        ):
            raise ValueError("warmup_frames must be a non-negative integer")
        self.warmup_frames = warmup_frames
        self._frames: list[FrameMetrics] = []

    @property
    def frame_count(self) -> int:
        """Number of frames currently recorded."""

        return len(self._frames)

    def add(
        self,
        metrics: FrameMetrics | None = None,
        *,
        capture_seconds: float | None = None,
        inference_seconds: float | None = None,
        decision_seconds: float | None = None,
        render_seconds: float | None = None,
        total_seconds: float | None = None,
    ) -> FrameMetrics:
        """Record one frame and return the stored :class:`FrameMetrics`.

        Callers can pass an existing ``FrameMetrics`` instance or provide all
        five durations as keyword arguments.  Mixing the two forms is rejected
        to keep integration mistakes visible.
        """

        durations = {
            "capture_seconds": capture_seconds,
            "inference_seconds": inference_seconds,
            "decision_seconds": decision_seconds,
            "render_seconds": render_seconds,
            "total_seconds": total_seconds,
        }

        if metrics is not None:
            if not isinstance(metrics, FrameMetrics):
                raise TypeError("metrics must be a FrameMetrics instance")
            if any(value is not None for value in durations.values()):
                raise TypeError("pass either metrics or duration keywords, not both")
            frame = metrics
        else:
            missing = [name for name, value in durations.items() if value is None]
            if missing:
                raise TypeError(f"missing required duration(s): {', '.join(missing)}")
            frame = FrameMetrics(
                capture_seconds=durations["capture_seconds"],  # type: ignore[arg-type]
                inference_seconds=durations["inference_seconds"],  # type: ignore[arg-type]
                decision_seconds=durations["decision_seconds"],  # type: ignore[arg-type]
                render_seconds=durations["render_seconds"],  # type: ignore[arg-type]
                total_seconds=durations["total_seconds"],  # type: ignore[arg-type]
            )

        self._frames.append(frame)
        return frame

    def summary(self) -> MetricsSummary:
        """Return a snapshot summary; all distribution values exclude warmup."""

        total_frames = len(self._frames)
        warmup_frames = min(self.warmup_frames, total_frames)
        steady = self._frames[warmup_frames:]
        steady_frames = len(steady)

        total_seconds = sum(frame.total_seconds for frame in steady)
        steady_average_fps = (
            steady_frames / total_seconds if steady_frames and total_seconds > 0.0 else 0.0
        )

        stages = {
            name: _summarize_stage([getattr(frame, f"{name}_seconds") for frame in steady])
            for name in STAGE_NAMES
        }
        return MetricsSummary(
            total_frames=total_frames,
            warmup_frames=warmup_frames,
            steady_frames=steady_frames,
            steady_average_fps=steady_average_fps,
            stages=MappingProxyType(stages),
        )


def _summarize_stage(seconds: list[float]) -> StageStatistics:
    if not seconds:
        return StageStatistics(samples=0, mean_ms=0.0, median_ms=0.0, p95_ms=0.0)

    milliseconds = sorted(value * 1000.0 for value in seconds)
    return StageStatistics(
        samples=len(milliseconds),
        mean_ms=fmean(milliseconds),
        median_ms=median(milliseconds),
        p95_ms=_nearest_rank_p95(milliseconds),
    )


def _nearest_rank_p95(sorted_values: list[float]) -> float:
    """Return deterministic nearest-rank p95 for a non-empty sorted sample.

    With ``n`` samples, nearest-rank p95 selects the observed value at the
    one-based rank ``ceil(0.95 * n)``.  It performs no interpolation, so the
    result is always an actual observation and behaves predictably for small
    samples.  The equivalent zero-based index is ``ceil(0.95 * n) - 1``.
    """

    return sorted_values[ceil(0.95 * len(sorted_values)) - 1]
