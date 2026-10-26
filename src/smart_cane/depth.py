"""Hardware-independent LiDAR depth sampling and track-level smoothing.

The module keeps registered depth data separate from detector output. A
``DepthFrame`` is expected to be aligned with the RGB image that produced the
YOLO detections; ``DistanceFusion`` maps stable boxes into that depth grid and
returns explicit validity and staleness metadata for the risk layer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor, isfinite
from statistics import median
from typing import Any, Literal, TypeAlias

from smart_cane.decision.temporal_filter import Direction, StableDetection
from smart_cane.models import Detection


DepthValues: TypeAlias = tuple[tuple[float, ...], ...]
ConfidenceValues: TypeAlias = tuple[tuple[int, ...], ...]
DistanceStatus = Literal[
    "valid",
    "stale",
    "insufficient_samples",
    "low_coverage",
    "unavailable",
]
DepthSource = Literal["raw_scene_depth", "smoothed_scene_depth"]
DistanceQuality = Literal["good", "degraded", "unavailable"]


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """A registered metric-depth image for one RGB frame.

    Confidence values follow ARKit's ordered convention: ``0`` is low, ``1``
    is medium and ``2`` is high. Non-finite and out-of-range distances are
    retained here because rejection belongs to ``DistanceFusion``.
    """

    distances_m: DepthValues
    confidence: ConfidenceValues
    frame_width: int
    frame_height: int
    frame_id: str | int | None = None
    captured_at: float | None = None
    depth_source: DepthSource = "raw_scene_depth"

    def __post_init__(self) -> None:
        distances = _normalize_distances(self.distances_m)
        confidence = _normalize_confidence(self.confidence)
        if len(distances) != len(confidence) or len(distances[0]) != len(confidence[0]):
            raise ValueError("distance and confidence grids must have identical dimensions")
        _require_positive_int("frame_width", self.frame_width)
        _require_positive_int("frame_height", self.frame_height)
        if self.captured_at is not None:
            captured_at = _finite_float("captured_at", self.captured_at)
            if captured_at < 0:
                raise ValueError("captured_at cannot be negative")
            object.__setattr__(self, "captured_at", captured_at)
        if self.frame_id is not None and (
            isinstance(self.frame_id, bool) or not isinstance(self.frame_id, (str, int))
        ):
            raise TypeError("frame_id must be a string, integer, or None")
        if self.depth_source not in ("raw_scene_depth", "smoothed_scene_depth"):
            raise ValueError("depth_source must be raw_scene_depth or smoothed_scene_depth")

        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "confidence", confidence)

    @property
    def depth_width(self) -> int:
        return len(self.distances_m[0])

    @property
    def depth_height(self) -> int:
        return len(self.distances_m)


@dataclass(frozen=True, slots=True)
class DistanceEstimate:
    """Distance evidence for one stable track in the current update.

    ``percentile_20_m`` is the current box candidate. ``candidate_median_m``
    and ``candidate_center_median_m`` expose whole-ROI and central-ROI
    diagnostics, while ``distance_m`` is the median of the track's latest three
    valid percentile candidates. A stale estimate may retain the last smoothed
    value for display, but ``is_valid`` is false and consumers must not use it
    for metric risk decisions.
    """

    distance_m: float | None
    percentile_20_m: float | None
    candidate_median_m: float | None
    sample_count: int
    roi_pixel_count: int
    coverage_ratio: float
    is_valid: bool
    is_stale: bool
    missed_frames: int
    frame_id: str | int | None = None
    status: DistanceStatus | None = None
    candidate_center_median_m: float | None = None
    depth_source: DepthSource | None = None
    quality: DistanceQuality | None = None
    freshness_frames: int | None = None
    candidate_status: DistanceStatus | None = None

    def __post_init__(self) -> None:
        for name in (
            "distance_m",
            "percentile_20_m",
            "candidate_median_m",
            "candidate_center_median_m",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = _finite_float(name, value)
                if normalized <= 0:
                    raise ValueError(f"{name} must be positive when present")
                object.__setattr__(self, name, normalized)
        _require_non_negative_int("sample_count", self.sample_count)
        _require_non_negative_int("roi_pixel_count", self.roi_pixel_count)
        _require_non_negative_int("missed_frames", self.missed_frames)
        coverage = _finite_float("coverage_ratio", self.coverage_ratio)
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage_ratio must be between 0 and 1")
        object.__setattr__(self, "coverage_ratio", coverage)
        if self.sample_count > self.roi_pixel_count:
            raise ValueError("sample_count cannot exceed roi_pixel_count")
        if self.is_valid:
            if self.distance_m is None or self.percentile_20_m is None:
                raise ValueError("a valid estimate requires metric distance values")
            if self.is_stale or self.missed_frames:
                raise ValueError("a valid estimate cannot be stale or missed")
        if self.is_stale and (self.is_valid or self.missed_frames <= 0):
            raise ValueError("a stale estimate must be invalid and have missed frames")
        status = self.status
        if status is None:
            status = "valid" if self.is_valid else "stale" if self.is_stale else "unavailable"
            object.__setattr__(self, "status", status)
        if status not in (
            "valid",
            "stale",
            "insufficient_samples",
            "low_coverage",
            "unavailable",
        ):
            raise ValueError("status is not a supported distance status")
        if self.is_valid and status != "valid":
            raise ValueError("a valid estimate must have valid status")
        if self.is_stale and status != "stale":
            raise ValueError("a stale estimate must have stale status")
        if status == "valid" and not self.is_valid:
            raise ValueError("valid status requires a valid estimate")
        if status == "stale" and not self.is_stale:
            raise ValueError("stale status requires a stale estimate")
        if self.depth_source not in (None, "raw_scene_depth", "smoothed_scene_depth"):
            raise ValueError("depth_source must be raw_scene_depth, smoothed_scene_depth, or None")
        quality = self.quality
        if quality is None:
            if self.is_valid:
                quality = "good"
            elif self.is_stale or status in ("insufficient_samples", "low_coverage"):
                quality = "degraded"
            else:
                quality = "unavailable"
            object.__setattr__(self, "quality", quality)
        if quality not in ("good", "degraded", "unavailable"):
            raise ValueError("quality must be good, degraded, or unavailable")
        if self.is_valid and quality != "good":
            raise ValueError("a valid estimate must have good quality")
        if self.is_stale and quality != "degraded":
            raise ValueError("a stale estimate must have degraded quality")
        freshness = self.freshness_frames
        if freshness is None:
            freshness = (
                0
                if self.is_valid
                or (not self.is_stale and status in ("insufficient_samples", "low_coverage"))
                else self.missed_frames
            )
            object.__setattr__(self, "freshness_frames", freshness)
        _require_non_negative_int("freshness_frames", freshness)
        if self.is_valid and freshness != 0:
            raise ValueError("a valid current estimate must have zero freshness_frames")
        if self.is_stale and freshness != self.missed_frames:
            raise ValueError("a stale estimate freshness must equal missed_frames")
        candidate_status = self.candidate_status
        if candidate_status is None:
            candidate_status = "valid" if self.is_valid else status
            object.__setattr__(self, "candidate_status", candidate_status)
        if candidate_status not in (
            "valid",
            "stale",
            "insufficient_samples",
            "low_coverage",
            "unavailable",
        ):
            raise ValueError("candidate_status is not a supported distance status")

    @property
    def usable(self) -> bool:
        """Whether the metric value is safe to use for the current risk update."""

        return self.is_valid and not self.is_stale and self.distance_m is not None


@dataclass(frozen=True, slots=True)
class FusedDetection:
    """A stable YOLO track paired with LiDAR distance evidence."""

    stable: StableDetection
    distance: DistanceEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.stable, StableDetection):
            raise TypeError("stable must be a StableDetection")
        if not isinstance(self.distance, DistanceEstimate):
            raise TypeError("distance must be a DistanceEstimate")

    @property
    def detection(self) -> Detection:
        return self.stable.detection

    @property
    def track_key(self) -> str:
        return self.stable.track_key

    @property
    def direction(self) -> Direction:
        return self.stable.direction


@dataclass(slots=True)
class _DistanceTrack:
    recent_candidates: deque[float]
    missed_frames: int = 0
    depth_source: DepthSource | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    percentile_20_m: float | None
    median_m: float | None
    center_median_m: float | None
    sample_count: int
    roi_pixel_count: int
    coverage_ratio: float
    status: DistanceStatus


class DistanceFusion:
    """Sample aligned depth inside stable boxes and smooth it by track.

    The default policy insets each box by 15%, accepts only medium/high
    confidence pixels in ``0 < d <= 5m``, requires at least 12 samples and 20%
    coverage, uses the 20th percentile as the per-frame candidate, and reports
    the median of the latest three candidates as the current distance.
    """

    def __init__(
        self,
        *,
        inset_ratio: float = 0.15,
        minimum_confidence: int = 1,
        maximum_distance_m: float = 5.0,
        minimum_samples: int = 12,
        minimum_coverage: float = 0.20,
        history_size: int = 3,
        max_stale_frames: int = 2,
    ) -> None:
        if not _is_finite_number(inset_ratio) or not 0.0 <= float(inset_ratio) < 0.5:
            raise ValueError("inset_ratio must be at least 0 and less than 0.5")
        if isinstance(minimum_confidence, bool) or minimum_confidence not in (1, 2):
            raise ValueError("minimum_confidence must be medium (1) or high (2)")
        maximum_distance = _finite_float("maximum_distance_m", maximum_distance_m)
        if maximum_distance <= 0:
            raise ValueError("maximum_distance_m must be positive")
        _require_positive_int("minimum_samples", minimum_samples)
        if not _is_finite_number(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between 0 and 1")
        _require_positive_int("history_size", history_size)
        _require_non_negative_int("max_stale_frames", max_stale_frames)

        self.inset_ratio = float(inset_ratio)
        self.minimum_confidence = minimum_confidence
        self.maximum_distance_m = maximum_distance
        self.minimum_samples = minimum_samples
        self.minimum_coverage = float(minimum_coverage)
        self.history_size = history_size
        self.max_stale_frames = max_stale_frames
        self._tracks: dict[str, _DistanceTrack] = {}

    @property
    def tracked_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        stable_detections: Iterable[StableDetection],
        depth_frame: DepthFrame | None,
    ) -> tuple[FusedDetection, ...]:
        """Fuse one ordered set of stable detections with its registered depth frame."""

        stable = tuple(stable_detections)
        if not all(isinstance(item, StableDetection) for item in stable):
            raise TypeError("stable_detections must contain only StableDetection instances")
        if depth_frame is not None and not isinstance(depth_frame, DepthFrame):
            raise TypeError("depth_frame must be a DepthFrame or None")
        track_keys = [item.track_key for item in stable]
        if len(track_keys) != len(set(track_keys)):
            raise ValueError("stable_detections cannot contain duplicate track keys")

        active = set(track_keys)
        for track_key in self._tracks.keys() - active:
            del self._tracks[track_key]

        fused: list[FusedDetection] = []
        for item in stable:
            if depth_frame is None or item.missed_frames > 0:
                estimate = self._miss(item.track_key, frame_id=_frame_id(depth_frame))
            else:
                self._validate_alignment(item, depth_frame)
                candidate = self._sample(item, depth_frame)
                if self._candidate_is_valid(candidate):
                    estimate = self._accept(
                        item.track_key,
                        candidate,
                        depth_frame.frame_id,
                        depth_frame.depth_source,
                    )
                else:
                    estimate = self._miss(
                        item.track_key,
                        frame_id=depth_frame.frame_id,
                        candidate=candidate,
                        depth_source=depth_frame.depth_source,
                    )
            fused.append(FusedDetection(item, estimate))
        return tuple(fused)

    def _candidate_is_valid(self, candidate: _Candidate) -> bool:
        return (
            candidate.percentile_20_m is not None
            and candidate.sample_count >= self.minimum_samples
            and candidate.coverage_ratio >= self.minimum_coverage
        )

    def _accept(
        self,
        track_key: str,
        candidate: _Candidate,
        frame_id: str | int | None,
        depth_source: DepthSource,
    ) -> DistanceEstimate:
        assert candidate.percentile_20_m is not None
        track = self._tracks.get(track_key)
        if track is None:
            track = _DistanceTrack(deque(maxlen=self.history_size))
            self._tracks[track_key] = track
        track.recent_candidates.append(candidate.percentile_20_m)
        track.missed_frames = 0
        track.depth_source = depth_source
        return DistanceEstimate(
            distance_m=float(median(track.recent_candidates)),
            percentile_20_m=candidate.percentile_20_m,
            candidate_median_m=candidate.median_m,
            sample_count=candidate.sample_count,
            roi_pixel_count=candidate.roi_pixel_count,
            coverage_ratio=candidate.coverage_ratio,
            is_valid=True,
            is_stale=False,
            missed_frames=0,
            frame_id=frame_id,
            status="valid",
            candidate_center_median_m=candidate.center_median_m,
            depth_source=depth_source,
            quality="good",
            freshness_frames=0,
            candidate_status=candidate.status,
        )

    def _miss(
        self,
        track_key: str,
        *,
        frame_id: str | int | None,
        candidate: _Candidate | None = None,
        depth_source: DepthSource | None = None,
    ) -> DistanceEstimate:
        track = self._tracks.get(track_key)
        if track is None:
            track = _DistanceTrack(deque(maxlen=self.history_size))
            self._tracks[track_key] = track
        track.missed_frames += 1
        retained = (
            float(median(track.recent_candidates))
            if track.recent_candidates and track.missed_frames <= self.max_stale_frames
            else None
        )
        if track.missed_frames > self.max_stale_frames:
            track.recent_candidates.clear()
        has_current_candidate = candidate is not None
        candidate = candidate or _Candidate(None, None, None, 0, 0, 0.0, "unavailable")
        status: DistanceStatus = "stale" if retained is not None else candidate.status
        estimate_source = track.depth_source if retained is not None else depth_source
        quality: DistanceQuality = (
            "degraded"
            if retained is not None or candidate.status in ("insufficient_samples", "low_coverage")
            else "unavailable"
        )
        if retained is None and track.missed_frames > self.max_stale_frames:
            track.depth_source = None
        return DistanceEstimate(
            distance_m=retained,
            percentile_20_m=candidate.percentile_20_m,
            candidate_median_m=candidate.median_m,
            sample_count=candidate.sample_count,
            roi_pixel_count=candidate.roi_pixel_count,
            coverage_ratio=candidate.coverage_ratio,
            is_valid=False,
            is_stale=retained is not None,
            missed_frames=track.missed_frames,
            frame_id=frame_id,
            status=status,
            candidate_center_median_m=candidate.center_median_m,
            depth_source=estimate_source,
            quality=quality,
            freshness_frames=(
                track.missed_frames if retained is not None or not has_current_candidate else 0
            ),
            candidate_status=candidate.status,
        )

    def _sample(self, stable: StableDetection, frame: DepthFrame) -> _Candidate:
        detection = stable.detection
        x_min, y_min, x_max, y_max = detection.bbox
        x_min = min(float(detection.frame_width), max(0.0, x_min))
        y_min = min(float(detection.frame_height), max(0.0, y_min))
        x_max = min(float(detection.frame_width), max(0.0, x_max))
        y_max = min(float(detection.frame_height), max(0.0, y_max))
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        if width <= 0.0 or height <= 0.0:
            return _Candidate(None, None, None, 0, 0, 0.0, "unavailable")

        x_min += width * self.inset_ratio
        x_max -= width * self.inset_ratio
        y_min += height * self.inset_ratio
        y_max -= height * self.inset_ratio
        left = max(0, min(frame.depth_width, floor(x_min / frame.frame_width * frame.depth_width)))
        right = max(0, min(frame.depth_width, ceil(x_max / frame.frame_width * frame.depth_width)))
        top = max(
            0, min(frame.depth_height, floor(y_min / frame.frame_height * frame.depth_height))
        )
        bottom = max(
            0,
            min(frame.depth_height, ceil(y_max / frame.frame_height * frame.depth_height)),
        )
        roi_pixel_count = max(0, right - left) * max(0, bottom - top)
        if roi_pixel_count == 0:
            return _Candidate(None, None, None, 0, 0, 0.0, "unavailable")

        samples: list[float] = []
        center_samples: list[float] = []
        center_left = left + (right - left) // 4
        center_right = right - (right - left) // 4
        center_top = top + (bottom - top) // 4
        center_bottom = bottom - (bottom - top) // 4
        for row in range(top, bottom):
            for column in range(left, right):
                value = frame.distances_m[row][column]
                if (
                    frame.confidence[row][column] >= self.minimum_confidence
                    and isfinite(value)
                    and 0.0 < value <= self.maximum_distance_m
                ):
                    samples.append(value)
                    if center_left <= column < center_right and center_top <= row < center_bottom:
                        center_samples.append(value)
        coverage = len(samples) / roi_pixel_count
        if not samples:
            return _Candidate(
                None,
                None,
                None,
                0,
                roi_pixel_count,
                coverage,
                "insufficient_samples",
            )
        samples.sort()
        percentile_index = max(0, ceil(len(samples) * 0.20) - 1)
        status: DistanceStatus
        if len(samples) < self.minimum_samples:
            status = "insufficient_samples"
        elif coverage < self.minimum_coverage:
            status = "low_coverage"
        else:
            status = "valid"
        return _Candidate(
            percentile_20_m=samples[percentile_index],
            median_m=float(median(samples)),
            center_median_m=float(median(center_samples)) if center_samples else None,
            sample_count=len(samples),
            roi_pixel_count=roi_pixel_count,
            coverage_ratio=coverage,
            status=status,
        )

    @staticmethod
    def _validate_alignment(stable: StableDetection, frame: DepthFrame) -> None:
        detection = stable.detection
        if (detection.frame_width, detection.frame_height) != (
            frame.frame_width,
            frame.frame_height,
        ):
            raise ValueError("depth frame must be registered to the detection frame dimensions")


def _normalize_distances(values: Any) -> DepthValues:
    rows = _rows(values, "distances_m")
    normalized: list[tuple[float, ...]] = []
    for row in rows:
        converted: list[float] = []
        for value in row:
            if isinstance(value, bool):
                raise TypeError("distances_m must contain numeric values")
            try:
                converted.append(float(value))
            except (TypeError, ValueError) as error:
                raise TypeError("distances_m must contain numeric values") from error
        normalized.append(tuple(converted))
    return tuple(normalized)


def _normalize_confidence(values: Any) -> ConfidenceValues:
    rows = _rows(values, "confidence")
    normalized: list[tuple[int, ...]] = []
    for row in rows:
        converted: list[int] = []
        for value in row:
            if isinstance(value, bool):
                raise TypeError("confidence must contain integer values from 0 to 2")
            try:
                integer = int(value)
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError("confidence must contain integer values from 0 to 2") from error
            if numeric != integer or integer not in (0, 1, 2):
                raise ValueError("confidence values must be low (0), medium (1), or high (2)")
            converted.append(integer)
        normalized.append(tuple(converted))
    return tuple(normalized)


def _rows(values: Any, name: str) -> tuple[tuple[Any, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a non-empty rectangular grid")
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise TypeError(f"{name} must be a non-empty rectangular grid") from error
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be a non-empty rectangular grid")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _frame_id(frame: DepthFrame | None) -> str | int | None:
    return frame.frame_id if frame is not None else None


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
