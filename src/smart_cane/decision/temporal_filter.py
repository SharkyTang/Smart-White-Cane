"""Lightweight temporal stability for frame-by-frame object detections.

The filter intentionally uses only the Python standard library.  It is not a
full object tracker: detections with the same label are associated by bounding
box IoU, then a small amount of history removes one-frame false positives and
keeps confirmed targets alive through short detector dropouts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

from smart_cane.models import BoundingBox, Detection


Direction = Literal["left", "center", "right"]


@dataclass(frozen=True, slots=True)
class TemporalFilterConfig:
    """All policy values used by :class:`TemporalFilter`."""

    iou_threshold: float = 0.30
    confirmation_window: int = 5
    minimum_confirmations: int = 3
    high_risk_area_ratio: float = 0.18
    high_risk_consecutive_frames: int = 2
    max_missing_frames: int = 3
    bbox_smoothing: float = 0.40
    left_boundary: float = 0.35
    right_boundary: float = 0.65
    direction_hysteresis: float = 0.05

    def __post_init__(self) -> None:
        self._validate_ratio("iou_threshold", self.iou_threshold)
        self._validate_positive_int("confirmation_window", self.confirmation_window)
        self._validate_positive_int("minimum_confirmations", self.minimum_confirmations)
        if self.minimum_confirmations > self.confirmation_window:
            raise ValueError("minimum_confirmations cannot exceed confirmation_window")

        self._validate_ratio("high_risk_area_ratio", self.high_risk_area_ratio)
        self._validate_positive_int(
            "high_risk_consecutive_frames", self.high_risk_consecutive_frames
        )
        self._validate_positive_int("max_missing_frames", self.max_missing_frames)

        if not self._is_finite_number(self.bbox_smoothing) or not 0.0 < self.bbox_smoothing <= 1.0:
            raise ValueError("bbox_smoothing must be greater than 0 and at most 1")
        if (
            not self._is_finite_number(self.left_boundary)
            or not self._is_finite_number(self.right_boundary)
            or not 0.0 < self.left_boundary < self.right_boundary < 1.0
        ):
            raise ValueError("direction boundaries must satisfy 0 < left < right < 1")
        if not self._is_finite_number(self.direction_hysteresis) or self.direction_hysteresis < 0.0:
            raise ValueError("direction_hysteresis cannot be negative")

        maximum_hysteresis = min(
            self.left_boundary,
            1.0 - self.right_boundary,
            (self.right_boundary - self.left_boundary) / 2.0,
        )
        if self.direction_hysteresis >= maximum_hysteresis:
            raise ValueError("direction_hysteresis is too large for the direction boundaries")

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        if not TemporalFilterConfig._is_finite_number(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


@dataclass(frozen=True, slots=True)
class StableDetection:
    """A confirmed target ready for risk assessment or sensor fusion.

    ``confirmed_frames`` counts matched observations, while ``missed_frames``
    exposes how stale a temporarily retained target is.
    """

    detection: Detection
    track_key: str
    direction: Direction
    confirmed_frames: int
    missed_frames: int = 0


@dataclass(slots=True)
class _Track:
    track_key: str
    detection: Detection
    direction: Direction
    observations: deque[bool]
    confirmed_frames: int = 1
    consecutive_high_risk_frames: int = 0
    missed_frames: int = 0
    is_confirmed: bool = False


class TemporalFilter:
    """Stabilise detections across calls to :meth:`update`."""

    def __init__(self, config: TemporalFilterConfig | None = None) -> None:
        self.config = config or TemporalFilterConfig()
        self._tracks: dict[str, _Track] = {}
        self._next_track_id = 1

    @property
    def track_count(self) -> int:
        """Number of active tracks, including tracks not confirmed yet."""

        return len(self._tracks)

    def reset(self) -> None:
        """Forget all history and restart deterministic track numbering."""

        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections: Iterable[Detection]) -> tuple[StableDetection, ...]:
        """Consume one frame of detections and return confirmed active targets.

        Confirmed targets remain in the result during a short detector dropout.
        They are removed as soon as ``max_missing_frames`` consecutive updates
        contain no matching detection.
        """

        current = tuple(detections)
        if not all(isinstance(item, Detection) for item in current):
            raise TypeError("detections must contain only Detection instances")
        self._validate_frame_dimensions(current)

        matches, unmatched_track_keys, unmatched_detection_indexes = self._associate(current)

        for track_key, detection_index in matches:
            self._update_track(self._tracks[track_key], current[detection_index])

        expired_track_keys: list[str] = []
        for track_key in unmatched_track_keys:
            track = self._tracks[track_key]
            track.observations.append(False)
            track.consecutive_high_risk_frames = 0
            track.missed_frames += 1
            if track.missed_frames >= self.config.max_missing_frames:
                expired_track_keys.append(track_key)

        for track_key in expired_track_keys:
            del self._tracks[track_key]

        for detection_index in unmatched_detection_indexes:
            self._create_track(current[detection_index])

        return tuple(
            StableDetection(
                detection=track.detection,
                track_key=track.track_key,
                direction=track.direction,
                confirmed_frames=track.confirmed_frames,
                missed_frames=track.missed_frames,
            )
            for track in self._tracks.values()
            if track.is_confirmed
        )

    def _associate(
        self, detections: tuple[Detection, ...]
    ) -> tuple[list[tuple[str, int]], list[str], list[int]]:
        candidates: list[tuple[float, int, int, str]] = []
        track_keys = list(self._tracks)

        for track_index, track_key in enumerate(track_keys):
            previous = self._tracks[track_key].detection
            for detection_index, detection in enumerate(detections):
                if not self._same_detection_space(previous, detection):
                    continue
                overlap = _bbox_iou(previous.bbox, detection.bbox)
                if overlap >= self.config.iou_threshold:
                    candidates.append((-overlap, track_index, detection_index, track_key))

        candidates.sort()
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[str, int]] = []
        for _, _, detection_index, track_key in candidates:
            if track_key in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_key)
            matched_detections.add(detection_index)
            matches.append((track_key, detection_index))

        unmatched_track_keys = [key for key in track_keys if key not in matched_tracks]
        unmatched_detection_indexes = [
            index for index in range(len(detections)) if index not in matched_detections
        ]
        return matches, unmatched_track_keys, unmatched_detection_indexes

    @staticmethod
    def _same_detection_space(previous: Detection, current: Detection) -> bool:
        return (
            previous.label.casefold() == current.label.casefold()
            and previous.frame_width == current.frame_width
            and previous.frame_height == current.frame_height
        )

    def _create_track(self, detection: Detection) -> None:
        track_key = f"{detection.label.casefold()}:{self._next_track_id}"
        self._next_track_id += 1
        is_high_risk = detection.area_ratio >= self.config.high_risk_area_ratio
        observations = deque((True,), maxlen=self.config.confirmation_window)
        track = _Track(
            track_key=track_key,
            detection=detection,
            direction=self._initial_direction(detection),
            observations=observations,
            consecutive_high_risk_frames=int(is_high_risk),
        )
        self._refresh_confirmation(track)
        self._tracks[track_key] = track

    def _update_track(self, track: _Track, detection: Detection) -> None:
        smoothed_bbox = _smooth_bbox(
            previous=track.detection.bbox,
            current=detection.bbox,
            current_weight=self.config.bbox_smoothing,
        )
        smoothed_detection = Detection(
            label=detection.label,
            confidence=detection.confidence,
            bbox=smoothed_bbox,
            frame_width=detection.frame_width,
            frame_height=detection.frame_height,
        )

        track.detection = smoothed_detection
        track.direction = self._direction_with_hysteresis(smoothed_detection, track.direction)
        track.observations.append(True)
        track.confirmed_frames += 1
        track.missed_frames = 0
        if detection.area_ratio >= self.config.high_risk_area_ratio:
            track.consecutive_high_risk_frames += 1
        else:
            track.consecutive_high_risk_frames = 0
        self._refresh_confirmation(track)

    def _refresh_confirmation(self, track: _Track) -> None:
        if track.is_confirmed:
            return
        ordinary_confirmation = sum(track.observations) >= self.config.minimum_confirmations
        high_risk_confirmation = (
            track.consecutive_high_risk_frames >= self.config.high_risk_consecutive_frames
        )
        track.is_confirmed = ordinary_confirmation or high_risk_confirmation

    def _initial_direction(self, detection: Detection) -> Direction:
        normalized_x, _ = detection.normalized_center
        if normalized_x < self.config.left_boundary:
            return "left"
        if normalized_x > self.config.right_boundary:
            return "right"
        return "center"

    def _direction_with_hysteresis(self, detection: Detection, previous: Direction) -> Direction:
        normalized_x, _ = detection.normalized_center
        left = self.config.left_boundary
        right = self.config.right_boundary
        hysteresis = self.config.direction_hysteresis

        if previous == "left":
            if normalized_x > right + hysteresis:
                return "right"
            if normalized_x >= left + hysteresis:
                return "center"
            return "left"

        if previous == "right":
            if normalized_x < left - hysteresis:
                return "left"
            if normalized_x <= right - hysteresis:
                return "center"
            return "right"

        if normalized_x < left - hysteresis:
            return "left"
        if normalized_x > right + hysteresis:
            return "right"
        return "center"

    @staticmethod
    def _validate_frame_dimensions(detections: tuple[Detection, ...]) -> None:
        if not detections:
            return
        dimensions = {(detection.frame_width, detection.frame_height) for detection in detections}
        if len(dimensions) > 1:
            raise ValueError("detections from one update must share frame dimensions")


def _smooth_bbox(
    *, previous: BoundingBox, current: BoundingBox, current_weight: float
) -> BoundingBox:
    """Apply an exponential moving average to each box coordinate."""

    previous_weight = 1.0 - current_weight
    return tuple(
        (old * previous_weight) + (new * current_weight)
        for old, new in zip(previous, current, strict=True)
    )


def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection over union for two ``xyxy`` pixel boxes."""

    first_x_min, first_y_min, first_x_max, first_y_max = first
    second_x_min, second_y_min, second_x_max, second_y_max = second

    intersection_width = max(0.0, min(first_x_max, second_x_max) - max(first_x_min, second_x_min))
    intersection_height = max(0.0, min(first_y_max, second_y_max) - max(first_y_min, second_y_min))
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x_max - first_x_min) * max(0.0, first_y_max - first_y_min)
    second_area = max(0.0, second_x_max - second_x_min) * max(0.0, second_y_max - second_y_min)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area
