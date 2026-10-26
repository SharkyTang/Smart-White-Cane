"""Deterministic, hardware-independent obstacle risk decisions."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

from smart_cane.depth import DistanceStatus, FusedDetection
from smart_cane.models import Detection


Direction = Literal["left", "center", "right"]
RiskLevel = Literal["low", "medium", "high"]
DistanceSource = Literal["lidar", "area_fallback"]

_RISK_RANK: Mapping[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, slots=True)
class AreaThresholds:
    """Frame-area thresholds used as a simple proximity proxy."""

    medium: float = 0.06
    high: float = 0.18

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium < self.high <= 1.0:
            raise ValueError("area thresholds must satisfy 0 <= medium < high <= 1")


@dataclass(frozen=True, slots=True)
class RiskRules:
    """Configuration for repeatable risk decisions.

    Per-label area thresholds make it possible to tune classes independently
    without placing detector-specific rules in the engine.  A later sensor
    adapter can add distance or motion evidence before selecting the most
    urgent assessment while preserving this result contract.
    """

    minimum_confidence: float = 0.45
    left_boundary: float = 0.35
    right_boundary: float = 0.65
    default_area_thresholds: AreaThresholds = field(default_factory=AreaThresholds)
    per_label_area_thresholds: Mapping[str, AreaThresholds] = field(default_factory=dict)
    alert_levels: frozenset[RiskLevel] = field(
        default_factory=lambda: frozenset(("medium", "high"))
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if not 0.0 < self.left_boundary < self.right_boundary < 1.0:
            raise ValueError("direction boundaries must satisfy 0 < left < right < 1")

        unknown_levels = set(self.alert_levels) - set(_RISK_RANK)
        if unknown_levels:
            raise ValueError(f"unknown alert levels: {sorted(unknown_levels)!r}")

        normalized_thresholds: dict[str, AreaThresholds] = {}
        for label, thresholds in self.per_label_area_thresholds.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("per-label threshold keys must be non-empty strings")
            if not isinstance(thresholds, AreaThresholds):
                raise TypeError("per-label threshold values must be AreaThresholds")
            normalized_thresholds[label.strip().casefold()] = thresholds

        object.__setattr__(
            self,
            "per_label_area_thresholds",
            MappingProxyType(normalized_thresholds),
        )
        object.__setattr__(self, "alert_levels", frozenset(self.alert_levels))

    def thresholds_for(self, label: str) -> AreaThresholds:
        """Return class-specific thresholds, falling back to the defaults."""

        return self.per_label_area_thresholds.get(
            label.strip().casefold(), self.default_area_thresholds
        )


@dataclass(frozen=True, slots=True)
class DistanceRiskRules:
    """Metric thresholds and temporal policy for LiDAR-first risk levels."""

    high_distance_m: float = 1.0
    medium_distance_m: float = 2.0
    hysteresis_m: float = 0.15
    enter_frames: int = 2
    exit_frames: int = 3

    def __post_init__(self) -> None:
        for name in ("high_distance_m", "medium_distance_m", "hysteresis_m"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not 0.0 < self.high_distance_m < self.medium_distance_m:
            raise ValueError("distance thresholds must satisfy 0 < high < medium")
        if self.hysteresis_m < 0.0:
            raise ValueError("hysteresis_m cannot be negative")
        for name in ("enter_frames", "exit_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def level_for(self, distance_m: float, current: RiskLevel) -> RiskLevel:
        """Classify distance while applying exit-side threshold hysteresis."""

        if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)):
            raise TypeError("distance_m must be a finite positive number")
        distance = float(distance_m)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_m must be a finite positive number")
        if current not in _RISK_RANK:
            raise ValueError("current must be low, medium, or high")

        if current == "high":
            if distance <= self.high_distance_m + self.hysteresis_m:
                return "high"
            if distance <= self.medium_distance_m + self.hysteresis_m:
                return "medium"
            return "low"
        if current == "medium":
            if distance <= self.high_distance_m:
                return "high"
            if distance <= self.medium_distance_m + self.hysteresis_m:
                return "medium"
            return "low"
        if distance <= self.high_distance_m:
            return "high"
        if distance <= self.medium_distance_m:
            return "medium"
        return "low"


@dataclass(slots=True)
class _DistanceRiskTrack:
    current: RiskLevel = "low"
    pending: RiskLevel | None = None
    pending_frames: int = 0


class DistanceRiskState:
    """Track metric risk transitions with two-frame entry and three-frame exit."""

    def __init__(self, rules: DistanceRiskRules | None = None) -> None:
        self.rules = rules or DistanceRiskRules()
        if not isinstance(self.rules, DistanceRiskRules):
            raise TypeError("rules must be DistanceRiskRules")
        self._tracks: dict[Hashable, _DistanceRiskTrack] = {}

    @property
    def tracked_count(self) -> int:
        return len(self._tracks)

    def update(self, track_key: Hashable, distance_m: float) -> RiskLevel:
        self._validate_track_key(track_key)
        track = self._tracks.setdefault(track_key, _DistanceRiskTrack())
        desired = self.rules.level_for(distance_m, track.current)
        if desired == track.current:
            track.pending = None
            track.pending_frames = 0
            return track.current

        if desired == track.pending:
            track.pending_frames += 1
        else:
            track.pending = desired
            track.pending_frames = 1
        required = (
            self.rules.enter_frames
            if _RISK_RANK[desired] > _RISK_RANK[track.current]
            else self.rules.exit_frames
        )
        if track.pending_frames >= required:
            track.current = desired
            track.pending = None
            track.pending_frames = 0
        return track.current

    def observe_missing(self, track_key: Hashable) -> None:
        """Cancel a pending transition without treating missing depth as evidence."""

        self._validate_track_key(track_key)
        track = self._tracks.get(track_key)
        if track is not None:
            track.pending = None
            track.pending_frames = 0

    def retain_active(self, track_keys: Iterable[Hashable]) -> int:
        try:
            active = set(track_keys)
        except TypeError as error:
            raise TypeError("track_keys must contain only hashable values") from error
        stale = self._tracks.keys() - active
        for track_key in stale:
            del self._tracks[track_key]
        return len(stale)

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _validate_track_key(track_key: Hashable) -> None:
        try:
            hash(track_key)
        except TypeError as error:
            raise TypeError("track_key must be hashable") from error


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """A small, transport-friendly decision for one detection."""

    label: str
    direction: Direction
    level: RiskLevel
    should_alert: bool
    message: str
    confidence: float
    area_ratio: float
    distance_m: float | None = None
    distance_source: DistanceSource = "area_fallback"
    distance_valid: bool = False
    distance_status: DistanceStatus = "unavailable"

    @property
    def rank(self) -> int:
        """Numeric ordering useful when combining several evidence sources."""

        return _RISK_RANK[self.level]

    @property
    def distance_meters(self) -> float | None:
        """Transport-friendly alias matching the RGB-D response contract."""

        return self.distance_m

    @property
    def risk_source(self) -> DistanceSource:
        """Name the evidence source without exposing the legacy field name."""

        return self.distance_source


class RiskEngine:
    """Convert object detections into deterministic warning decisions."""

    def __init__(
        self,
        rules: RiskRules | None = None,
        *,
        distance_rules: DistanceRiskRules | None = None,
        distance_state: DistanceRiskState | None = None,
    ) -> None:
        self.rules = rules or RiskRules()
        self.distance_rules = distance_rules or DistanceRiskRules()
        if not isinstance(self.distance_rules, DistanceRiskRules):
            raise TypeError("distance_rules must be DistanceRiskRules")
        if distance_state is not None and not isinstance(distance_state, DistanceRiskState):
            raise TypeError("distance_state must be DistanceRiskState")
        self.distance_state = distance_state or DistanceRiskState(self.distance_rules)

    def assess(
        self,
        detection: Detection | FusedDetection,
        *,
        direction_override: Direction | None = None,
    ) -> RiskAssessment:
        """Assess one detection using confidence, position, and visible area.

        ``direction_override`` lets a temporal tracker supply a direction that
        has already passed through boundary hysteresis.
        """

        if direction_override is not None and direction_override not in (
            "left",
            "center",
            "right",
        ):
            raise ValueError("direction_override must be left, center, or right")
        fused = detection if isinstance(detection, FusedDetection) else None
        domain_detection = fused.detection if fused is not None else detection
        if not isinstance(domain_detection, Detection):
            raise TypeError("detection must be a Detection or FusedDetection")
        stable_direction = fused.direction if fused is not None else None
        direction = direction_override or stable_direction or self._direction_for(domain_detection)
        area_ratio = domain_detection.area_ratio
        thresholds = self.rules.thresholds_for(domain_detection.label)
        distance_m = None
        distance_source: DistanceSource = "area_fallback"
        distance_status: DistanceStatus = "unavailable"
        if fused is not None and fused.distance.usable:
            distance_m = fused.distance.distance_m
            assert distance_m is not None
            distance_source = "lidar"
            assert fused.distance.status is not None
            distance_status = fused.distance.status
        elif fused is not None:
            assert fused.distance.status is not None
            distance_status = fused.distance.status
            self.distance_state.observe_missing(fused.track_key)

        if domain_detection.confidence < self.rules.minimum_confidence:
            if fused is not None and distance_m is not None:
                self.distance_state.observe_missing(fused.track_key)
            level: RiskLevel = "low"
        elif distance_m is not None and fused is not None:
            level = self.distance_state.update(fused.track_key, distance_m)
        elif area_ratio >= thresholds.high:
            level = "high"
        elif area_ratio >= thresholds.medium:
            level = "medium"
        else:
            level = "low"

        should_alert = (
            domain_detection.confidence >= self.rules.minimum_confidence
            and level in self.rules.alert_levels
        )
        message = self._message_for(domain_detection.label, direction, level, distance_m)

        return RiskAssessment(
            label=domain_detection.label,
            direction=direction,
            level=level,
            should_alert=should_alert,
            message=message,
            confidence=domain_detection.confidence,
            area_ratio=area_ratio,
            distance_m=distance_m,
            distance_source=distance_source,
            distance_valid=distance_m is not None,
            distance_status=distance_status,
        )

    def retain_active(self, track_keys: Iterable[Hashable]) -> int:
        """Forget metric transition state for tracks no longer active."""

        return self.distance_state.retain_active(track_keys)

    def reset(self) -> None:
        """Clear all metric transition history."""

        self.distance_state.reset()

    def assess_all(
        self, detections: Iterable[Detection | FusedDetection]
    ) -> tuple[RiskAssessment, ...]:
        """Assess detections in input order."""

        return tuple(self.assess(detection) for detection in detections)

    def most_urgent(
        self, detections: Iterable[Detection | FusedDetection]
    ) -> RiskAssessment | None:
        """Select one warning using a stable, documented priority order.

        Higher risk wins first, followed by objects in the centre of the path,
        valid nearer metric distance, larger visible area, and detector
        confidence. Python's stable ``max`` keeps the first input when all
        priority values are equal.
        """

        assessments = self.assess_all(detections)
        if not assessments:
            return None
        return max(
            assessments,
            key=lambda item: (
                item.rank,
                item.direction == "center",
                item.distance_valid,
                -item.distance_m if item.distance_m is not None else float("-inf"),
                item.area_ratio,
                item.confidence,
            ),
        )

    def _direction_for(self, detection: Detection) -> Direction:
        normalized_x, _ = detection.normalized_center
        if normalized_x < self.rules.left_boundary:
            return "left"
        if normalized_x > self.rules.right_boundary:
            return "right"
        return "center"

    @staticmethod
    def _message_for(
        label: str,
        direction: Direction,
        level: RiskLevel,
        distance_m: float | None = None,
    ) -> str:
        location = {
            "left": "on the left",
            "center": "ahead",
            "right": "on the right",
        }[direction]
        if level == "high":
            message = f"Danger: {label} {location}"
        elif level == "medium":
            message = f"Caution: {label} {location}"
        else:
            message = f"{label.capitalize()} detected {location}"
        if distance_m is not None:
            return f"{message}, {distance_m:.1f} meters."
        return f"{message}."
