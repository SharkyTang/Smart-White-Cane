"""Deterministic, hardware-independent obstacle risk decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

from smart_cane.models import Detection


Direction = Literal["left", "center", "right"]
RiskLevel = Literal["low", "medium", "high"]

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
class RiskAssessment:
    """A small, transport-friendly decision for one detection."""

    label: str
    direction: Direction
    level: RiskLevel
    should_alert: bool
    message: str
    confidence: float
    area_ratio: float

    @property
    def rank(self) -> int:
        """Numeric ordering useful when combining several evidence sources."""

        return _RISK_RANK[self.level]


class RiskEngine:
    """Convert object detections into deterministic warning decisions."""

    def __init__(self, rules: RiskRules | None = None) -> None:
        self.rules = rules or RiskRules()

    def assess(
        self,
        detection: Detection,
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
        direction = direction_override or self._direction_for(detection)
        area_ratio = detection.area_ratio
        thresholds = self.rules.thresholds_for(detection.label)

        if detection.confidence < self.rules.minimum_confidence:
            level: RiskLevel = "low"
        elif area_ratio >= thresholds.high:
            level = "high"
        elif area_ratio >= thresholds.medium:
            level = "medium"
        else:
            level = "low"

        should_alert = (
            detection.confidence >= self.rules.minimum_confidence
            and level in self.rules.alert_levels
        )
        message = self._message_for(detection.label, direction, level)

        return RiskAssessment(
            label=detection.label,
            direction=direction,
            level=level,
            should_alert=should_alert,
            message=message,
            confidence=detection.confidence,
            area_ratio=area_ratio,
        )

    def assess_all(self, detections: Iterable[Detection]) -> tuple[RiskAssessment, ...]:
        """Assess detections in input order."""

        return tuple(self.assess(detection) for detection in detections)

    def most_urgent(self, detections: Iterable[Detection]) -> RiskAssessment | None:
        """Select one warning using a stable, documented priority order.

        Higher risk wins first, followed by objects in the centre of the path,
        larger visible area, and detector confidence.  Python's stable ``max``
        keeps the first input when all priority values are equal.
        """

        assessments = self.assess_all(detections)
        if not assessments:
            return None
        return max(
            assessments,
            key=lambda item: (
                item.rank,
                item.direction == "center",
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
    def _message_for(label: str, direction: Direction, level: RiskLevel) -> str:
        location = {
            "left": "on the left",
            "center": "ahead",
            "right": "on the right",
        }[direction]
        if level == "high":
            return f"Danger: {label} {location}."
        if level == "medium":
            return f"Caution: {label} {location}."
        return f"{label.capitalize()} detected {location}."
