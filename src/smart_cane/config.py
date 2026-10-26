from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


DEFAULT_MODEL = "yolo26n.pt"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime settings kept independent from hardware-specific adapters."""

    source: int | str = 0
    model: str = DEFAULT_MODEL
    confidence: float = 0.45
    device: str | int | None = None
    image_size: int = 640
    warmup_frames: int = 5
    confirmation_window: int = 5
    minimum_confirmations: int = 3
    high_risk_area_ratio: float = 0.18
    high_risk_consecutive_frames: int = 2
    max_missing_frames: int = 3
    direction_hysteresis: float = 0.05
    alert_cooldown_seconds: float = 3.0
    distance_high_meters: float = 1.0
    distance_medium_meters: float = 2.0
    distance_hysteresis_meters: float = 0.15
    distance_max_meters: float = 5.0
    distance_min_samples: int = 12
    distance_min_valid_ratio: float = 0.20
    distance_history_size: int = 3
    distance_enter_confirmations: int = 2
    distance_exit_confirmations: int = 3
    audio_cooldown_seconds: float | None = None
    audio_enabled: bool = False
    voice: str | None = None
    display: bool = True
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model cannot be blank")
        if (
            isinstance(self.confidence, bool)
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")
        if self.device is not None:
            if isinstance(self.device, bool) or not isinstance(self.device, (str, int)):
                raise ValueError("device must be a name or non-negative index")
            if isinstance(self.device, str) and not self.device.strip():
                raise ValueError("device cannot be blank")
            if isinstance(self.device, int) and self.device < 0:
                raise ValueError("device index cannot be negative")
        _require_positive_int("image_size", self.image_size)
        _require_non_negative_int("warmup_frames", self.warmup_frames)
        _require_positive_int("confirmation_window", self.confirmation_window)
        _require_positive_int("minimum_confirmations", self.minimum_confirmations)
        if self.minimum_confirmations > self.confirmation_window:
            raise ValueError("minimum_confirmations cannot exceed confirmation_window")
        if (
            isinstance(self.high_risk_area_ratio, bool)
            or not isfinite(self.high_risk_area_ratio)
            or not 0.06 < self.high_risk_area_ratio <= 1.0
        ):
            raise ValueError("high_risk_area_ratio must be greater than 0.06 and at most 1")
        _require_positive_int("high_risk_consecutive_frames", self.high_risk_consecutive_frames)
        _require_positive_int("max_missing_frames", self.max_missing_frames)
        if (
            isinstance(self.direction_hysteresis, bool)
            or not isfinite(self.direction_hysteresis)
            or not 0.0 <= self.direction_hysteresis < 0.15
        ):
            raise ValueError("direction_hysteresis must be at least 0 and less than 0.15")
        self._validate_distance_settings()
        _validate_cooldown("alert_cooldown_seconds", self.alert_cooldown_seconds)
        if self.audio_cooldown_seconds is not None:
            _validate_cooldown("audio_cooldown_seconds", self.audio_cooldown_seconds)
            if (
                self.alert_cooldown_seconds != 3.0
                and self.alert_cooldown_seconds != self.audio_cooldown_seconds
            ):
                raise ValueError("audio_cooldown_seconds conflicts with alert_cooldown_seconds")
            object.__setattr__(
                self,
                "alert_cooldown_seconds",
                float(self.audio_cooldown_seconds),
            )
        if self.voice is not None:
            if not isinstance(self.voice, str) or not self.voice.strip():
                raise ValueError("voice cannot be blank")
        if self.max_frames is not None:
            _require_positive_int("max_frames", self.max_frames)

    def _validate_distance_settings(self) -> None:
        distances = {
            "distance_high_meters": self.distance_high_meters,
            "distance_medium_meters": self.distance_medium_meters,
            "distance_hysteresis_meters": self.distance_hysteresis_meters,
            "distance_max_meters": self.distance_max_meters,
        }
        for name, value in distances.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if not (self.distance_high_meters < self.distance_medium_meters < self.distance_max_meters):
            raise ValueError("distance thresholds must satisfy high < medium < maximum distance")
        if self.distance_medium_meters + self.distance_hysteresis_meters > self.distance_max_meters:
            raise ValueError("distance hysteresis must fit below the maximum distance")
        if (
            isinstance(self.distance_min_valid_ratio, bool)
            or not isinstance(self.distance_min_valid_ratio, (int, float))
            or not isfinite(self.distance_min_valid_ratio)
            or not 0.0 < self.distance_min_valid_ratio <= 1.0
        ):
            raise ValueError("distance_min_valid_ratio must be greater than 0 and at most 1")
        _require_positive_int("distance_min_samples", self.distance_min_samples)
        _require_positive_int("distance_history_size", self.distance_history_size)
        _require_positive_int("distance_enter_confirmations", self.distance_enter_confirmations)
        _require_positive_int("distance_exit_confirmations", self.distance_exit_confirmations)


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index and everything else as a path."""

    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    return str(Path(stripped).expanduser())


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_cooldown(name: str, value: float) -> None:
    if isinstance(value, bool) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
