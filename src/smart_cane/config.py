from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime settings kept independent from hardware-specific adapters."""

    source: int | str = 0
    model: str = "yolo11n.pt"
    confidence: float = 0.45
    device: str | None = None
    audio_enabled: bool = False
    audio_cooldown_seconds: float = 2.5
    voice: str | None = None
    display: bool = True
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.audio_cooldown_seconds < 0:
            raise ValueError("audio cooldown cannot be negative")
        if self.voice is not None and not self.voice.strip():
            raise ValueError("voice cannot be blank")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index and everything else as a path."""

    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    return str(Path(stripped).expanduser())
