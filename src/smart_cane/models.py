"""Domain models shared by the smart-cane perception pipeline.

The models in this module deliberately do not depend on a camera library or a
particular object detector.  An adapter can translate OpenCV, YOLO, recorded
fixtures, or future sensor-fusion output into these small data objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias


BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """A detector result expressed in pixel coordinates.

    ``bbox`` follows the common ``(x_min, y_min, x_max, y_max)`` convention.
    Coordinates outside the frame are accepted because detectors can produce
    them near an image edge; :attr:`area_ratio` clips the box to the frame.
    """

    label: str
    confidence: float
    bbox: BoundingBox
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as error:
            raise ValueError("confidence must be a number between 0 and 1") from error
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a number between 0 and 1")

        if (
            isinstance(self.frame_width, bool)
            or not isinstance(self.frame_width, int)
            or self.frame_width <= 0
        ):
            raise ValueError("frame_width must be a positive integer")
        if (
            isinstance(self.frame_height, bool)
            or not isinstance(self.frame_height, int)
            or self.frame_height <= 0
        ):
            raise ValueError("frame_height must be a positive integer")

        try:
            if len(self.bbox) != 4:
                raise ValueError
            bbox = tuple(float(value) for value in self.bbox)
        except (TypeError, ValueError) as error:
            raise ValueError("bbox must contain four numeric coordinates") from error

        if not all(isfinite(value) for value in bbox):
            raise ValueError("bbox coordinates must be finite")

        x_min, y_min, x_max, y_max = bbox
        if x_max < x_min or y_max < y_min:
            raise ValueError("bbox maximum coordinates must not be below minimums")

        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox", bbox)

    @property
    def center_x(self) -> float:
        """Horizontal centre of the bounding box in pixels."""

        x_min, _, x_max, _ = self.bbox
        return (x_min + x_max) / 2.0

    @property
    def center_y(self) -> float:
        """Vertical centre of the bounding box in pixels."""

        _, y_min, _, y_max = self.bbox
        return (y_min + y_max) / 2.0

    @property
    def center(self) -> tuple[float, float]:
        """Bounding-box centre as an ``(x, y)`` pixel pair."""

        return self.center_x, self.center_y

    @property
    def normalized_center(self) -> tuple[float, float]:
        """Frame-relative centre, clamped to the inclusive range 0..1."""

        x = min(1.0, max(0.0, self.center_x / self.frame_width))
        y = min(1.0, max(0.0, self.center_y / self.frame_height))
        return x, y

    @property
    def area_ratio(self) -> float:
        """Fraction of the visible frame covered by the clipped bounding box."""

        x_min, y_min, x_max, y_max = self.bbox
        visible_x_min = min(float(self.frame_width), max(0.0, x_min))
        visible_y_min = min(float(self.frame_height), max(0.0, y_min))
        visible_x_max = min(float(self.frame_width), max(0.0, x_max))
        visible_y_max = min(float(self.frame_height), max(0.0, y_max))

        width = max(0.0, visible_x_max - visible_x_min)
        height = max(0.0, visible_y_max - visible_y_min)
        return (width * height) / (self.frame_width * self.frame_height)
