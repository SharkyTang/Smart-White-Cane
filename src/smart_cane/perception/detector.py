"""Ultralytics YOLO adapter for the project domain model.

The heavyweight third-party package is imported only when a prediction first
needs a model.  Callers can also inject a model-shaped test double, keeping unit
tests and demo fixtures independent of Ultralytics.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from math import isfinite
from typing import Any

from smart_cane.config import DEFAULT_MODEL
from smart_cane.models import Detection


class DetectorError(RuntimeError):
    """Raised when the detector cannot be loaded or its output is malformed."""


class YoloDetector:
    """Run a YOLO model and translate its boxes into :class:`Detection` values."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        confidence: float = 0.45,
        device: str | int | None = None,
        image_size: int = 640,
        model: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name cannot be blank")
        if isinstance(confidence, bool) or not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        self.model_name = model_name.strip()
        self.confidence = float(confidence)
        self.requested_device = device
        self.device = resolve_device(device)
        self.image_size = image_size
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ModuleNotFoundError as error:
            raise DetectorError(
                "Ultralytics is required for YOLO inference; install the project dependencies first"
            ) from error

        try:
            self._model = ultralytics.YOLO(self.model_name)
        except Exception as error:
            raise DetectorError(f"could not load YOLO model {self.model_name!r}") from error
        return self._model

    @property
    def model(self) -> Any:
        """Loaded model, initialized on first access."""

        return self._load_model()

    def predict_raw(self, frame: Any) -> Any:
        """Return the model's unmodified prediction result for ``frame``."""

        if frame is None:
            raise ValueError("frame cannot be None")

        options: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
            "device": self.device,
        }

        try:
            return self.model.predict(**options)
        except Exception as error:
            raise DetectorError("YOLO inference failed") from error

    def detect(self, frame: Any, *, raw: bool = False) -> list[Detection] | Any:
        """Detect objects in ``frame``.

        By default, results are lightweight, model-agnostic :class:`Detection`
        instances.  Pass ``raw=True`` when an experiment needs the original
        Ultralytics result object (for example, to use its plotting helpers).
        """

        raw_results = self.predict_raw(frame)
        if raw:
            return raw_results
        return self._to_detections(raw_results, frame)

    def __call__(self, frame: Any) -> list[Detection]:
        result = self.detect(frame)
        # ``raw`` is false above, so this cast-free assertion documents the API.
        assert isinstance(result, list)
        return result

    def warmup(self, frame: Any) -> list[Detection]:
        """Load the model and run one measured prediction before steady-state work."""

        return self(frame)

    def _to_detections(self, raw_results: Any, frame: Any) -> list[Detection]:
        detections: list[Detection] = []
        for result in _result_sequence(raw_results):
            frame_height, frame_width = _frame_dimensions(result, frame)
            names = getattr(result, "names", None) or getattr(self.model, "names", None)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            try:
                box_iterator = iter(boxes)
            except TypeError as error:
                raise DetectorError("YOLO result boxes are not iterable") from error

            for box in box_iterator:
                coordinates = _flat_numbers(getattr(box, "xyxy", None))
                confidence_values = _flat_numbers(getattr(box, "conf", None))
                class_values = _flat_numbers(getattr(box, "cls", None))
                if len(coordinates) < 4 or not confidence_values or not class_values:
                    raise DetectorError("YOLO returned an incomplete bounding box")

                class_id = int(class_values[0])
                detections.append(
                    Detection(
                        label=_class_name(names, class_id),
                        confidence=float(confidence_values[0]),
                        bbox=tuple(float(value) for value in coordinates[:4]),
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )
        return detections


def _result_sequence(raw_results: Any) -> Iterable[Any]:
    if raw_results is None:
        return ()
    if hasattr(raw_results, "boxes"):
        return (raw_results,)
    try:
        return iter(raw_results)
    except TypeError as error:
        raise DetectorError("YOLO returned an unsupported result object") from error


def _frame_dimensions(result: Any, frame: Any) -> tuple[int, int]:
    original_shape = getattr(result, "orig_shape", None)
    if original_shape is not None and len(original_shape) >= 2:
        height, width = int(original_shape[0]), int(original_shape[1])
    else:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            raise DetectorError("cannot determine frame dimensions")
        height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise DetectorError("frame dimensions must be positive")
    return height, width


def _flat_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        else:
            try:
                flattened.append(float(item))
            except (TypeError, ValueError) as error:
                raise DetectorError("YOLO returned non-numeric box data") from error

    visit(value)
    return flattened


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        label = names.get(class_id)
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        label = names[class_id]
    else:
        label = None
    return str(label) if label is not None else f"class_{class_id}"


def resolve_device(
    device: str | int | None,
    *,
    torch_module: Any | None = None,
) -> str | int:
    """Select a reliable inference device for the Mac-first prototype.

    CPU remains the measured default. Explicit MPS requests fall back to CPU
    when the installed PyTorch build or current process cannot use MPS.
    Other explicit Ultralytics device strings are passed through unchanged.
    """

    if isinstance(device, bool) or not isinstance(device, (str, int, type(None))):
        raise TypeError("device must be a string, integer index, or None")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("device index cannot be negative")
        return device
    if device is None or device.strip().casefold() == "auto":
        return "cpu"

    normalized = device.strip().casefold()
    if not normalized:
        raise ValueError("device cannot be blank")
    if normalized != "mps":
        return device.strip()

    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ModuleNotFoundError:
            return "cpu"

    try:
        available = bool(torch_module.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        available = False
    return "mps" if available else "cpu"
