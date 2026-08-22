"""OpenCV-backed camera and video input.

OpenCV is imported only when a source is opened, so importing the project and
running domain tests does not require native camera dependencies.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from types import TracebackType
from typing import Any


class CameraError(RuntimeError):
    """Base error for camera-source failures."""


class CameraOpenError(CameraError):
    """Raised when OpenCV cannot open the requested source."""


class CameraReadError(CameraError):
    """Raised when an opened source cannot provide another frame."""


class CameraSource:
    """Read frames from a camera index or video path through OpenCV.

    The source opens explicitly with :meth:`open`, on the first :meth:`read`,
    or when entering a ``with`` block.  Calling :meth:`release` more than once
    is safe.
    """

    def __init__(
        self,
        source: int | str | os.PathLike[str] = 0,
        *,
        backend: int | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        if isinstance(source, bool) or not isinstance(source, (int, str, os.PathLike)):
            raise TypeError("source must be a camera index or video path")
        if backend is not None and (isinstance(backend, bool) or not isinstance(backend, int)):
            raise TypeError("backend must be an OpenCV backend integer")

        self.source: int | str = os.fspath(source) if isinstance(source, os.PathLike) else source
        self.backend = backend
        self._cv2 = cv2_module
        self._capture: Any | None = None

    def _load_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        try:
            self._cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError as error:
            raise CameraOpenError(
                "OpenCV is required for camera input; install the project dependencies first"
            ) from error
        return self._cv2

    @property
    def is_open(self) -> bool:
        """Whether the underlying OpenCV capture is currently open."""

        if self._capture is None:
            return False
        try:
            return bool(self._capture.isOpened())
        except (AttributeError, TypeError):
            return False

    def open(self) -> CameraSource:
        """Open the configured source and return ``self``."""

        if self.is_open:
            return self

        # Discard a stale capture before attempting to open a replacement.
        self.release()
        cv2 = self._load_cv2()
        try:
            if self.backend is None:
                capture = cv2.VideoCapture(self.source)
            else:
                capture = cv2.VideoCapture(self.source, self.backend)
        except Exception as error:  # OpenCV may expose backend-specific errors.
            raise CameraOpenError(f"could not open camera/video source {self.source!r}") from error

        self._capture = capture
        if not self.is_open:
            self.release()
            message = f"could not open camera/video source {self.source!r}"
            if sys.platform == "darwin" and isinstance(self.source, int):
                message += (
                    "; allow camera access for Codex or your terminal in "
                    "System Settings > Privacy & Security > Camera, then restart it"
                )
            raise CameraOpenError(message)
        return self

    def read(self) -> Any:
        """Return the next frame, opening the source first if necessary."""

        if not self.is_open:
            self.open()
        assert self._capture is not None

        try:
            success, frame = self._capture.read()
        except Exception as error:  # OpenCV may expose backend-specific errors.
            raise CameraReadError(f"failed to read from source {self.source!r}") from error
        if not success or frame is None:
            raise CameraReadError(f"source {self.source!r} did not provide another frame")
        return frame

    def frames(self) -> Iterator[Any]:
        """Yield frames until the source reaches its end or cannot be read."""

        while True:
            try:
                yield self.read()
            except CameraReadError:
                return

    def __iter__(self) -> Iterator[Any]:
        return self.frames()

    def release(self) -> None:
        """Release native camera resources; safe to call repeatedly."""

        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                # Cleanup should not mask the original processing exception.
                pass

    close = release

    def __enter__(self) -> CameraSource:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.release()
        return False
