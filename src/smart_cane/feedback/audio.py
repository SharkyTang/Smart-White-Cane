"""Optional, rate-limited speech feedback for macOS.

The adapter intentionally has no dependency on the rest of the perception
stack.  The process runner and monotonic clock are injectable so the cooldown
policy can be tested without speaking or waiting in real time.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any


Runner = Callable[..., Any]
Clock = Callable[[], float]


class AudioFeedback:
    """Speak short alerts with macOS ``say`` while suppressing repetitions.

    Args:
        enabled: When false, :meth:`speak` is a no-op.
        cooldown_seconds: Minimum time between successful announcements.
        voice: Optional macOS voice name passed to ``say -v``.
        rate: Optional positive words-per-minute value passed to ``say -r``.
        clock: Monotonic clock used for cooldown decisions.
        runner: ``subprocess.Popen`` compatible callable used to execute ``say``.

    ``speak`` returns ``True`` when the command launches successfully. Only one
    speech process can run at a time, so alerts never block the perception loop
    or form an unbounded queue. Missing executables and launch failures are
    treated as unavailable feedback rather than crashing the application.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        cooldown_seconds: float = 2.5,
        voice: str | None = None,
        rate: int | None = None,
        clock: Clock = time.monotonic,
        runner: Runner = subprocess.Popen,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if voice is not None and not voice.strip():
            raise ValueError("voice cannot be blank")
        if rate is not None and rate <= 0:
            raise ValueError("rate must be positive")

        self.enabled = bool(enabled)
        self.cooldown_seconds = float(cooldown_seconds)
        self.voice = voice
        self.rate = rate
        self._clock = clock
        self._runner = runner
        self._last_spoken_at: float | None = None
        self._active_process: Any | None = None

    @property
    def last_spoken_at(self) -> float | None:
        """Monotonic timestamp of the last successful announcement."""

        return self._last_spoken_at

    def reset_cooldown(self) -> None:
        """Allow the next announcement immediately."""

        self._last_spoken_at = None

    def _command(self, message: str) -> Sequence[str]:
        command: list[str] = ["say"]
        if self.voice is not None:
            command.extend(("-v", self.voice))
        if self.rate is not None:
            command.extend(("-r", str(self.rate)))
        command.append(message)
        return command

    def speak(self, message: str, *, force: bool = False) -> bool:
        """Speak ``message`` if enabled and outside the cooldown window.

        Arguments are passed as a sequence with ``shell=False``; message text is
        never interpreted by a shell.  ``force`` bypasses the cooldown but does
        not bypass the ``enabled`` switch.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()
        if not self.enabled or not message:
            return False

        now = float(self._clock())
        if (
            not force
            and self._last_spoken_at is not None
            and now - self._last_spoken_at < self.cooldown_seconds
        ):
            return False

        if self._active_process is not None:
            poll = getattr(self._active_process, "poll", None)
            if callable(poll) and poll() is None:
                return False
            wait = getattr(self._active_process, "wait", None)
            if callable(wait):
                wait()
            self._active_process = None

        try:
            process = self._runner(
                list(self._command(message)),
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        returncode = getattr(process, "returncode", 0)
        if returncode not in (None, 0):
            return False

        if callable(getattr(process, "poll", None)):
            self._active_process = process
        self._last_spoken_at = now
        return True

    def notify(self, message: str, *, force: bool = False) -> bool:
        """Semantic alias used by the application feedback boundary."""

        return self.speak(message, force=force)
