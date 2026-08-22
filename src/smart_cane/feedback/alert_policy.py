"""Stateful policy for deciding when a risk assessment deserves speech.

Risk scoring and speech execution remain separate concerns: ``RiskEngine``
decides whether an observation is dangerous, this policy suppresses repeated
announcements, and ``AudioFeedback`` handles process overlap and playback.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

from smart_cane.decision.risk_engine import RiskAssessment


AlertReason = Literal[
    "not_alertable",
    "first_alert",
    "risk_upgraded",
    "direction_changed",
    "signature_changed",
    "cooldown_elapsed",
    "cooldown_active",
]
AlertSignature: TypeAlias = tuple[str, str, str]
Clock: TypeAlias = Callable[[], float]


@dataclass(frozen=True, slots=True)
class AlertDecision:
    """Outcome of evaluating one tracked risk assessment."""

    should_emit: bool
    reason: AlertReason
    assessment: RiskAssessment
    cooldown_remaining: float = 0.0

    @property
    def message(self) -> str:
        """Speech-ready text from the underlying risk assessment."""

        return self.assessment.message


@dataclass(slots=True)
class _TrackState:
    last_assessment: RiskAssessment | None = None
    emitted_at: dict[AlertSignature, float] = field(default_factory=dict)
    pending_signature: AlertSignature | None = None
    pending_reason: AlertReason | None = None


class AlertPolicy:
    """Suppress repeated alerts while preserving meaningful changes.

    Cooldown is scoped to ``track + label + direction + risk level``.  A
    medium-to-high upgrade or a change between the left/centre/right direction
    categories emits immediately, even if that exact signature was recently
    announced.  Call :meth:`retain_active` once per tracking cycle so vanished
    objects lose their history and can be announced if they later reappear.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        if isinstance(cooldown_seconds, bool):
            raise TypeError("cooldown_seconds must be a finite number")
        try:
            normalized_cooldown = float(cooldown_seconds)
        except (TypeError, ValueError) as error:
            raise TypeError("cooldown_seconds must be a finite number") from error
        if not isfinite(normalized_cooldown) or normalized_cooldown < 0:
            raise ValueError("cooldown_seconds must be finite and non-negative")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.cooldown_seconds = normalized_cooldown
        self._clock = clock
        self._tracks: dict[Hashable, _TrackState] = {}

    @property
    def tracked_count(self) -> int:
        """Number of track histories retained by the policy."""

        return len(self._tracks)

    def evaluate(
        self,
        track_key: Hashable,
        assessment: RiskAssessment,
    ) -> AlertDecision:
        """Evaluate and immediately commit an alert decision.

        This convenience method preserves the original single-step API. Live
        feedback code should use :meth:`propose` followed by :meth:`commit` so
        a failed audio delivery does not consume the cooldown.
        """

        decision = self.propose(track_key, assessment)
        if decision.should_emit:
            self.commit(track_key, decision)
        return decision

    def propose(
        self,
        track_key: Hashable,
        assessment: RiskAssessment,
    ) -> AlertDecision:
        """Return a candidate without consuming its cooldown.

        The latest observation is retained for upgrade and direction-change
        comparisons. Call :meth:`commit` only after the output adapter accepts
        a candidate. An uncommitted candidate is eligible again next frame.
        """

        self._validate_track_key(track_key)
        if not isinstance(assessment, RiskAssessment):
            raise TypeError("assessment must be a RiskAssessment")

        now = float(self._clock())
        if not isfinite(now):
            raise ValueError("clock must return a finite timestamp")

        state = self._tracks.setdefault(track_key, _TrackState())
        previous = state.last_assessment

        if not assessment.should_alert:
            state.last_assessment = assessment
            state.pending_signature = None
            state.pending_reason = None
            return AlertDecision(False, "not_alertable", assessment)

        signature = self._signature(assessment)
        last_emitted_at = state.emitted_at.get(signature)

        if state.pending_signature == signature and state.pending_reason is not None:
            return self._propose(
                state,
                signature,
                assessment,
                reason=state.pending_reason,
            )

        if not state.emitted_at:
            return self._propose(
                state,
                signature,
                assessment,
                reason="first_alert",
            )

        if (
            previous is not None
            and previous.label.casefold() == assessment.label.casefold()
            and previous.level == "medium"
            and assessment.level == "high"
        ):
            return self._propose(
                state,
                signature,
                assessment,
                reason="risk_upgraded",
            )

        if (
            previous is not None
            and previous.label.casefold() == assessment.label.casefold()
            and previous.direction != assessment.direction
        ):
            return self._propose(
                state,
                signature,
                assessment,
                reason="direction_changed",
            )

        if last_emitted_at is None:
            return self._propose(
                state,
                signature,
                assessment,
                reason="signature_changed",
            )

        elapsed = max(0.0, now - last_emitted_at)
        if elapsed >= self.cooldown_seconds:
            return self._propose(
                state,
                signature,
                assessment,
                reason="cooldown_elapsed",
            )

        state.last_assessment = assessment
        state.pending_signature = None
        state.pending_reason = None
        return AlertDecision(
            False,
            "cooldown_active",
            assessment,
            cooldown_remaining=self.cooldown_seconds - elapsed,
        )

    def commit(self, track_key: Hashable, decision: AlertDecision) -> bool:
        """Consume the cooldown for a candidate accepted by an output adapter."""

        self._validate_track_key(track_key)
        if not isinstance(decision, AlertDecision):
            raise TypeError("decision must be an AlertDecision")
        if not decision.should_emit:
            return False

        now = float(self._clock())
        if not isfinite(now):
            raise ValueError("clock must return a finite timestamp")
        state = self._tracks.setdefault(track_key, _TrackState())
        signature = self._signature(decision.assessment)
        state.emitted_at[signature] = now
        state.last_assessment = decision.assessment
        if state.pending_signature == signature:
            state.pending_signature = None
            state.pending_reason = None
        return True

    def retain_active(self, track_keys: Iterable[Hashable]) -> int:
        """Forget state for tracks absent from ``track_keys``.

        Returns the number of histories removed.  Validation occurs before any
        state is changed so an unhashable key cannot leave a partial cleanup.
        """

        try:
            active = set(track_keys)
        except TypeError as error:
            raise TypeError("track_keys must contain only hashable values") from error

        stale = self._tracks.keys() - active
        for track_key in stale:
            del self._tracks[track_key]
        return len(stale)

    @staticmethod
    def _validate_track_key(track_key: Hashable) -> None:
        try:
            hash(track_key)
        except TypeError as error:
            raise TypeError("track_key must be hashable") from error

    @staticmethod
    def _signature(assessment: RiskAssessment) -> AlertSignature:
        return (
            assessment.label.strip().casefold(),
            assessment.direction,
            assessment.level,
        )

    @staticmethod
    def _propose(
        state: _TrackState,
        signature: AlertSignature,
        assessment: RiskAssessment,
        *,
        reason: AlertReason,
    ) -> AlertDecision:
        state.last_assessment = assessment
        state.pending_signature = signature
        state.pending_reason = reason
        return AlertDecision(True, reason, assessment)
