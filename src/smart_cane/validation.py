"""Metadata-only analysis helpers for controlled LiDAR validation sessions."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


JsonRecord = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One measured metric and its acceptance status."""

    value: float | int | None
    target: str
    passed: bool | None


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Aggregate metrics derived without retaining RGB or depth buffers."""

    frame_count: int
    assessment_count: int
    valid_lidar_count: int
    speech_count: int
    median_absolute_error_m: MetricResult
    p95_absolute_error_m: MetricResult
    distance_coverage: MetricResult
    frame_id_mismatches: MetricResult
    replayed_or_out_of_order_frames: MetricResult
    p95_round_trip_ms: MetricResult
    effective_hz: MetricResult
    p95_speech_start_ms: MetricResult
    p95_high_risk_speech_start_ms: MetricResult
    cooldown_violations: MetricResult
    invalid_numeric_speeches: MetricResult
    expected_alert_misses: MetricResult
    expected_alerts_missing_event_id: MetricResult
    scenarios_below_30_samples: MetricResult
    scenarios_below_3_runs: MetricResult
    failing_required_scenarios: MetricResult
    scenarios: Mapping[str, Mapping[str, float | int | str | None]]

    @property
    def has_required_measurements(self) -> bool:
        required = (
            self.median_absolute_error_m,
            self.p95_absolute_error_m,
            self.distance_coverage,
            self.frame_id_mismatches,
            self.replayed_or_out_of_order_frames,
            self.p95_round_trip_ms,
            self.effective_hz,
            self.p95_speech_start_ms,
            self.p95_high_risk_speech_start_ms,
            self.cooldown_violations,
            self.invalid_numeric_speeches,
            self.expected_alert_misses,
            self.expected_alerts_missing_event_id,
            self.scenarios_below_30_samples,
            self.scenarios_below_3_runs,
            self.failing_required_scenarios,
        )
        return all(item.passed is not None for item in required)

    @property
    def passed(self) -> bool | None:
        statuses = [
            self.median_absolute_error_m.passed,
            self.p95_absolute_error_m.passed,
            self.distance_coverage.passed,
            self.frame_id_mismatches.passed,
            self.replayed_or_out_of_order_frames.passed,
            self.p95_round_trip_ms.passed,
            self.effective_hz.passed,
            self.p95_speech_start_ms.passed,
            self.p95_high_risk_speech_start_ms.passed,
            self.cooldown_violations.passed,
            self.invalid_numeric_speeches.passed,
            self.expected_alert_misses.passed,
            self.expected_alerts_missing_event_id.passed,
            self.scenarios_below_30_samples.passed,
            self.scenarios_below_3_runs.passed,
            self.failing_required_scenarios.passed,
        ]
        if any(status is False for status in statuses):
            return False
        if any(status is None for status in statuses):
            return None
        return True


def summarize_validation(records: Iterable[JsonRecord]) -> ValidationSummary:
    """Summarize frame and speech JSONL records from the native client."""

    frames: list[JsonRecord] = []
    speeches: list[JsonRecord] = []
    for record in records:
        record_type = record.get("record_type")
        if record_type == "frame":
            frames.append(record)
        elif record_type == "speech":
            speeches.append(record)

    assessment_count = 0
    valid_lidar_count = 0
    required_observations = 0
    valid_required_observations = 0
    absolute_errors: list[float] = []
    scenario_errors: dict[str, list[float]] = defaultdict(list)
    scenario_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    scenario_groups: dict[str, set[str]] = defaultdict(set)
    scenario_truths: dict[str, set[float | None]] = defaultdict(set)
    run_observations: dict[tuple[str, str, str], int] = defaultdict(int)
    scenario_runs: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    expected_events: set[str] = set()
    expected_alerts_missing_event_id = 0
    alert_ids_by_event: dict[str, set[str]] = defaultdict(set)
    alert_context: dict[str, dict[str, object]] = {}
    frame_ids_by_session: dict[str, list[int]] = defaultdict(list)
    capture_times_by_session: dict[str, list[float]] = defaultdict(list)

    for frame in frames:
        session_id = str(frame.get("session_id") or "single-session")
        ground_truth = _finite_number(frame.get("ground_truth_meters"))
        validation_group = str(frame.get("validation_group", "ordinary"))
        scenario = str(frame.get("scenario_id") or "unspecified")
        run_id = str(frame.get("run_id") or f"session-{session_id}")
        run_key = (session_id, run_id, scenario)
        explicit_event_id = frame.get("expected_alert_event_id")
        event_id = f"{session_id}:{run_id}:{explicit_event_id or scenario}"
        scenario_groups[scenario].add(validation_group)
        scenario_truths[scenario].add(ground_truth)
        scenario_counts[scenario]
        run_observations[run_key]
        scenario_runs[scenario].add(run_key)
        if frame.get("expected_alert") is True:
            expected_events.add(event_id)
            if not isinstance(explicit_event_id, str) or not explicit_event_id.strip():
                expected_alerts_missing_event_id += 1

        assessments = frame.get("assessments", ())
        assessment_by_track: dict[object, Mapping[str, Any]] = {}
        has_stable_assessment = False
        if isinstance(assessments, Sequence) and not isinstance(assessments, (str, bytes)):
            for assessment in assessments:
                if not isinstance(assessment, Mapping):
                    continue
                has_stable_assessment = True
                assessment_count += 1
                assessment_by_track[assessment.get("track_key")] = assessment
                status = assessment.get("distance_status")
                source = assessment.get("risk_source", assessment.get("source"))
                meters = _finite_number(assessment.get("distance_meters"))
                is_valid_lidar = status == "valid" and source == "lidar" and meters is not None
                valid_lidar_count += int(is_valid_lidar)
                scenario_counts[scenario][0] += 1
                if is_valid_lidar:
                    scenario_counts[scenario][1] += 1

                required = (
                    validation_group == "ordinary"
                    and ground_truth is not None
                    and 0.5 <= ground_truth <= 4.0
                )
                if not required:
                    continue
                required_observations += 1
                if is_valid_lidar:
                    valid_required_observations += 1
                    error = abs(meters - ground_truth)
                    absolute_errors.append(error)
                    scenario_errors[scenario].append(error)
        if has_stable_assessment:
            run_observations[run_key] += 1

        alert = frame.get("alert")
        if isinstance(alert, Mapping) and alert.get("should_emit") is True:
            alert_id = alert.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                alert_ids_by_event[event_id].add(alert_id)
                related = assessment_by_track.get(alert.get("track_key"), {})
                alert_context[alert_id] = {
                    "session_id": session_id,
                    "track_key": alert.get("track_key"),
                    "label": alert.get("label"),
                    "direction": alert.get("direction"),
                    "level": alert.get("level"),
                    "capture_timestamp_seconds": frame.get("capture_timestamp_seconds"),
                    "risk_source": alert.get("risk_source", related.get("risk_source")),
                    "distance_status": alert.get("distance_status", related.get("distance_status")),
                }

        frame_id = _non_negative_int(frame.get("frame_id"))
        if frame_id is not None:
            frame_ids_by_session[session_id].append(frame_id)
        captured_at = _finite_number(frame.get("capture_timestamp_seconds"))
        if captured_at is not None:
            capture_times_by_session[session_id].append(captured_at)

    coverage = _ratio(valid_required_observations, required_observations)
    frame_mismatches = sum(
        1
        for frame in frames
        if frame.get("response_frame_id", frame.get("frame_id")) != frame.get("frame_id")
    )
    frame_id_count = sum(len(values) for values in frame_ids_by_session.values())
    replayed_frames = sum(
        current <= previous
        for values in frame_ids_by_session.values()
        for previous, current in zip(values, values[1:])
    )
    round_trip_values = _numeric_values(frame.get("rtt_ms") for frame in frames)
    speech_latencies = _numeric_values(speech.get("speech_start_latency_ms") for speech in speeches)
    spoken_alert_ids = {
        alert_id
        for speech in speeches
        if isinstance((alert_id := speech.get("alert_id")), str) and alert_id
    }
    expected_alert_misses = sum(
        not bool(alert_ids_by_event[event_id] & spoken_alert_ids) for event_id in expected_events
    )

    invalid_numeric_speeches = 0
    numeric_speech_evidence_complete = bool(speeches)
    high_risk_latencies: list[float] = []
    speech_events_by_signature: dict[tuple[object, ...], list[float]] = defaultdict(list)
    cooldown_evidence_complete = False
    for speech in speeches:
        alert_id = speech.get("alert_id")
        context = alert_context.get(alert_id, {}) if isinstance(alert_id, str) else {}
        distance_key = (
            "spoken_distance_meters"
            if "spoken_distance_meters" in speech
            else "distance_meters"
            if "distance_meters" in speech
            else None
        )
        if distance_key is None:
            numeric_speech_evidence_complete = False
        spoken_distance = _finite_number(speech.get(distance_key)) if distance_key else None
        status = speech.get("distance_status", context.get("distance_status"))
        source = speech.get("risk_source", context.get("risk_source"))
        if spoken_distance is not None and (status != "valid" or source != "lidar"):
            invalid_numeric_speeches += 1

        latency = _finite_number(speech.get("speech_start_latency_ms"))
        level = speech.get("level", context.get("level"))
        if level == "high" and latency is not None:
            high_risk_latencies.append(latency)

        signature = tuple(
            speech.get(key, context.get(key))
            for key in ("session_id", "track_key", "label", "direction", "level")
        )
        started_at = _finite_number(speech.get("speech_start_timestamp_seconds"))
        if started_at is None and latency is not None:
            captured_at = _finite_number(context.get("capture_timestamp_seconds"))
            if captured_at is not None:
                started_at = captured_at + latency / 1_000.0
        if started_at is not None and all(value is not None for value in signature):
            speech_events_by_signature[signature].append(started_at)

    cooldown_violations = 0
    for timestamps in speech_events_by_signature.values():
        ordered = sorted(timestamps)
        if len(ordered) >= 2:
            cooldown_evidence_complete = True
        cooldown_violations += sum(
            current - previous < 3.0 for previous, current in zip(ordered, ordered[1:])
        )

    scenarios: dict[str, Mapping[str, float | int | str | None]] = {}
    failing_required_scenarios = 0
    scenarios_below_30 = 0
    scenarios_below_3_runs = 0
    for scenario, (total, valid) in sorted(scenario_counts.items()):
        errors = scenario_errors.get(scenario, ())
        scenario_coverage = _ratio(valid, total)
        scenario_median = statistics.median(errors) if errors else None
        scenario_p95 = _nearest_rank_percentile(errors, 0.95)
        groups = scenario_groups.get(scenario, {"ordinary"})
        truths = scenario_truths.get(scenario, {None})
        metadata_consistent = len(groups) == 1 and len(truths) == 1
        group = next(iter(groups)) if len(groups) == 1 else "mixed"
        truth = next(iter(truths)) if len(truths) == 1 else None
        runs = scenario_runs.get(scenario, set())
        runs_below_30 = sum(run_observations[run] < 30 for run in runs)
        scenarios_below_30 += runs_below_30
        scenarios_below_3_runs += int(len(runs) < 3)
        required = group == "ordinary" and truth is not None and 0.5 <= truth <= 4.0
        if not metadata_consistent:
            failing_required_scenarios += 1
        elif required:
            failing_required_scenarios += int(
                scenario_coverage is None
                or scenario_coverage < 0.90
                or scenario_median is None
                or scenario_median > 0.15
                or scenario_p95 is None
                or scenario_p95 > 0.30
            )
        scenarios[scenario] = {
            "validation_group": group,
            "ground_truth_meters": truth,
            "metadata_consistent": "yes" if metadata_consistent else "no",
            "run_count": len(runs),
            "runs_below_30": runs_below_30,
            "observations": total,
            "valid_observations": valid,
            "coverage": scenario_coverage,
            "median_absolute_error_m": scenario_median,
            "p95_absolute_error_m": scenario_p95,
        }

    median_error = statistics.median(absolute_errors) if absolute_errors else None
    p95_error = _nearest_rank_percentile(absolute_errors, 0.95)
    p95_rtt = _nearest_rank_percentile(round_trip_values, 0.95)
    hz = _effective_hz_across_sessions(capture_times_by_session.values())
    p95_speech = _nearest_rank_percentile(speech_latencies, 0.95)
    p95_high_risk_speech = _nearest_rank_percentile(high_risk_latencies, 0.95)

    return ValidationSummary(
        frame_count=len(frames),
        assessment_count=assessment_count,
        valid_lidar_count=valid_lidar_count,
        speech_count=len(speeches),
        median_absolute_error_m=_metric(median_error, "<= 0.15 m", lambda value: value <= 0.15),
        p95_absolute_error_m=_metric(p95_error, "<= 0.30 m", lambda value: value <= 0.30),
        distance_coverage=_metric(coverage, ">= 90%", lambda value: value >= 0.90),
        frame_id_mismatches=_metric(frame_mismatches, "= 0", lambda value: value == 0),
        replayed_or_out_of_order_frames=_metric(
            replayed_frames if frame_id_count else None,
            "= 0",
            lambda value: value == 0,
        ),
        p95_round_trip_ms=_metric(p95_rtt, "<= 250 ms", lambda value: value <= 250.0),
        effective_hz=_metric(hz, ">= 8 Hz", lambda value: value >= 8.0),
        p95_speech_start_ms=_metric(
            p95_speech,
            "<= 800 ms",
            lambda value: value <= 800.0,
        ),
        p95_high_risk_speech_start_ms=_metric(
            p95_high_risk_speech,
            "<= 500 ms",
            lambda value: value <= 500.0,
        ),
        cooldown_violations=_metric(
            cooldown_violations if cooldown_evidence_complete else None,
            "= 0 (< 3 s repeats)",
            lambda value: value == 0,
        ),
        invalid_numeric_speeches=_metric(
            invalid_numeric_speeches if numeric_speech_evidence_complete else None,
            "= 0",
            lambda value: value == 0,
        ),
        expected_alert_misses=_metric(
            expected_alert_misses if expected_events else None,
            "= 0",
            lambda value: value == 0,
        ),
        expected_alerts_missing_event_id=_metric(
            expected_alerts_missing_event_id if expected_events else None,
            "= 0",
            lambda value: value == 0,
        ),
        scenarios_below_30_samples=_metric(
            scenarios_below_30 if scenarios else None,
            "= 0 (each run >= 30 stable samples)",
            lambda value: value == 0,
        ),
        scenarios_below_3_runs=_metric(
            scenarios_below_3_runs if scenarios else None,
            "= 0 (each scenario >= 3 runs)",
            lambda value: value == 0,
        ),
        failing_required_scenarios=_metric(
            failing_required_scenarios if scenarios else None,
            "= 0",
            lambda value: value == 0,
        ),
        scenarios=scenarios,
    )


def render_validation_markdown(summary: ValidationSummary) -> str:
    """Render a concise acceptance report suitable for the milestone document."""

    overall = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}[summary.passed]
    rows = (
        ("Median absolute error", summary.median_absolute_error_m, "m"),
        ("P95 absolute error", summary.p95_absolute_error_m, "m"),
        ("Valid LiDAR coverage", summary.distance_coverage, "%"),
        ("Frame ID mismatches", summary.frame_id_mismatches, ""),
        ("Replayed/out-of-order frames", summary.replayed_or_out_of_order_frames, ""),
        ("P95 round trip", summary.p95_round_trip_ms, "ms"),
        ("Effective processing rate", summary.effective_hz, "Hz"),
        ("P95 speech start", summary.p95_speech_start_ms, "ms"),
        ("P95 high-risk speech start", summary.p95_high_risk_speech_start_ms, "ms"),
        ("Cooldown violations", summary.cooldown_violations, ""),
        ("Invalid numeric speeches", summary.invalid_numeric_speeches, ""),
        ("Expected alert misses", summary.expected_alert_misses, ""),
        ("Expected alerts without event ID", summary.expected_alerts_missing_event_id, ""),
        ("Runs below 30 stable samples", summary.scenarios_below_30_samples, ""),
        ("Scenarios below three runs", summary.scenarios_below_3_runs, ""),
        ("Failing required scenarios", summary.failing_required_scenarios, ""),
    )
    lines = [
        "# LiDAR validation summary",
        "",
        f"Metadata gates: **{overall}**",
        "",
        (
            "Final milestone conclusion: **NOT DETERMINED BY THIS REPORT ALONE**. "
            "Complete the registration/material matrix, three repeats per combination, "
            "failure recovery and five-minute no-crash checklist manually."
        ),
        "",
        (
            f"Frames: {summary.frame_count}; assessments: {summary.assessment_count}; "
            f"valid LiDAR: {summary.valid_lidar_count}; speech events: {summary.speech_count}."
        ),
        "",
        "| Metric | Result | Target | Status |",
        "|---|---:|---:|---|",
    ]
    for label, metric, unit in rows:
        value = _format_metric_value(metric.value, unit)
        status = {True: "PASS", False: "FAIL", None: "NOT MEASURED"}[metric.passed]
        lines.append(f"| {label} | {value} | {metric.target} | {status} |")

    if summary.scenarios:
        lines.extend(
            [
                "",
                "## Scenarios",
                "",
                "| Scenario | Group / truth | Runs | Valid / total | Coverage | Median error | P95 error |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in summary.scenarios.items():
            coverage = values["coverage"]
            median_error = values["median_absolute_error_m"]
            p95_error = values["p95_absolute_error_m"]
            truth = _format_optional_float(values["ground_truth_meters"], "m")
            lines.append(
                f"| {name} | {values['validation_group']} / {truth} | "
                f"{values['run_count']} ({values['runs_below_30']} below 30) | "
                f"{values['valid_observations']} / {values['observations']} | "
                f"{_format_optional_percent(coverage)} | "
                f"{_format_optional_float(median_error, 'm')} | "
                f"{_format_optional_float(p95_error, 'm')} |"
            )
    return "\n".join(lines) + "\n"


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _numeric_values(values: Iterable[object]) -> list[float]:
    return [number for value in values if (number := _finite_number(value)) is not None]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _effective_hz_across_sessions(groups: Iterable[Sequence[float]]) -> float | None:
    intervals = 0
    elapsed = 0.0
    for capture_times in groups:
        if len(capture_times) < 2:
            continue
        ordered = sorted(capture_times)
        session_elapsed = ordered[-1] - ordered[0]
        if session_elapsed <= 0:
            continue
        intervals += len(ordered) - 1
        elapsed += session_elapsed
    return intervals / elapsed if intervals and elapsed > 0 else None


def _metric(value: float | int | None, target: str, predicate: Any) -> MetricResult:
    return MetricResult(
        value=value, target=target, passed=None if value is None else predicate(value)
    )


def _format_metric_value(value: float | int | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{float(value):.1%}"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}{(' ' + unit) if unit else ''}"


def _format_optional_percent(value: object) -> str:
    number = _finite_number(value)
    return f"{number:.1%}" if number is not None else "—"


def _format_optional_float(value: object, unit: str) -> str:
    number = _finite_number(value)
    return f"{number:.3f} {unit}" if number is not None else "—"
