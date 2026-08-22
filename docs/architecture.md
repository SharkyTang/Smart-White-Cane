# Architecture

Milestone 0.2 keeps machine-learning, stateful policy and hardware adapters separate.

```text
CameraSource
    │ frame
    ▼
YoloDetector ─────────────── raw Detection[]
    │                              │
    ▼                              ▼
TemporalFilter              MetricsCollector
    │ StableDetection[]       capture / inference /
    │ track + smooth box      decision / render / total
    │ stable direction
    ▼
RiskEngine
    │ RiskAssessment[]
    ▼
AlertPolicy ── candidate ──> output accepted? ── yes ──> commit cooldown
    │                               │ no
    │                               └──────────── retry next frame
    ▼
AudioFeedback / visual overlay
```

## Boundaries

- `models.py` and `decision/` do not import OpenCV, Ultralytics or platform audio APIs.
- `CameraSource` owns OpenCV capture lifetime and never stores frames.
- `YoloDetector` translates third-party results into project `Detection` values.
- `TemporalFilter` is a deliberately small IoU tracker, not a general multi-object tracking system.
- `RiskEngine` receives the stable direction as an override, so raw box jitter cannot undo direction hysteresis.
- `AlertPolicy` decides whether a tracked assessment deserves an event; `AudioFeedback` only handles delivery.
- Live audio uses propose/commit delivery: failed or busy output does not consume the policy cooldown.
- `MetricsCollector` receives durations in seconds and reports steady distributions in milliseconds.
- Future camera, distance, vibration or embedded adapters should preserve these boundaries.

## Frame lifecycle

1. Read one frame.
2. Run YOLO at the configured image size and device.
3. Match same-label boxes to prior tracks by IoU and smooth their coordinates.
4. Confirm an ordinary target after 3 observations in 5 frames, or a large target after 2 consecutive frames.
5. Retain a confirmed target through at most 2 consecutive misses; remove it on the third.
6. Apply direction hysteresis and risk thresholds.
7. Rank all stable targets. If the highest-ranked target is cooling down, consider the next target.
8. Deliver at most one alert candidate in the frame.
9. Commit the cooldown only after the selected output accepts it; with audio disabled, the logical event itself is the accepted output.
10. Record stage timings and optionally render the overlay.

## Timing semantics

- The first prediction is both the cold-start measurement and the first processed frame.
- The first `warmup_frames` processed frames are excluded from steady statistics.
- `steady_average_fps` is pipeline service throughput derived from each steady frame's `total_seconds`.
- `total_seconds` includes small unclassified overhead such as alert dispatch and metrics preparation; it is not defined as the exact sum of the four named stages.
- The on-screen FPS uses completed steady frames only and is shown one frame later; cold-start timing cannot pollute it.

## Provisional risk heuristic

There is no reliable physical distance sensor in this milestone. Visible bounding-box area is a transparent proximity proxy, and horizontal box position supplies left/centre/right. The configurable high-area threshold is shared by fast temporal confirmation and high-risk classification so the two meanings cannot drift.

This heuristic must never be described as physical distance, collision certainty or navigation safety.
