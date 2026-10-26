# Architecture

Milestone 0.4 keeps machine-learning, stateful policy and hardware adapters
separate while adding a native iPhone RGB-D client. The Milestone 0.3 Safari
thin client remains available as an RGB-only fallback and regression baseline.

```text
CameraSource
    │ frame
    ▼
YoloDetector ─────────────── raw Detection[]
    │                              │
    ▼                              ▼
TemporalFilter              MetricsCollector / validation log
    │ StableDetection[]       capture / inference /
    │ track + smooth box      decision / render / total
    │ stable direction
    ├── RGB-only StableDetection[] ──────┐
    │                                      │
    └── + registered DepthFrame              │
                   │                      │
             DistanceFusion                   │
                   │ FusedDetection[]      │
                   └──────────────────────┘
                                   │
                                   ▼
                              RiskEngine
    │ RiskAssessment[]
    ▼
AlertPolicy ── candidate ──> output accepted? ── yes ──> commit cooldown
    │                               │ no
    │                               └──────────── retry next frame
    ▼
AudioFeedback / Mac overlay / iPhone JSON response
```

## Boundaries

- `models.py` and `decision/` do not import OpenCV, Ultralytics or platform audio APIs.
- `CameraSource` owns OpenCV capture lifetime and never stores frames.
- `MobilePipeline` decodes bounded JPEG or RGB-D requests in memory and serializes detector and state access behind one lock; it never stores the request body.
- `YoloDetector` translates third-party results into project `Detection` values.
- `TemporalFilter` is a deliberately small IoU tracker, not a general multi-object tracking system.
- `RiskEngine` receives the stable direction as an override, so raw box jitter cannot undo direction hysteresis.
- `AlertPolicy` decides whether a tracked assessment deserves an event; `AudioFeedback` only handles delivery.
- Live audio uses propose/commit delivery: failed or busy output does not consume the policy cooldown.
- Safari delivery commits an alert only after the complete JSON response has
  been written. Native RGB-D delivery instead waits for an idempotent
  `/api/alert/ack` sent when `AVSpeechSynthesizer` actually starts speaking.
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

## Safari request lifecycle

1. A user gesture starts Safari's rear-camera preview and unlocks browser audio.
2. Canvas scales the current frame and encodes one JPEG in memory.
3. The page waits for `POST /api/frame` to finish before scheduling another frame, so old frames cannot queue.
4. The Mac decodes, detects and evaluates the frame through the same `TemporalFilter`, `RiskEngine` and `AlertPolicy` used by the camera loop.
5. The response contains raw detections, stable assessments, at most one alert candidate and stage timings.
6. Safari draws the latest stable assessments and emits Chinese speech/tone only for `alert.should_emit`.
7. The HTTP handler commits that alert after the response write succeeds.
8. Stop/start calls `POST /api/session/reset`, clearing tracks and alert history for the single-phone prototype.

The transport is same-origin HTTPS on a trusted local network. The page loads no CDN assets, the service has no cloud integration, and generated certificate keys are ignored from version control. The server is intentionally a single-phone learning prototype rather than an authenticated multi-user service.

## Native RGB-D request lifecycle

1. `ARWorldTrackingConfiguration` supplies `capturedImage`, `sceneDepth` and
   `confidenceMap` from one `ARFrame`; unsupported devices stop before capture.
2. The client applies one portrait transform to RGB, depth and confidence. RGB
   is aspect-fit and never cropped or mirrored, keeping box mapping reversible.
3. RGB is scaled to a longest edge of 640 and JPEG encoded. Registered depth is
   encoded as little-endian Float16 metres and confidence as UInt8.
4. `SCF1` carries four big-endian payload lengths followed by metadata JSON,
   JPEG, depth and confidence. The request content type is versioned and the
   complete body is capped at 2 MB.
5. Only one request may be in flight. While it is running, capture replaces one
   pending slot with the latest frame instead of queuing old frames.
6. The Mac validates schema, lengths, dimensions, aspect ratios and session /
   frame ordering before inference. It echoes `session_id` and `frame_id`; the
   client ignores mismatched responses.
7. YOLO detections pass through `TemporalFilter`. Each stable box is then mapped
   into the registered depth image by normalized coordinates and fused before
   `RiskEngine.assess()`.
8. An emitted alert keeps one stable `alert_id` and its original
   `alert_frame_id` until acknowledged. Repeated responses do not cause repeated
   speech on iOS.
9. `AVSpeechSynthesizerDelegate.didStart` records the real speech-start time and
   posts the alert ID with its origin frame. Only then does the Mac commit the
   cooldown; a lost ACK is safe to retry.
10. Stop, backgrounding, AR interruption or recovery resets the session. Serious
    thermal state lowers the target rate to 5 Hz; critical state stops capture.

## Depth fusion semantics

- `DepthFrame` contains one aligned depth/confidence grid and frame identity;
  raw `Detection` remains unchanged.
- Each stable box is normalized into the depth grid, clipped, then inset by 15%
  on all sides to reduce edge/background contamination.
- Samples must have medium or high ARKit confidence, be finite and satisfy
  `0 < depth <= 5 m`. A frame candidate needs at least 12 samples and 20%
  coverage of the inset ROI.
- The formal candidate is the near-side 20th percentile. The whole-ROI median
  and centre-block median are diagnostics only.
- The selected distance is the median of the latest three valid candidates for
  that track. Missing YOLO or invalid depth produces an explicit stale/invalid
  status; a retained stale value is display-only and cannot drive risk or spoken
  metres.
- Metric levels default to high at `<= 1.0 m`, medium at `1.0-2.0 m`, and low
  above `2.0 m`. Upgrades require two fresh samples; downgrades require three
  and cross an additional 0.15 m hysteresis boundary.
- Without usable depth, the existing area heuristic remains active and is
  labelled `area_fallback`; its warning must not contain a metre value.
- Multi-target order is risk, centre direction, reliable/near distance, visible
  area, then detector confidence.

## Timing semantics

- The first prediction is both the cold-start measurement and the first processed frame.
- The first `warmup_frames` processed frames are excluded from steady statistics.
- `steady_average_fps` is pipeline service throughput derived from each steady frame's `total_seconds`.
- `total_seconds` includes small unclassified overhead such as alert dispatch and metrics preparation; it is not defined as the exact sum of the four named stages.
- The on-screen FPS uses completed steady frames only and is shown one frame later; cold-start timing cannot pollute it.

## Risk evidence boundary

The native path can use registered LiDAR depth as metric evidence for a target
that YOLO has already detected. Visible bounding-box area remains a transparent
fallback, and horizontal box position supplies left/centre/right. The
configurable high-area threshold is shared by fast temporal confirmation and
fallback high-risk classification so the two meanings cannot drift.

LiDAR does not repair a missed or incorrect YOLO box. Even a valid value is the
nearest reliable surface inside that box, not collision certainty or navigation
safety. Glass, mirrors, dark/absorbing material, fine geometry and negative
obstacles require separate evidence and explicit degradation.
