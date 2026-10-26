# Decision log

## D-001 — Build a Mac-based learning prototype first

Use the built-in camera, a pre-trained detector and macOS audio so the perception-to-feedback loop can be learned before embedded hardware is available.

## D-002 — Keep the core hardware-agnostic

Camera, detector and feedback are adapters. Risk decisions operate on small data classes so they can later accept Raspberry Pi cameras and distance sensors.

## D-003 — Prefer a transparent baseline over premature complexity

The prototype uses deterministic risk rules. GPS navigation, SLAM, cloud connectivity, model training and user studies remain out of scope until the core loop is measurable.

## D-004 — Treat the prototype as non-safety-critical

Bounding-box size is not distance. Audio messages and on-screen risk labels are learning outputs only, not navigation guidance.

## D-005 — Stabilise detections before changing the model

Milestone 0.2 keeps YOLO11n at 640 pixels and adds a small same-label IoU tracker. Ordinary targets need 3 observations in a 5-frame window; large targets use a 2-frame fast path; confirmed targets clear on the third consecutive miss. This makes temporal behavior explicit and testable before considering a heavier model.

## D-006 — Add direction hysteresis at the tracking boundary

Left/centre/right is stateful because boxes near a boundary otherwise alternate directions. The tracker supplies the stable direction to the risk engine, while the risk engine remains usable without tracking through its normal box-based direction calculation.

## D-007 — Separate alert policy from delivery

Cooldown is scoped to track, label, direction and risk level. A medium-to-high upgrade or direction change bypasses cooldown. The live application proposes an event first and commits it only after the output accepts it; failed audio therefore remains eligible for retry. Forced upgrades replace an active lower-priority speech process.

## D-008 — Measure cold start separately from steady state

The first model prediction is reported as cold inference. A configurable number of leading frames is excluded from steady FPS and latency distributions. Stage timings cover capture, inference, decision, render and total service time. Runs with no steady samples are labelled as such instead of reporting a misleading zero measurement.

## D-009 — Prefer the measured CPU baseline and make acceleration explicit

CPU is the default because the real-camera baseline sustains the 30 FPS capture rate. Explicit MPS requests are used only when PyTorch reports MPS available; otherwise the detector falls back to CPU and reports the fallback. Resolution reduction and frame skipping are deferred until measurements justify them.

## D-010 — Use iPhone as a local thin client before building a native app

Milestone 0.3 keeps YOLO and the existing stateful decision pipeline on the Mac. Safari captures the iPhone rear camera, sends one JPEG only after the previous request finishes, and renders the returned result. This reuses the measured pipeline, prevents stale-frame queues and avoids premature Core ML, WebGPU or native-app work.

## D-011 — Keep mobile frames local, transient and explicitly trusted

The mobile server binds to the local network, accepts bounded in-memory JPEG requests and never writes frames to disk. iPhone camera access uses a project-local CA and an IP-address SAN rather than a public tunnel, so frames do not transit a cloud service. The CA private key stays ignored on the Mac and the user must explicitly install and trust only its public certificate on the test iPhone.

## D-012 — Treat iPhone web feedback as audio-first

iPhone Safari does not expose a dependable web vibration API. The prototype therefore uses Chinese speech, a short alert tone and a high-contrast visual state. Native haptics remain a later option only if controlled testing demonstrates that audio-first feedback is insufficient.

## D-013 — Add native ARKit only for registered RGB-D capture

Milestone 0.4 adds an iOS 18 SwiftUI client because Safari cannot expose the
iPhone LiDAR depth/confidence data needed by this experiment. The same
`ARFrame` supplies RGB, raw `sceneDepth` and `confidenceMap`; Mac YOLO remains
the detector. This is not a Core ML migration, SLAM/navigation feature, or
reason to remove the independently runnable Milestone 0.3 Safari path.

## D-014 — Define distance as the nearest reliable surface in a stable box

Metric fusion happens after temporal confirmation and before risk assessment.
The stable box is mapped by normalized coordinates into registered depth,
inset 15%, and filtered to medium/high-confidence finite samples in `(0, 5] m`.
At least 12 pixels and 20% ROI coverage are required. The formal frame value is
the 20th percentile and the track value is the median of its latest three valid
frame values. ROI and centre medians are recorded only for comparison. This
definition deliberately favours a nearby reliable surface without treating a
single minimum/outlier as truth.

## D-015 — Prefer metric risk but retain an explicit non-metric fallback

Fresh LiDAR evidence uses 1 m and 2 m boundaries, 0.15 m downgrade hysteresis,
two-sample upgrades and three-sample downgrades. Invalid or stale depth cannot
drive metric risk or spoken metres; the original area rule remains available
as `area_fallback` and may produce only a generic obstacle warning. This keeps
the older input mode functional without misrepresenting box area as distance.

## D-016 — Commit native cooldown only when speech actually starts

An RGB-D alert owns a stable `alert_id` and the frame in which that alert was
first proposed. The iOS client deduplicates that ID and acknowledges it from
`AVSpeechSynthesizerDelegate.didStart`. The server commits cooldown only on the
idempotent ACK and accepts it even if newer frames have arrived. A lost ACK can
therefore be retried without replaying speech or consuming cooldown too early.

## D-017 — Make RGB-D ordering and resource bounds part of the protocol

The `SCF1` envelope uses a fixed versioned media type, big-endian UInt32 section
lengths, JSON metadata, JPEG RGB, little-endian Float16 metres and UInt8
confidence. The server checks the full payload against a 2 MB cap, validates
lengths/dimensions/aspect ratios and accepts only increasing frame IDs in the
current session. The client permits one request in flight and replaces only a
single latest-frame slot. These are correctness rules, not transport tuning.

## D-018 — Keep validation evidence metadata-only and claims conditional

Validation Mode is off by default and never stores RGB/depth pixels. It records
scenario/frame IDs, distance candidates, selected distance/status/risk, alert
reason and latency in JSONL for a reproducible report. Automated tests validate
encoding and policy, but only the fixed-tripod iPhone 16 Pro Max matrix can
support a controlled-indoor conclusion. Registration, error, latency or
incorrect numeric speech failure stops progression to walking tests.

## D-019 — Move the active detector baseline to YOLO26n

New Milestone 0.4 validation runs use the latest Ultralytics generation's nano
detection checkpoint, `yolo26n.pt`, at the existing 640-pixel input size and
0.45 confidence threshold. YOLO11n results remain historical evidence but are
not required as a new control run. The nano scale is retained because the live
iPhone RGB-D loop is latency-sensitive; larger YOLO26 scales require separate
performance evidence before use. Each RGB-D response and validation frame log
records the configured model name so exported evidence identifies the detector
used for that run.
