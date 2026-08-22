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
