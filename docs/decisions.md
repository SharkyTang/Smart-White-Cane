# Decision log

## D-001 — Build a Mac-based learning prototype first

Use the built-in camera, a pre-trained detector and macOS audio so the perception-to-feedback loop can be learned before embedded hardware is available.

## D-002 — Keep the core hardware-agnostic

Camera, detector and feedback are adapters. Risk decisions operate on small data classes so they can later accept Raspberry Pi cameras and distance sensors.

## D-003 — Prefer a transparent baseline over premature complexity

Milestone 0.1 uses deterministic risk rules. GPS navigation, SLAM, cloud connectivity, model training and user studies remain out of scope until the core loop is measurable.

## D-004 — Treat the prototype as non-safety-critical

Bounding-box size is not distance. Audio messages and on-screen risk labels are learning outputs only, not navigation guidance.
