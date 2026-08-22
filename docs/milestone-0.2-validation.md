# Milestone 0.2 validation

Date: 2026-08-23

## Acceptance scope

- Continuous-frame confirmation and one-frame false-positive suppression
- Large-target fast confirmation
- Three-miss track clearing and brief-dropout retention
- Direction hysteresis
- Same-signature alert cooldown
- Immediate risk-upgrade and direction-change alerts
- Multiple-target fairness
- Failed-audio retry and forced-message replacement
- Cold versus steady performance metrics
- Real Mac camera operation without saving frames

## Automated checks

Commands:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
git diff --check
```

Final regression: 74 tests passed; Ruff formatting, Ruff lint and whitespace checks passed.

## Real-camera smoke test

Command:

```bash
uv run smart-cane --source 0 --device cpu --no-display --max-frames 90
```

Result:

```text
Processed 90 frames on cpu: 92 raw / 91 stable detections, 1 alert events / 0 spoken.
Cold inference 897.8 ms; steady 30.0 FPS after 5 warmup frames;
inference mean/median/p95 16.6/16.7/17.4 ms.
```

After the final alert-delivery and shared-threshold refinements, the same 90-frame command was repeated successfully:

```text
Processed 90 frames on cpu: 14 raw / 15 stable detections, 0 alert events / 0 spoken.
Cold inference 890.2 ms; steady 29.9 FPS after 5 warmup frames;
inference mean/median/p95 16.5/16.6/17.5 ms.
```

Detection counts differ because the visible camera scene changed; throughput remained at the camera rate.

## Sixty-second real-camera acceptance

Command:

```bash
uv run smart-cane --source 0 --device cpu --no-display --max-frames 1800
```

Result:

```text
Processed 1800 frames on cpu: 1074 raw / 1089 stable detections,
25 alert events / 0 spoken.
Cold inference 890.9 ms; steady 30.0 FPS after 5 warmup frames;
inference mean/median/p95 16.8/16.7/17.4 ms.
```

Notes:

- Audio and display were disabled so the run measured the headless perception pipeline.
- No frames, screenshots or video were saved.
- `stable` counts confirmed target appearances per frame. A confirmed track remains present during a short raw-detector dropout, so the stable count can exceed the raw count.
- The test validates engineering stability and throughput only. It does not validate obstacle-distance accuracy, accessibility, outdoor safety or suitability for real navigation.
