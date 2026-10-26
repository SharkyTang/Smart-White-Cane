# Milestone 0.4 iPhone LiDAR validation

For step-by-step Xcode/device setup in Chinese, see the
[中文版操作与验收手册](milestone-0.4-lidar-validation-zh-CN.md).

Date: 2026-08-25

## Status and claim boundary

Milestone 0.4 adds a native iOS RGB-D client, a versioned in-memory transport,
LiDAR/YOLO fusion, metric risk rules, explicit speech acknowledgement and a
metadata-only validation report. Automated tests can verify the protocol and
policy without hardware; the engineering claim remains **not validated** until
the physical checklist and thresholds below are completed on an iPhone 16 Pro
Max.

Passing this milestone can support only the following statement:

> In a controlled indoor setup, for targets already recognised by YOLO, the
> selected RGB box, LiDAR distance and speech policy met the recorded test
> thresholds.

It cannot establish safety for missed detections, glass or mirrors, negative
obstacles, roads, stairs, independent navigation or any medical use.

## Implemented data path

```text
iPhone ARFrame
  capturedImage + sceneDepth + confidenceMap
                 |
                 | SCF1, one request in flight, latest frame only
                 v
POST /api/rgbd-frame on Mac
  JPEG decode -> YOLO -> TemporalFilter -> DistanceFusion
              -> RiskEngine -> AlertPolicy
                 |
                 | response echoes session_id + frame_id
                 v
iPhone overlay + AVSpeechSynthesizer
                 |
                 | didStart -> POST /api/alert/ack
                 v
Mac commits the 3-second cooldown
```

The existing Safari `POST /api/frame` route remains available and independent.
RGB and depth are processed in memory and are not written to disk.

## Before a physical run

1. On the Mac, generate or refresh the local certificate and start the service:

   ```bash
   scripts/create_mobile_cert.sh
   uv run smart-cane-mobile
   ```

2. In Xcode, open `ios/SmartCaneLiDAR/SmartCaneLiDAR.xcodeproj`.
3. Sign in with an Apple ID, choose a Team, and use a unique bundle identifier
   if Xcode requests one.
4. Connect and trust the iPhone 16 Pro Max, enable Developer Mode, and select it
   as the run destination.
5. Install and fully trust only `.mobile-certs/mobile-ca.cer` on the iPhone.
   Never copy the CA private key to the phone.
6. Enter the HTTPS Mac address shown by `smart-cane-mobile`, tap the sound test,
   then start capture. Keep the app in the foreground.
7. Enable Validation Mode only for the controlled measurement run. It records
   IDs, metric candidates, selected risk, alert reason and latency; it does not
   record RGB or depth pixels.

## Automated verification

Run from the repository root:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
node --check src/smart_cane/mobile/app.js
bash -n scripts/create_mobile_cert.sh
git diff --check
uv build
xcodebuild -project ios/SmartCaneLiDAR/SmartCaneLiDAR.xcodeproj \
  -scheme SmartCaneLiDAR \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro Max' test
```

The simulator covers packet encoding, coordinate mapping, request ordering,
retry/session behavior, text/rounding and speech deduplication. It cannot supply
LiDAR data, validate camera/depth registration, or replace the device run.

Automated result for this checkout on 2026-08-25:

- [x] Python unit and loopback integration tests pass.
- [x] Ruff lint and format checks pass.
- [x] JavaScript and shell syntax checks pass.
- [x] Source distribution and wheel build successfully as version `0.4.0`.
- [x] iOS app and XCTest targets compile with `build-for-testing` and signing disabled.
- [ ] XCTest executes in a simulator. This Mac currently has no installed
  CoreSimulator runtime/device, so compiled tests have not run.
- [ ] Xcode device signing and iPhone connection are available.

## Fixed physical setup

- Mount the iPhone in portrait orientation on a tripod.
- Measure from the camera plane to the target's nearest visible surface with a
  tape measure; do not measure from the tripod or phone body edge.
- Keep the phone stationary. Move the target only when a dynamic threshold test
  explicitly calls for it.
- Use a controlled, well-lit indoor space and keep a sighted helper present.
- Do not walk while relying on the app.
- Record the app version/commit, iOS version, Mac model, Wi-Fi network, target,
  material, position, ground-truth distance, scenario ID and run ID.

## Static distance and registration matrix

Distances: `0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0 m`.

Ordinary targets:

- cardboard box;
- chair;
- stationary person.

Boundary-material targets, reported separately:

- thin pole;
- dark cloth;
- glass;
- mirror.

For every target/distance combination, enter three distinct run IDs and collect
at least 30 stable frames under each ID. The analyser gates every run separately
and also requires at least three runs per scenario. For an ordinary opaque target,
also place the target at left, centre,
right and each corner at representative distances. Verify that the same physical
surface remains inside the returned YOLO box and that its distance is consistent
across the field of view.

The 5 m point is diagnostic because the decision filter rejects values above
5 m; the primary accuracy threshold applies to 0.5-4 m.

## Dynamic and failure scenarios

Use a stable scenario ID for the setup, a distinct run ID for each repeat and a
unique expected-event ID for every scripted alert event. Enable “expected alert”
only for that event. The analyser joins the event's `alert_id` to the actual
speech-start record; it fails the metadata gate if an expected event has no
explicit event ID.

- Move a detected opaque target from above 2 m to below 2 m, then back above
  2.15 m; verify confirmation and downgrade hysteresis.
- Move it from above 1 m to below 1 m, then back above 1.15 m; verify immediate
  medium-to-high upgrade and delayed downgrade.
- Hold distance steady while moving left/centre/right; verify direction-change
  alerts and reversible overlay mapping.
- Present two targets; verify priority by risk, centre position, reliable/near
  distance, area and confidence, in that order.
- Repeat the same category/direction/risk for more than 3 seconds; verify one
  alert per cooldown interval and no speech caused by 0.1 m jitter alone.
- Make depth invalid or low-confidence while YOLO remains valid; verify
  `area_fallback`, a generic warning and no spoken metre value.
- Interrupt Wi-Fi, background/foreground the app, deny camera permission,
  interrupt ARKit, and trigger serious/critical thermal policy where practical.
  Verify session reset, no queued history, 5 Hz at serious and stopped capture
  at critical.

## Acceptance thresholds

| Metric | Required result |
|---|---:|
| Ordinary opaque target median absolute error, 0.5-4 m | `<= 0.15 m` |
| Ordinary opaque target P95 absolute error, 0.5-4 m | `<= 0.30 m` |
| Valid-distance coverage | `>= 90%` |
| Response session/frame match | `100%` |
| End-to-end frame round trip P95 | `<= 250 ms` |
| Effective processed rate | `>= 8 Hz` |
| Ordinary confirmed event to speech start P95 | `<= 800 ms` |
| Two-frame high-risk confirmation to speech start | `<= 500 ms` |
| Scripted medium/high events missed | `0` |
| Invalid-depth alerts speaking a metre value | `0` |
| Unchanged alert cooldown | `>= 3.0 s` |
| Continuous foreground run | `>= 5 min`, no crash or replay |
| RGB/depth pixels written by app/service | `0` |

Glass, mirror, dark and thin targets must remain separate from the ordinary
opaque aggregate. A failure on one of those materials must not be hidden by the
ordinary-target average.

## Export and analysis

Export the app's Validation Mode JSONL through the share sheet, then run:

```bash
uv run python scripts/analyze_lidar_validation.py \
  /absolute/path/to/export.jsonl \
  --output docs/milestone-0.4-lidar-results.md
```

The analyser expects metadata-only `frame` and `speech` records. Its report
summarises absolute error, coverage, frame matching, round-trip latency,
effective rate, ordinary/high-risk speech-start latency, 3-second cooldown,
missing expected speech events, missing event IDs, three-run coverage, per-run
sample counts and any numeric speech produced while depth was invalid.
Boundary-material scenarios remain visible
but are excluded from the ordinary opaque accuracy aggregate. Review every
scenario row before using the aggregate outcome.

## Result record

Complete this section only after the device run.

- Device / iOS:
- Mac / macOS:
- Git commit:
- Date and environment:
- Exported JSONL SHA-256:
- Ordinary opaque target result:
- Boundary-material result:
- Registration result:
- Dynamic alert result:
- Recovery and thermal result:
- Five-minute run result:
- Final conclusion: `not run / pass / limited pass / fail`
- Allowed distance/material scope if limited:
- Follow-up defects:

If registration, metric error, latency or incorrect numeric speech fails a key
threshold, stop before any walking experiment and correct the fusion or policy.
