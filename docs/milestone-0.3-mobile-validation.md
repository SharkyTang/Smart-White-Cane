# Milestone 0.3 mobile validation

Date: 2026-08-23

## Acceptance scope

- iPhone Safari rear-camera permission and live preview over trusted local HTTPS
- JPEG frames processed in memory by the existing Mac YOLO and decision pipeline
- No overlapping uploads or stale-frame queue
- Stable boxes and left / centre / right risk state rendered on the phone
- Chinese speech and alert tone only after a committed alert event
- Session reset, stop/start, network failure and recovery
- No camera-frame files written by the mobile service

## Automated and desktop checks

Run:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
git diff --check
uv build
```

The HTTP adapter tests use generated in-memory bytes and fake detector results. They verify request limits, JPEG rejection, serialized inference, session reset, response shape and alert commit-after-delivery without loading a model or opening a camera.

The desktop smoke test may use plain loopback HTTP because it checks static assets and API wiring only:

```bash
uv run smart-cane-mobile --insecure-http --host 127.0.0.1
curl -fsS http://127.0.0.1:8443/healthz
```

Completed result:

- `88` unit and loopback integration tests passed.
- Ruff lint, Ruff format check, JavaScript syntax check, shell syntax check and `git diff --check` passed.
- The `0.3.0` source distribution and wheel built successfully; the wheel contains `mobile/index.html`, `mobile/styles.css`, `mobile/app.js`, `mobile_cli.py` and `mobile_server.py`.
- The generated local CA verified the server certificate. An HTTPS `/healthz` request using that CA returned `{"status":"ok"}`.
- The real `yolo11n.pt` service accepted a JPEG over the frame API. Cold decode / inference / total were approximately `6.9 / 1413.4 / 1420.3 ms`; the immediate second request was approximately `2.6 / 20.2 / 22.8 ms`. This is an API wiring smoke test, not a mobile latency result.
- The in-app browser loaded all three local assets with no console warnings or errors. At `390 × 844` and `320 × 568` CSS-pixel viewports there was no horizontal overflow; at 320 px width all control buttons remained 56 px high and at least 139 px wide.

## Physical iPhone checklist

This section requires the actual iPhone and is not satisfied by a desktop browser or automated test.

- [ ] The locally generated CA is installed and fully trusted; Safari shows no certificate warning.
- [ ] Safari requests camera permission and selects the rear camera.
- [ ] “开始检测” starts a live, correctly oriented preview without fullscreen takeover.
- [ ] A known indoor object produces an aligned box and risk/direction label.
- [ ] A stable medium/high event produces one Chinese spoken warning and one tone.
- [ ] Repeated unchanged detections respect the cooldown instead of speaking every frame.
- [ ] Stop/start begins a fresh session without carrying old tracks or cooldown history.
- [ ] Wi-Fi interruption shows an explicit disconnected state; recovery does not replay queued frames.
- [ ] Locking the phone or moving Safari to the background pauses the test and is explained onscreen.
- [ ] Mac inspection confirms that the mobile service created no image or video files.

Test indoors with a stationary helper or static objects. Do not walk on roads, stairs, kerbs or other hazardous areas while looking at or relying on this prototype.

## Result boundary

Passing the checklist demonstrates only that an iPhone can act as a local camera and feedback client. It does not validate physical distance, drop-off detection, outdoor robustness, accessibility with VoiceOver, battery life, thermal behavior or navigation safety.
