"""Command-line entry point for the Safari and native iPhone test server."""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

from smart_cane.config import DEFAULT_MODEL, RuntimeConfig
from smart_cane.perception.detector import DetectorError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-cane-mobile",
        description=("Serve the local iPhone RGB/RGB-D APIs and run YOLO inference on this Mac."),
    )
    parser.add_argument("--host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8443, help="listen port (default: 8443)")
    parser.add_argument("--certfile", type=Path, default=None, help="TLS server certificate")
    parser.add_argument("--keyfile", type=Path, default=None, help="TLS private key")
    parser.add_argument(
        "--insecure-http",
        action="store_true",
        help="serve plain HTTP for desktop debugging; iPhone camera access will not work",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model or weight path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="minimum detector confidence (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device such as cpu or mps (default: measured CPU path)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        metavar="PIXELS",
        help="YOLO square inference size (default: 640)",
    )
    parser.add_argument(
        "--confirmation-window",
        type=int,
        default=5,
        metavar="N",
        help="temporal observation window (default: 5)",
    )
    parser.add_argument(
        "--minimum-confirmations",
        type=int,
        default=3,
        metavar="N",
        help="observations required for a stable target (default: 3)",
    )
    parser.add_argument(
        "--high-risk-confirmations",
        type=int,
        default=2,
        metavar="N",
        help="consecutive large-target observations (default: 2)",
    )
    parser.add_argument(
        "--high-risk-area-ratio",
        type=float,
        default=0.18,
        metavar="RATIO",
        help="large-target and high-risk area threshold (default: 0.18)",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=3,
        metavar="N",
        help="consecutive misses before target removal (default: 3)",
    )
    parser.add_argument(
        "--direction-hysteresis",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="left/centre/right boundary hysteresis (default: 0.05)",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="repeat cooldown for one tracked warning (default: 3.0)",
    )
    parser.add_argument(
        "--distance-high",
        type=float,
        default=1.0,
        metavar="METERS",
        help="LiDAR high-risk entry distance (default: 1.0)",
    )
    parser.add_argument(
        "--distance-medium",
        type=float,
        default=2.0,
        metavar="METERS",
        help="LiDAR medium-risk entry distance (default: 2.0)",
    )
    parser.add_argument(
        "--distance-hysteresis",
        type=float,
        default=0.15,
        metavar="METERS",
        help="extra distance required before lowering a LiDAR risk band (default: 0.15)",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=5.0,
        metavar="METERS",
        help="largest LiDAR value accepted as usable evidence (default: 5.0)",
    )
    return parser


def _default_certificate_paths() -> tuple[Path, Path]:
    project_dir = Path(__file__).resolve().parents[2]
    certificate_dir = project_dir / ".mobile-certs"
    return certificate_dir / "mobile-server.crt", certificate_dir / "mobile-server.key"


def _local_ipv4_addresses() -> tuple[str, ...]:
    candidates: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.add(item[4][0])
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            candidates.add(probe.getsockname()[0])
    except OSError:
        pass

    usable = []
    for value in candidates:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if address.version == 4 and not address.is_loopback and not address.is_link_local:
            usable.append(value)
    return tuple(sorted(usable))


def _resolve_tls_paths(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, Path] | None:
    if args.insecure_http:
        if args.certfile is not None or args.keyfile is not None:
            parser.error("--insecure-http cannot be combined with --certfile or --keyfile")
        return None

    default_cert, default_key = _default_certificate_paths()
    certfile = args.certfile or default_cert
    keyfile = args.keyfile or default_key
    if not certfile.is_file() or not keyfile.is_file():
        parser.error(
            "trusted HTTPS certificates are missing; run "
            "`scripts/create_mobile_cert.sh` first, or use --insecure-http only for desktop tests"
        )
    return certfile, keyfile


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    tls_paths = _resolve_tls_paths(args, parser)

    try:
        config = RuntimeConfig(
            model=args.model,
            confidence=args.confidence,
            device=args.device,
            image_size=args.image_size,
            confirmation_window=args.confirmation_window,
            minimum_confirmations=args.minimum_confirmations,
            high_risk_area_ratio=args.high_risk_area_ratio,
            high_risk_consecutive_frames=args.high_risk_confirmations,
            max_missing_frames=args.max_missing_frames,
            direction_hysteresis=args.direction_hysteresis,
            alert_cooldown_seconds=args.alert_cooldown,
            distance_high_meters=args.distance_high,
            distance_medium_meters=args.distance_medium,
            distance_hysteresis_meters=args.distance_hysteresis,
            distance_max_meters=args.distance_max,
            audio_enabled=False,
            display=False,
        )
    except ValueError as error:
        parser.error(str(error))

    from smart_cane.mobile_server import run_mobile_server

    scheme = "https" if tls_paths is not None else "http"
    wildcard_host = args.host in {"0.0.0.0", "::"}
    display_host = "localhost" if wildcard_host else args.host
    print(f"Smart Cane mobile test server: {scheme}://{display_host}:{args.port}")
    print(
        f"YOLO model: {config.model} · image size: {config.image_size} · "
        f"confidence: {config.confidence:.2f}"
    )
    if wildcard_host:
        for address in _local_ipv4_addresses():
            print(f"iPhone client: {scheme}://{address}:{args.port}")
    if tls_paths is None:
        print(
            "Warning: iPhone Safari cannot grant camera access on this HTTP origin.",
            file=sys.stderr,
        )

    try:
        run_mobile_server(
            config,
            host=args.host,
            port=args.port,
            certfile=tls_paths[0] if tls_paths is not None else None,
            keyfile=tls_paths[1] if tls_paths is not None else None,
        )
    except (DetectorError, OSError, RuntimeError) as error:
        print(f"smart-cane-mobile: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
