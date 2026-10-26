from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from smart_cane.mobile_cli import main


class MobileCliTests(unittest.TestCase):
    def test_insecure_desktop_server_passes_runtime_config(self) -> None:
        output = io.StringIO()
        error = io.StringIO()

        with (
            mock.patch("smart_cane.mobile_cli._local_ipv4_addresses", return_value=()),
            mock.patch("smart_cane.mobile_server.run_mobile_server") as runner,
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            result = main(
                [
                    "--insecure-http",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                    "--image-size",
                    "320",
                    "--minimum-confirmations",
                    "2",
                ]
            )

        self.assertEqual(result, 0)
        self.assertIn("http://127.0.0.1:8765", output.getvalue())
        self.assertIn("YOLO model: yolo26n.pt", output.getvalue())
        self.assertIn("cannot grant camera access", error.getvalue())
        config = runner.call_args.args[0]
        self.assertEqual(config.model, "yolo26n.pt")
        self.assertEqual(config.image_size, 320)
        self.assertEqual(config.minimum_confirmations, 2)
        self.assertEqual(config.distance_high_meters, 1.0)
        self.assertEqual(config.distance_medium_meters, 2.0)
        self.assertEqual(config.distance_hysteresis_meters, 0.15)
        self.assertIsNone(runner.call_args.kwargs["certfile"])

    def test_tls_paths_are_forwarded(self) -> None:
        output = io.StringIO()

        with (
            mock.patch(
                "smart_cane.mobile_cli._local_ipv4_addresses",
                return_value=("192.168.1.8",),
            ),
            mock.patch("pathlib.Path.is_file", return_value=True),
            mock.patch("smart_cane.mobile_server.run_mobile_server") as runner,
            contextlib.redirect_stdout(output),
        ):
            result = main(["--certfile", "server.crt", "--keyfile", "server.key"])

        self.assertEqual(result, 0)
        self.assertIn("https://localhost:8443", output.getvalue())
        self.assertIn("https://192.168.1.8:8443", output.getvalue())
        self.assertEqual(str(runner.call_args.kwargs["certfile"]), "server.crt")
        self.assertEqual(str(runner.call_args.kwargs["keyfile"]), "server.key")

    def test_runtime_server_error_is_reported(self) -> None:
        output = io.StringIO()
        error = io.StringIO()

        with (
            mock.patch("smart_cane.mobile_cli._local_ipv4_addresses", return_value=()),
            mock.patch(
                "smart_cane.mobile_server.run_mobile_server",
                side_effect=OSError("address already in use"),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            result = main(["--insecure-http"])

        self.assertEqual(result, 2)
        self.assertIn("address already in use", error.getvalue())


if __name__ == "__main__":
    unittest.main()
