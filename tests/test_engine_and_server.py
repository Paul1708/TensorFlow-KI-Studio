"""Integration tests for BrainEngine inference and the local StudioHTTPServer API."""

from __future__ import annotations

import base64
import http.client
import json
import threading
import unittest
from pathlib import Path

import numpy as np

from ki_studio.config import ModelMode
from ki_studio.core.engine import BrainEngine
from ki_studio.paths import StudioPaths
from ki_studio.server.http_server import StudioHTTPServer

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBrainEngineAndHTTPServer(unittest.TestCase):
    engine: BrainEngine
    server: StudioHTTPServer
    thread: threading.Thread
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        paths = StudioPaths(
            bundle_dir=REPO_ROOT,
            app_dir=REPO_ROOT,
            web_dir=REPO_ROOT / "src" / "ki_studio" / "web",
        )
        cls.engine = BrainEngine(paths=paths)
        cls.server = StudioHTTPServer(engine=cls.engine, host="127.0.0.1", port=0, paths=paths)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            req_headers = dict(headers or {})
            payload_bytes: bytes | None = None
            if isinstance(body, dict):
                payload_bytes = json.dumps(body).encode("utf-8")
                req_headers.setdefault("Content-Type", "application/json")
            elif isinstance(body, bytes):
                payload_bytes = body
            conn.request(method, path, body=payload_bytes, headers=req_headers)
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, resp.getheader("Content-Type", ""), data
        finally:
            conn.close()

    def test_bundled_models_loaded_and_predict_samples(self) -> None:
        status = self.engine.get_status()
        self.assertTrue(status["zahlen_bereit"])
        self.assertTrue(status["smiley_bereit"])

        for digit_file, expected_digit in (
            ("beispiel_zahl_3.png", "3"),
            ("beispiel_zahl_5.png", "5"),
            ("beispiel_zahl_8.png", "8"),
        ):
            res = self.engine.analyze_sample_file(
                ModelMode.DIGITS, REPO_ROOT / "samples" / digit_file, digit_file
            )
            self.assertFalse(res["empty"])
            self.assertEqual(res["prediction"]["label"], expected_digit)
            self.assertTrue(res["conv_layers"])
            self.assertTrue(res["hidden_layers"])
            prob_sum = sum(o["prob"] for o in res["outputs"])
            self.assertAlmostEqual(prob_sum, 100.0, delta=0.2)

        happy = self.engine.analyze_sample_file(
            ModelMode.SMILEY,
            REPO_ROOT / "samples" / "smiley_gluecklich.png",
            "smiley_gluecklich.png",
        )
        sad = self.engine.analyze_sample_file(
            ModelMode.SMILEY, REPO_ROOT / "samples" / "smiley_traurig.png", "smiley_traurig.png"
        )
        self.assertGreater(happy["prediction"]["happiness"], 60.0)
        self.assertLess(sad["prediction"]["happiness"], 40.0)

        blank_res = self.engine.analyze_raw_image(
            ModelMode.DIGITS, np.full((64, 64), 255, dtype=np.uint8)
        )
        self.assertTrue(blank_res["empty"])
        self.assertIsNone(blank_res["prediction"])

    def test_static_assets_and_path_traversal_guard(self) -> None:
        code_html, ctype_html, body_html = self._request("GET", "/")
        self.assertEqual(code_html, 200)
        self.assertIn("text/html", ctype_html)
        self.assertIn(b"/static/css/styles.css", body_html)
        self.assertIn(b"/static/js/app.js", body_html)

        code_css, ctype_css, _ = self._request("GET", "/static/css/styles.css")
        self.assertEqual(code_css, 200)
        self.assertIn("text/css", ctype_css)

        code_js, ctype_js, _ = self._request("GET", "/static/js/app.js")
        self.assertEqual(code_js, 200)
        self.assertIn("javascript", ctype_js)

        for path in ("/favicon.ico", "/icon-256.png", "/manifest.json"):
            code, _, _ = self._request("GET", path)
            self.assertEqual(code, 200)

        code_trav, _, _ = self._request("GET", "/static/../../pyproject.toml")
        self.assertIn(code_trav, (403, 404))

    def test_api_endpoints_and_error_handling(self) -> None:
        for endpoint in ("/api/boot", "/api/status", "/api/data/status", "/api/train/status"):
            code, ctype, raw = self._request("GET", endpoint)
            self.assertEqual(code, 200)
            self.assertIn("application/json", ctype)
            json.loads(raw.decode("utf-8"))

        code_hb, _, raw_hb = self._request("POST", "/api/heartbeat", {})
        self.assertEqual(code_hb, 200)
        self.assertEqual(json.loads(raw_hb.decode("utf-8")), {"ok": True})

        for mode in ("zahlen", "smiley"):
            code_rnd, _, raw_rnd = self._request("POST", "/api/random", {"mode": mode})
            self.assertEqual(code_rnd, 200)
            parsed = json.loads(raw_rnd.decode("utf-8"))
            self.assertFalse(parsed["empty"])
            self.assertIsNotNone(parsed["prediction"])

        code_sample, _, raw_sample = self._request(
            "POST", "/api/sample_file", {"mode": "zahlen", "filename": "beispiel_zahl_3.png"}
        )
        self.assertEqual(code_sample, 200)
        self.assertEqual(json.loads(raw_sample.decode("utf-8"))["prediction"]["label"], "3")

        code_bad_sample, _, _ = self._request(
            "POST", "/api/sample_file", {"mode": "zahlen", "filename": "../pyproject.toml"}
        )
        self.assertEqual(code_bad_sample, 404)

        sample_png = (REPO_ROOT / "samples" / "beispiel_zahl_5.png").read_bytes()
        b64_data = "data:image/png;base64," + base64.b64encode(sample_png).decode("ascii")
        code_analyze, _, raw_analyze = self._request(
            "POST", "/api/analyze", {"mode": "zahlen", "image_b64": b64_data}
        )
        self.assertEqual(code_analyze, 200)
        self.assertEqual(json.loads(raw_analyze.decode("utf-8"))["prediction"]["label"], "5")

        code_bad_json, _, _ = self._request("POST", "/api/analyze", b"{broken json")
        self.assertEqual(code_bad_json, 400)


if __name__ == "__main__":
    unittest.main()
