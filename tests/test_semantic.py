"""Tier 3 semantic adapter tests.

The service leg is tested against a tiny in-process fake HTTP server
that mimics the real memory service's /health and /recall shapes
(read from services/memory-service/memory-service.py). The fallback
leg is tested with the service pointed at a closed port.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mem3.semantic import SemanticAdapter

CLOSED_PORT_URL = "http://127.0.0.1:9"  # nothing listens here


class _FakeService(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/recall":
            self._send(200, {"results": [
                {"id": 7, "type": "note", "title": "Spark memory",
                 "text": "PARA notes with bge-m3 embeddings",
                 "path": "para/memory.md", "date": "2026-11-20",
                 "author": "muse", "source": "para", "salience": 8,
                 "score": 0.91, "relevance": 0.9, "recency": 0.8,
                 "importance": 0.8},
            ]})
        else:
            self._send(404, {"error": "not found"})


class TestFallback(unittest.TestCase):
    def setUp(self):
        self.ad = SemanticAdapter(base_url=CLOSED_PORT_URL, timeout=2.0)
        self.ad.add_local_record(
            "The Spark memory service stores PARA notes with bge-m3.",
            title="Spark memory service")
        self.ad.add_local_record(
            "Wardrobe app v1.1 was validated live on the emulator.",
            title="Wardrobe validation")

    def test_service_down_reports_down(self):
        self.assertFalse(self.ad.is_service_up())

    def test_fallback_engages_when_service_down(self):
        results, source = self.ad.recall("Spark memory bge-m3")
        self.assertEqual(source, "local-fallback")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Spark memory service")
        self.assertEqual(results[0]["source"], "local-fallback")

    def test_fallback_ranks_by_overlap(self):
        results, _ = self.ad.recall("Wardrobe emulator validation")
        self.assertEqual(results[0]["title"], "Wardrobe validation")

    def test_fallback_no_match_returns_empty(self):
        results, source = self.ad.recall("quantum knitting patterns")
        self.assertEqual(source, "local-fallback")
        self.assertEqual(results, [])

    def test_fallback_ignores_stopwords_only_query(self):
        results, _ = self.ad.recall("what is the")
        self.assertEqual(results, [])

    def test_fallback_top_k_respected(self):
        for i in range(5):
            self.ad.add_local_record(f"Spark memory note number {i}",
                                     title=f"note {i}")
        results, _ = self.ad.recall("Spark memory", top_k=2)
        self.assertEqual(len(results), 2)


class TestServiceLeg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeService)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.ad = SemanticAdapter(
            base_url=f"http://127.0.0.1:{self.port}", timeout=5.0)

    def test_service_up(self):
        self.assertTrue(self.ad.is_service_up())

    def test_service_recall_shape(self):
        results, source = self.ad.recall("bge-m3", top_k=5)
        self.assertEqual(source, "service")
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["id"], 7)
        self.assertEqual(r["title"], "Spark memory")
        self.assertEqual(r["text"], "PARA notes with bge-m3 embeddings")
        self.assertEqual(r["score"], 0.91)
        self.assertEqual(r["source"], "service")
        self.assertEqual(r["author"], "muse")

    def test_service_preferred_over_fallback(self):
        self.ad.add_local_record("bge-m3 local decoy", title="decoy")
        results, source = self.ad.recall("bge-m3")
        self.assertEqual(source, "service")
        self.assertEqual(results[0]["title"], "Spark memory")


if __name__ == "__main__":
    unittest.main()
