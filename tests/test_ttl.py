"""TTL sweeper + eviction-log tests (engine level)."""

import os
import tempfile
import unittest

from mem3 import MemoryEngine


class TestSweeper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Short idle TTL so orphaned scratchpads can be exercised fast.
        self.eng = MemoryEngine(self.tmp.name, session_idle_ttl=100,
                                semantic_base_url="http://127.0.0.1:9")

    def tearDown(self):
        self.tmp.cleanup()

    def _session_with_record(self, sid, ttl, now):
        self.eng.start_session(sid, now=now)
        self.eng.remember(sid, "k", f"value for {sid}", kind="fact",
                          now=now)
        return self.eng.end_session(sid, ttl_seconds=ttl, now=now)

    def test_expired_episodic_evicted_live_kept(self):
        self._session_with_record("short", ttl=10, now=1000.0)
        self._session_with_record("long", ttl=100000, now=1000.0)
        events = self.eng.sweep(now=1011.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tier"], "episodic")
        self.assertEqual(events[0]["record_id"], "short")
        kept = [r.session_id for r in self.eng.episodic]
        self.assertEqual(kept, ["long"])

    def test_eviction_logged_to_file(self):
        self._session_with_record("short", ttl=10, now=1000.0)
        self.eng.sweep(now=2000.0)
        log = self.eng.eviction_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["tier"], "episodic")
        self.assertIn("expired", log[0]["reason"])
        # Log survives an engine reload.
        eng2 = MemoryEngine(self.tmp.name, session_idle_ttl=100,
                            semantic_base_url="http://127.0.0.1:9")
        self.assertEqual(len(eng2.eviction_log()), 1)

    def test_nothing_expired_nothing_evicted(self):
        self._session_with_record("long", ttl=100000, now=1000.0)
        self.assertEqual(self.eng.sweep(now=1010.0), [])
        self.assertEqual(self.eng.eviction_log(), [])

    def test_idle_scratchpad_purged_and_logged(self):
        self.eng.start_session("idle", now=1000.0)
        self.eng.remember("idle", "k", "never compressed", now=1000.0)
        events = self.eng.sweep(now=2000.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tier"], "scratchpad")
        self.assertEqual(events[0]["record_id"], "idle")
        self.assertFalse(self.eng.scratchpad.has_session("idle"))
        self.assertEqual(self.eng.episodic, [])  # never compressed

    def test_active_scratchpad_not_purged(self):
        self.eng.start_session("active", now=1000.0)
        self.eng.remember("active", "k", "v", now=1000.0)
        self.assertEqual(self.eng.sweep(now=1050.0), [])
        self.assertTrue(self.eng.scratchpad.has_session("active"))

    def test_semantic_records_never_evicted(self):
        self.eng.add_semantic_record("Spark memory with bge-m3",
                                     title="mem")
        self._session_with_record("short", ttl=10, now=1000.0)
        self.eng.sweep(now=5000.0)
        self.assertEqual(len(self.eng.semantic.local_records), 1)
        results, source = self.eng.semantic.recall("Spark memory bge-m3")
        self.assertEqual(source, "local-fallback")
        self.assertEqual(len(results), 1)

    def test_evicted_record_not_recalled(self):
        self._session_with_record("short", ttl=10, now=1000.0)
        self.eng.sweep(now=2000.0)
        self.assertEqual(self.eng.recall_episodic(query="value"), [])

    def test_recall_skips_expired_even_before_sweep(self):
        self._session_with_record("short", ttl=10, now=1000.0)
        hits = self.eng.recall("value for short", now=2000.0)
        self.assertEqual([h for h in hits if h.tier == "episodic"], [])


if __name__ == "__main__":
    unittest.main()
