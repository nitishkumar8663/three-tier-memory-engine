"""Unified recall + episodic search + persistence tests (engine level)."""

import tempfile
import time
import unittest

from mem3 import MemoryEngine
from mem3.errors import SessionError

# Realistic epoch base: the default 30-day episodic TTL is measured
# against wall-clock time, so fake 1970-era timestamps would make
# every record look expired. Use "now" throughout.
NOW = time.time()


class TestUnifiedRecall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = MemoryEngine(self.tmp.name,
                                semantic_base_url="http://127.0.0.1:9")
        self.eng.add_semantic_record(
            "The Spark memory service stores PARA notes with bge-m3.",
            title="Spark memory service")

    def tearDown(self):
        self.tmp.cleanup()

    def test_recall_order_scratchpad_episodic_semantic(self):
        self.eng.start_session("s1", now=NOW)
        self.eng.remember("s1", "launch", "Launch is 2026-12-01",
                          kind="fact", now=NOW + 1)
        self.eng.end_session("s1", now=NOW + 2)
        self.eng.start_session("s2", now=NOW + 3)
        self.eng.remember("s2", "launch", "Launch is 2026-12-01",
                          kind="fact", now=NOW + 4)
        tiers = [h.tier for h in self.eng.recall("launch")]
        # Semantic leg has no "launch" overlap, so exactly two tiers hit.
        self.assertEqual(tiers, ["scratchpad", "episodic"])
        # Scratchpad hit carries its key + session.
        self.assertEqual(tiers[0], "scratchpad")
        self.assertEqual(
            [h for h in self.eng.recall("launch")
             if h.tier == "scratchpad"][0].key, "launch")

    def test_recall_scoped_to_session(self):
        self.eng.start_session("a", now=NOW)
        self.eng.remember("a", "k", "apple pie", now=NOW + 1)
        self.eng.start_session("b", now=NOW)
        self.eng.remember("b", "k", "banana bread", now=NOW + 1)
        hits = self.eng.recall("apple", session_id="a")
        self.assertEqual([h.text for h in hits
                          if h.tier == "scratchpad"], ["apple pie"])
        hits_b = self.eng.recall("apple", session_id="b")
        self.assertEqual([h for h in hits_b if h.tier == "scratchpad"], [])

    def test_recall_marks_semantic_fallback_source(self):
        hits = self.eng.recall("Spark memory bge-m3")
        sem = [h for h in hits if h.tier == "semantic"]
        self.assertEqual(len(sem), 1)
        self.assertEqual(sem[0].source, "local-fallback")
        self.assertEqual(sem[0].title, "Spark memory service")

    def test_end_session_with_only_chatter_leaves_no_record(self):
        self.eng.start_session("noisy", now=NOW)
        self.eng.note("noisy", "lol", kind="chatter", now=NOW + 1)
        rec = self.eng.end_session("noisy", now=NOW + 2)
        self.assertIsNone(rec)
        self.assertEqual(self.eng.episodic, [])
        self.assertFalse(self.eng.scratchpad.has_session("noisy"))

    def test_recall_episodic_filters(self):
        for sid, text in (("s1", "Launch is 2026-12-01"),
                          ("s2", "Budget is £500")):
            self.eng.start_session(sid, now=NOW)
            self.eng.remember(sid, "k", text, kind="fact", now=NOW + 1)
            self.eng.end_session(sid, now=NOW + 2)
        self.assertEqual(
            [r.session_id for r in
             self.eng.recall_episodic(session_id="s2")], ["s2"])
        self.assertEqual(
            [r.session_id for r in
             self.eng.recall_episodic(query="budget")], ["s2"])
        self.assertEqual(
            [r.session_id for r in self.eng.recall_episodic(query="zzz")],
            [])
        # Time window: both created at NOW + 2; a window after that is empty.
        self.assertEqual(
            self.eng.recall_episodic(since=NOW + 10000), [])

    def test_remember_unknown_session_raises(self):
        with self.assertRaises(SessionError):
            self.eng.remember("ghost", "k", "v")

    def test_persistence_roundtrip(self):
        self.eng.start_session("s1", now=NOW)
        self.eng.remember("s1", "k", "persistent fact", kind="fact",
                          now=NOW + 1)
        rec = self.eng.end_session("s1", now=NOW + 2)
        self.assertIsNotNone(rec)
        eng2 = MemoryEngine(self.tmp.name,
                            semantic_base_url="http://127.0.0.1:9")
        self.assertEqual(len(eng2.episodic), 1)
        self.assertEqual(eng2.episodic[0].session_id, "s1")
        self.assertEqual(eng2.episodic[0].facts, ["persistent fact"])
        self.assertEqual(len(eng2.semantic.local_records), 1)
        # Live scratchpad also survives a reload.
        self.eng.start_session("live", now=NOW + 3)
        self.eng.remember("live", "k2", "still here", now=NOW + 4)
        eng3 = MemoryEngine(self.tmp.name,
                            semantic_base_url="http://127.0.0.1:9")
        self.assertEqual(eng3.scratchpad.get("live", "k2"), "still here")


if __name__ == "__main__":
    unittest.main()
