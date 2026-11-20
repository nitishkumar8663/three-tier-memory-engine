"""Tier 2 episodic compression tests: keep/drop/dedupe/TTL/search."""

import unittest

from mem3.episodic import EpisodicRecord, compress_session
from mem3.errors import MemoryError


def _entries(*pairs):
    # pairs: (kind, value)
    return [{"kind": k, "value": v} for k, v in pairs]


def _events(*pairs):
    return [{"kind": k, "text": v} for k, v in pairs]


class TestCompress(unittest.TestCase):
    def test_keeps_decisions_facts_outcomes(self):
        rec = compress_session(
            "s1", _entries(("decision", "Use SQLite first"),
                           ("fact", "Launch is 2026-12-01"),
                           ("outcome", "Prototype shipped")),
            [], 1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        self.assertEqual(rec.decisions, ["Use SQLite first"])
        self.assertEqual(rec.facts, ["Launch is 2026-12-01"])
        self.assertEqual(rec.outcomes, ["Prototype shipped"])
        self.assertEqual(rec.key_points,
                         ["Use SQLite first", "Launch is 2026-12-01",
                          "Prototype shipped"])

    def test_drops_chatter_and_filler(self):
        rec = compress_session(
            "s1", _entries(("fact", "Budget is £500")),
            _events(("chatter", "lol ok"), ("chatter", "thanks!"),
                    ("note", "hmm")),
            1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        self.assertEqual(rec.facts, ["Budget is £500"])
        self.assertEqual(rec.dropped_count, 3)
        self.assertEqual(rec.source_count, 4)

    def test_dedupes_repeated_text(self):
        rec = compress_session(
            "s1", _entries(("fact", "Launch is 2026-12-01"),
                           ("fact", "launch is 2026-12-01!"),
                           ("fact", "Launch is 2026-12-01")),
            [], 1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        self.assertEqual(rec.facts, ["Launch is 2026-12-01"])
        self.assertEqual(rec.dropped_count, 2)

    def test_fact_like_note_kept_plain_note_dropped(self):
        rec = compress_session(
            "s1", [],
            _events(("note", "Ravi agreed to own the migration"),
                    ("note", "nice weather today")),
            1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        self.assertEqual(rec.facts, ["Ravi agreed to own the migration"])
        self.assertEqual(rec.dropped_count, 1)

    def test_note_with_number_kept(self):
        rec = compress_session(
            "s1", [], _events(("note", "The bill came to 42 pounds")),
            1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        self.assertEqual(rec.facts, ["The bill came to 42 pounds"])

    def test_all_chatter_raises(self):
        with self.assertRaises(MemoryError):
            compress_session(
                "s1", [], _events(("chatter", "lol"), ("chatter", "ok")),
                1000.0, 2000.0, ttl_seconds=3600, now=2000.0)

    def test_empty_session_raises(self):
        with self.assertRaises(MemoryError):
            compress_session("s1", [], [], 1000.0, 2000.0,
                             ttl_seconds=3600, now=2000.0)

    def test_decision_order_first_seen(self):
        rec = compress_session(
            "s1", _entries(("fact", "Fact one"),
                           ("decision", "Decision one"),
                           ("fact", "Fact two")),
            [], 1000.0, 2000.0, ttl_seconds=3600, now=2000.0)
        # Decisions bucket first in key_points regardless of input order.
        self.assertEqual(rec.key_points,
                         ["Decision one", "Fact one", "Fact two"])


class TestEpisodicRecord(unittest.TestCase):
    def _rec(self, ttl=3600, now=2000.0):
        return EpisodicRecord("s1", 1000.0, 2000.0,
                              ["Decide X"], ["Fact Y"], [],
                              dropped_count=1, source_count=3,
                              ttl_seconds=ttl, created_at=now)

    def test_ttl_expiry(self):
        rec = self._rec(ttl=100, now=1000.0)
        self.assertFalse(rec.is_expired(1099.0))
        self.assertTrue(rec.is_expired(1100.0))
        self.assertTrue(rec.is_expired(2000.0))

    def test_no_ttl_never_expires(self):
        rec = self._rec(ttl=None)
        self.assertFalse(rec.is_expired(10 ** 12))

    def test_matches_keyword_and_session(self):
        rec = self._rec()
        self.assertTrue(rec.matches("decide x"))
        self.assertTrue(rec.matches("FACT y"))
        self.assertTrue(rec.matches("s1"))
        self.assertFalse(rec.matches("unrelated"))

    def test_roundtrip_dict(self):
        rec = self._rec()
        back = EpisodicRecord.from_dict(rec.to_dict())
        self.assertEqual(back.to_dict(), rec.to_dict())
        self.assertEqual(back.key_points, rec.key_points)


if __name__ == "__main__":
    unittest.main()
