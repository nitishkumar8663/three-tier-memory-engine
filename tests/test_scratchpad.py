"""Tier 1 scratchpad tests: scoping, isolation, clearing, search."""

import unittest

from mem3.errors import SessionError
from mem3.scratchpad import Scratchpad


class TestScratchpad(unittest.TestCase):
    def setUp(self):
        self.sp = Scratchpad()

    def test_put_get_roundtrip(self):
        self.sp.start_session("s1")
        self.sp.put("s1", "k", "v", kind="fact")
        self.assertEqual(self.sp.get("s1", "k"), "v")

    def test_get_missing_returns_default(self):
        self.sp.start_session("s1")
        self.assertIsNone(self.sp.get("s1", "nope"))
        self.assertEqual(self.sp.get("s1", "nope", "dflt"), "dflt")

    def test_sessions_isolated(self):
        self.sp.start_session("a")
        self.sp.start_session("b")
        self.sp.put("a", "k", "value-a")
        self.sp.put("b", "k", "value-b")
        self.assertEqual(self.sp.get("a", "k"), "value-a")
        self.assertEqual(self.sp.get("b", "k"), "value-b")

    def test_end_session_clears_and_returns_entries(self):
        self.sp.start_session("s1", now=1000.0)
        self.sp.put("s1", "a", "1", kind="fact", now=1001.0)
        self.sp.put("s1", "b", "2", kind="decision", now=1002.0)
        entries = self.sp.end_session("s1")
        self.assertEqual([e["key"] for e in entries], ["a", "b"])
        self.assertFalse(self.sp.has_session("s1"))
        with self.assertRaises(SessionError):
            self.sp.get("s1", "a")

    def test_end_unknown_session_raises(self):
        with self.assertRaises(SessionError):
            self.sp.end_session("ghost")

    def test_put_unknown_session_raises(self):
        with self.assertRaises(SessionError):
            self.sp.put("ghost", "k", "v")

    def test_duplicate_start_raises(self):
        self.sp.start_session("s1")
        with self.assertRaises(SessionError):
            self.sp.start_session("s1")

    def test_reput_keeps_created_at_updates_value(self):
        self.sp.start_session("s1")
        first = self.sp.put("s1", "k", "v1", now=1000.0)
        second = self.sp.put("s1", "k", "v2", now=2000.0)
        self.assertEqual(second["created_at"], 1000.0)
        self.assertEqual(second["updated_at"], 2000.0)
        self.assertEqual(self.sp.get("s1", "k"), "v2")
        self.assertEqual(first["created_at"], second["created_at"])

    def test_list_by_kind(self):
        self.sp.start_session("s1")
        self.sp.put("s1", "a", "1", kind="fact", now=1.0)
        self.sp.put("s1", "b", "2", kind="decision", now=2.0)
        self.sp.put("s1", "c", "3", kind="fact", now=3.0)
        self.assertEqual([e["key"] for e in self.sp.list("s1", kind="fact")],
                         ["a", "c"])
        self.assertEqual(len(self.sp.list("s1")), 3)

    def test_search_case_insensitive(self):
        self.sp.start_session("s1")
        self.sp.put("s1", "launch-date", "Launch is 2026-12-01")
        self.sp.put("s1", "other", "unrelated")
        hits = self.sp.search("s1", "LAUNCH")
        self.assertEqual([e["key"] for e in hits], ["launch-date"])

    def test_delete(self):
        self.sp.start_session("s1")
        self.sp.put("s1", "k", "v")
        self.assertTrue(self.sp.delete("s1", "k"))
        self.assertFalse(self.sp.delete("s1", "k"))
        self.assertIsNone(self.sp.get("s1", "k"))

    def test_invalid_kind_rejected(self):
        self.sp.start_session("s1")
        with self.assertRaises(ValueError):
            self.sp.put("s1", "k", "v", kind="nonsense")

    def test_clear_session(self):
        self.sp.start_session("s1")
        self.sp.put("s1", "k", "v")
        self.sp.clear_session("s1")
        self.assertTrue(self.sp.has_session("s1"))
        self.assertEqual(self.sp.list("s1"), [])


if __name__ == "__main__":
    unittest.main()
