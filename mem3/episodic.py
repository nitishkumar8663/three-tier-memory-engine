"""Tier 2: episodic session compression.

When a session ends, its scratchpad entries + session events are
compressed into one compact episodic summary record. The compression
is deterministic and extractive. No LLM, no embeddings:

- Dedupe: entries/events with the same normalised text are kept once
  (first occurrence wins).
- Keep: kinds ``decision``, ``fact``, ``outcome``. The things a later
  session needs to know.
- Keep (conditional): ``note`` entries are kept only if they carry a
  fact-like signal (a number, a proper-noun-ish capitalised token, or
  one of the decision/fact marker words); bare notes are dropped.
- Drop: kind ``chatter`` and filler one-liners ("ok", "thanks", "lol",
  "hmm", …). Session noise, not memory.
- Order: decisions first, then facts, then outcomes, each in
  first-seen order; ``key_points`` is the merged list in that order.

Each record carries a TTL (``expires_at``); the sweeper in
:mod:`mem3.engine` evicts expired records and logs the eviction.
"""

import re
import time

from mem3.errors import MemoryError

KEEP_KINDS = ("decision", "fact", "outcome")

# Filler one-liners: session noise, never memory.
_FILLER = {
    "ok", "okay", "thanks", "thank you", "thx", "lol", "hmm", "hm",
    "yes", "no", "sure", "got it", "noted", "cool", "great", "nice",
    "bye", "hi", "hello", "hey", "brb", ":-)", ":)", "👍",
}

# Words that mark a note as fact-bearing rather than small talk.
_MARKER_WORDS = {
    "decided", "decision", "agreed", "chose", "choose", "picked",
    "must", "should", "will", "won't", "cannot", "can't", "blocked",
    "deadline", "cost", "price", "budget", "risk", "depends",
    "because", "therefore", "result", "failed", "succeeded", "fixed",
}


def _normalise(text):
    """Lowercase, collapse whitespace, strip punctuation (for dedupe)."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_filler(text):
    return _normalise(text) in _FILLER


def _note_is_fact_like(text):
    words = set(_normalise(text).split())
    if words & _MARKER_WORDS:
        return True
    if re.search(r"\d", text):  # a number: a date, price, count, version
        return True
    # A capitalised token mid-sentence (a name, place, product).
    tokens = str(text).split()
    if any(t[:1].isupper() and i > 0 for i, t in enumerate(tokens)):
        return True
    return False


def _classify(item):
    """Return the bucket ('decisions'|'facts'|'outcomes') or None to drop."""
    kind = item.get("kind", "note")
    text = str(item.get("text", item.get("value", ""))).strip()
    if not text or _is_filler(text):
        return None
    if kind == "chatter":
        return None
    if kind in KEEP_KINDS:
        return {"decision": "decisions", "fact": "facts",
                "outcome": "outcomes"}[kind]
    if kind == "note":
        return "facts" if _note_is_fact_like(text) else None
    return None


class EpisodicRecord:
    """One compressed session summary. JSON-serialisable via ``to_dict``."""

    def __init__(self, session_id, started_at, ended_at, decisions,
                 facts, outcomes, dropped_count, source_count,
                 ttl_seconds, created_at=None):
        self.session_id = session_id
        self.started_at = started_at
        self.ended_at = ended_at
        self.decisions = list(decisions)
        self.facts = list(facts)
        self.outcomes = list(outcomes)
        self.dropped_count = dropped_count
        self.source_count = source_count
        self.created_at = created_at if created_at is not None else time.time()
        self.expires_at = (
            self.created_at + ttl_seconds if ttl_seconds is not None else None
        )

    @property
    def key_points(self):
        return self.decisions + self.facts + self.outcomes

    def is_expired(self, now=None):
        now = time.time() if now is None else now
        return self.expires_at is not None and now >= self.expires_at

    def matches(self, query):
        """Keyword match (case-insensitive substring) over the summary."""
        q = query.lower()
        if q in self.session_id.lower():
            return True
        return any(q in point.lower() for point in self.key_points)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "decisions": self.decisions,
            "facts": self.facts,
            "outcomes": self.outcomes,
            "key_points": self.key_points,
            "dropped_count": self.dropped_count,
            "source_count": self.source_count,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data):
        rec = cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            decisions=data.get("decisions", []),
            facts=data.get("facts", []),
            outcomes=data.get("outcomes", []),
            dropped_count=data.get("dropped_count", 0),
            source_count=data.get("source_count", 0),
            ttl_seconds=None,
            created_at=data.get("created_at"),
        )
        rec.expires_at = data.get("expires_at")
        return rec


def compress_session(session_id, entries, events, started_at, ended_at,
                     ttl_seconds, now=None):
    """Compress one finished session into an :class:`EpisodicRecord`.

    *entries*: scratchpad entries (dicts with ``value``/``kind``);
    *events*: session events (dicts with ``text``/``kind``).
    Raises :class:`MemoryError` if the session produced nothing
    keepable. An empty summary is not a memory.
    """
    items = []
    for e in entries:
        items.append({"kind": e.get("kind", "fact"),
                      "text": str(e.get("value", ""))})
    for ev in events:
        items.append({"kind": ev.get("kind", "note"),
                      "text": str(ev.get("text", ""))})

    seen = set()
    buckets = {"decisions": [], "facts": [], "outcomes": []}
    dropped = 0
    for item in items:
        text = item["text"].strip()
        if not text:
            dropped += 1
            continue
        norm = _normalise(text)
        if norm in seen:
            dropped += 1  # duplicate
            continue
        seen.add(norm)
        bucket = _classify(item)
        if bucket is None:
            dropped += 1
            continue
        buckets[bucket].append(text)

    total_kept = sum(len(v) for v in buckets.values())
    if total_kept == 0:
        raise MemoryError(
            f"session {session_id!r} produced nothing worth remembering "
            f"({len(items)} items, all chatter/filler/duplicates)"
        )
    return EpisodicRecord(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        decisions=buckets["decisions"],
        facts=buckets["facts"],
        outcomes=buckets["outcomes"],
        dropped_count=dropped,
        source_count=len(items),
        ttl_seconds=ttl_seconds,
        created_at=now if now is not None else time.time(),
    )
