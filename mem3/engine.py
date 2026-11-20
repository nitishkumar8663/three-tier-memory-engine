"""Unified 3-tier memory engine.

One object over the three tiers:

- :meth:`MemoryEngine.remember`: write a fact/decision/outcome/note
  into the active session's scratchpad (Tier 1), and optionally log
  a session event for the episodic compressor.
- :meth:`MemoryEngine.recall`: unified recall across tiers, in
  order: scratchpad (live session) → episodic summaries → semantic
  (service, or local fallback). Each hit is a :class:`RecallResult`
  naming its tier and score, so callers know where it came from.
- :meth:`MemoryEngine.end_session`: compress Tier 1 + events into a
  Tier 2 episodic record (TTL'd), then clear the scratchpad.
- :meth:`MemoryEngine.sweep`: enforce per-tier retention: scratchpad
  entries die with their session (and orphaned scratchpads older
  than ``session_idle_ttl`` are purged); expired episodic records are
  evicted and logged; semantic records are NEVER evicted by this
  engine (they belong to the common memory service).

Persistence: the engine's own data directory (never the live memory
store) holds ``scratchpad.json``, ``episodic.json``, ``semantic.json``
and ``eviction.jsonl``. JSON writes are atomic (see
:mod:`mem3.store`).
"""

import os
import time
from dataclasses import dataclass, field

from mem3 import store as _store
from mem3.episodic import EpisodicRecord, compress_session
from mem3.errors import MemoryError, SessionError
from mem3.scratchpad import Scratchpad
from mem3.semantic import SemanticAdapter

# Defaults: overridable per engine / per record.
DEFAULT_EPISODIC_TTL_SECONDS = 30 * 24 * 3600   # 30 days
DEFAULT_SESSION_IDLE_TTL_SECONDS = 24 * 3600   # 24 h


@dataclass
class RecallResult:
    tier: str            # "scratchpad" | "episodic" | "semantic"
    text: str
    score: float
    session_id: str = None
    key: str = None
    kind: str = None
    source: str = None   # semantic leg: "service" | "local-fallback"
    record_id: object = None
    title: str = None
    extra: dict = field(default_factory=dict)


class MemoryEngine:
    def __init__(self, data_dir, episodic_ttl=DEFAULT_EPISODIC_TTL_SECONDS,
                 session_idle_ttl=DEFAULT_SESSION_IDLE_TTL_SECONDS,
                 semantic_base_url=None):
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.episodic_ttl = episodic_ttl
        self.session_idle_ttl = session_idle_ttl
        self.scratchpad = Scratchpad()
        self.semantic = SemanticAdapter(
            **({"base_url": semantic_base_url}
               if semantic_base_url else {}))
        self.episodic = []       # list[EpisodicRecord]
        self._events = {}        # session_id -> list of event dicts
        self._load()

    # -- persistence -------------------------------------------------

    def _paths(self):
        d = self.data_dir
        return {
            "scratchpad": os.path.join(d, "scratchpad.json"),
            "episodic": os.path.join(d, "episodic.json"),
            "semantic": os.path.join(d, "semantic.json"),
            "events": os.path.join(d, "events.json"),
            "eviction": os.path.join(d, "eviction.jsonl"),
        }

    def _load(self):
        p = self._paths()
        self.scratchpad.from_state(
            _store.read_json(p["scratchpad"], {}))
        self.episodic = [
            EpisodicRecord.from_dict(r)
            for r in _store.read_json(p["episodic"], [])
        ]
        self.semantic.from_state(_store.read_json(p["semantic"], {}))
        self._events = _store.read_json(p["events"], {})

    def _save(self):
        p = self._paths()
        _store.atomic_write_json(p["scratchpad"],
                                 self.scratchpad.to_state())
        _store.atomic_write_json(
            p["episodic"], [r.to_dict() for r in self.episodic])
        _store.atomic_write_json(p["semantic"], self.semantic.to_state())
        _store.atomic_write_json(p["events"], self._events)

    def save(self):
        """Persist all tiers (also called automatically on mutation)."""
        self._save()

    # -- sessions -----------------------------------------------------

    def start_session(self, session_id, now=None):
        self.scratchpad.start_session(session_id, now=now)
        self._events.setdefault(session_id, [])
        self._save()
        return session_id

    def note(self, session_id, text, kind="chatter", now=None):
        """Log a session event (utterance/note) for the compressor."""
        if session_id not in self._events:
            raise SessionError(f"unknown session: {session_id!r}")
        self._events[session_id].append({
            "text": text, "kind": kind,
            "ts": now if now is not None else time.time(),
        })
        self._save()

    def end_session(self, session_id, ttl_seconds=None, now=None):
        """Compress the session into Tier 2, then clear Tier 1.

        Returns the :class:`EpisodicRecord`, or ``None`` if the
        session produced nothing keepable (still clears the
        scratchpad; an empty session leaves no memory behind).
        """
        now = time.time() if now is None else now
        started = self.scratchpad.session_started_at(session_id)
        entries = self.scratchpad.end_session(session_id)
        events = self._events.pop(session_id, [])
        record = None
        try:
            record = compress_session(
                session_id, entries, events, started, now,
                ttl_seconds if ttl_seconds is not None
                else self.episodic_ttl,
                now=now)
            self.episodic.append(record)
        except MemoryError:
            record = None  # nothing keepable: no episodic record
        self._save()
        return record

    # -- Tier 1 write -------------------------------------------------

    def remember(self, session_id, key, value, kind="fact", now=None):
        """Write into the active session's scratchpad (Tier 1)."""
        entry = self.scratchpad.put(session_id, key, value, kind=kind,
                                    now=now)
        self._save()
        return entry

    # -- Tier 3 local seed --------------------------------------------

    def add_semantic_record(self, text, **kwargs):
        """Seed the local fallback corpus (Tier 3 fallback leg)."""
        rid = self.semantic.add_local_record(text, **kwargs)
        self._save()
        return rid

    # -- unified recall -----------------------------------------------

    def recall(self, query, session_id=None, top_k=5, now=None):
        """Recall across tiers, best-first within each tier, in order:
        scratchpad → episodic → semantic. Returns RecallResults."""
        results = []
        q = query.lower()

        # Tier 1: live scratchpad of the given session (or all sessions).
        sessions = ([session_id] if session_id
                    else self.scratchpad.session_ids())
        for sid in sessions:
            if not self.scratchpad.has_session(sid):
                continue
            for entry in self.scratchpad.search(sid, query):
                hay = (entry["key"] + " " + str(entry["value"])).lower()
                score = 1.0 if q in entry["key"].lower() else 0.9
                results.append(RecallResult(
                    tier="scratchpad", text=str(entry["value"]),
                    score=score, session_id=sid, key=entry["key"],
                    kind=entry["kind"]))

        # Tier 2: episodic summaries (skip expired; the sweeper owns
        # eviction, but recall must never surface a dead record).
        for rec in self.episodic:
            if rec.is_expired(now):
                continue
            if rec.matches(query):
                results.append(RecallResult(
                    tier="episodic",
                    text="; ".join(rec.key_points),
                    score=0.7, session_id=rec.session_id,
                    title=f"session {rec.session_id}",
                    extra={"key_points": rec.key_points,
                           "decisions": rec.decisions,
                           "facts": rec.facts,
                           "outcomes": rec.outcomes}))

        # Tier 3: semantic adapter (service, or local fallback).
        sem_results, source = self.semantic.recall(query, top_k=top_k)
        for r in sem_results:
            results.append(RecallResult(
                tier="semantic", text=r.get("text", ""),
                score=r.get("score") if isinstance(r.get("score"),
                                                   (int, float)) else 0.5,
                source=source, record_id=r.get("id"),
                title=r.get("title"),
                extra={"type": r.get("type"), "path": r.get("path"),
                       "date": r.get("date"),
                       "author": r.get("author")}))

        # Cap per tier, keep tier order: scratchpad, episodic, semantic.
        capped = []
        counts = {"scratchpad": 0, "episodic": 0, "semantic": 0}
        for r in results:
            if counts[r.tier] < top_k:
                capped.append(r)
                counts[r.tier] += 1
        return capped

    def recall_episodic(self, query=None, session_id=None,
                        since=None, until=None, now=None):
        """Search Tier 2 by keyword / session id / time window."""
        out = []
        for rec in self.episodic:
            if rec.is_expired(now):
                continue
            if session_id and rec.session_id != session_id:
                continue
            if since is not None and rec.created_at < since:
                continue
            if until is not None and rec.created_at > until:
                continue
            if query and not rec.matches(query):
                continue
            out.append(rec)
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out

    # -- TTL sweeper ----------------------------------------------------

    def sweep(self, now=None):
        """Enforce per-tier retention. Returns the eviction events.

        - Scratchpad: sessions idle longer than ``session_idle_ttl``
          are purged (their entries are gone. The session is over,
          nothing was compressed, so this is a loss, logged as such).
        - Episodic: expired records are evicted and logged.
        - Semantic: NEVER evicted by this engine.
        Every eviction is appended to ``eviction.jsonl``.
        """
        now = time.time() if now is None else now
        events = []
        paths = self._paths()

        # Tier 1: orphaned / idle scratchpads.
        for sid in list(self.scratchpad.session_ids()):
            started = self.scratchpad.session_started_at(sid) or now
            if now - started >= self.session_idle_ttl:
                n = len(self.scratchpad.list(sid))
                self.scratchpad.end_session(sid)
                self._events.pop(sid, None)
                events.append(self._log_eviction(
                    paths["eviction"], "scratchpad", sid, now,
                    f"session idle > {self.session_idle_ttl}s; "
                    f"{n} uncompressed entries discarded"))

        # Tier 2: expired episodic records.
        kept = []
        for rec in self.episodic:
            if rec.is_expired(now):
                events.append(self._log_eviction(
                    paths["eviction"], "episodic", rec.session_id, now,
                    f"episodic record expired (expires_at="
                    f"{rec.expires_at}); {len(rec.key_points)} key points "
                    f"evicted"))
            else:
                kept.append(rec)
        self.episodic = kept
        self._save()
        return events

    def _log_eviction(self, path, tier, record_id, now, reason):
        event = {
            "ts": now,
            "tier": tier,
            "record_id": record_id,
            "reason": reason,
        }
        _store.append_jsonl(path, event)
        return event

    def eviction_log(self):
        """All recorded eviction events, oldest first."""
        return _store.read_jsonl(self._paths()["eviction"])
