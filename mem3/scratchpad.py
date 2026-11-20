"""Tier 1: working scratchpad.

A short-lived, per-session scratch area for facts the agent is
actively using. Scoped to a session/run id: sessions are fully
isolated, and a scratchpad's lifetime is its session. Ending the
session clears it (its contents are handed to the episodic
compressor first; see :mod:`mem3.engine`).

Kinds: ``fact``, ``decision``, ``outcome``, ``note``, ``chatter``.
The kind is metadata for the compressor; the scratchpad itself
stores everything the agent puts in it.
"""

import time

from mem3.errors import SessionError

KINDS = ("fact", "decision", "outcome", "note", "chatter")


def _now():
    return time.time()


class Scratchpad:
    """In-memory per-session scratchpads, optionally persisted."""

    def __init__(self):
        # session_id -> {key -> entry dict}
        self._sessions = {}
        # session_id -> started_at (epoch seconds)
        self._started = {}

    # -- sessions ---------------------------------------------------

    def start_session(self, session_id, now=None):
        if session_id in self._sessions:
            raise SessionError(f"session already active: {session_id!r}")
        self._sessions[session_id] = {}
        self._started[session_id] = now if now is not None else _now()
        return session_id

    def has_session(self, session_id):
        return session_id in self._sessions

    def session_ids(self):
        return sorted(self._sessions)

    def session_started_at(self, session_id):
        return self._started.get(session_id)

    def end_session(self, session_id):
        """Clear the session's scratchpad, returning its entries.

        Entries are returned as a list of dicts (key, value, kind,
        created_at, updated_at), oldest first. The session is gone
        after this call.
        """
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        entries = sorted(
            self._sessions.pop(session_id).values(),
            key=lambda e: e["created_at"],
        )
        self._started.pop(session_id, None)
        return entries

    def clear_session(self, session_id):
        """Drop a session's entries without returning them."""
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        self._sessions[session_id] = {}

    # -- entries ----------------------------------------------------

    def put(self, session_id, key, value, kind="fact", now=None):
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        ts = now if now is not None else _now()
        entry = self._sessions[session_id].get(key)
        created = entry["created_at"] if entry else ts
        self._sessions[session_id][key] = {
            "key": key,
            "value": value,
            "kind": kind,
            "created_at": created,
            "updated_at": ts,
        }
        return self._sessions[session_id][key]

    def get(self, session_id, key, default=None):
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        entry = self._sessions[session_id].get(key)
        return entry["value"] if entry is not None else default

    def get_entry(self, session_id, key):
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        return self._sessions[session_id].get(key)

    def delete(self, session_id, key):
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        return self._sessions[session_id].pop(key, None) is not None

    def list(self, session_id, kind=None):
        """All entries of a session, oldest first; optionally by kind."""
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        entries = sorted(
            self._sessions[session_id].values(),
            key=lambda e: e["created_at"],
        )
        if kind is not None:
            entries = [e for e in entries if e["kind"] == kind]
        return entries

    def search(self, session_id, query):
        """Keyword search over a session's keys + values (case-insensitive)."""
        if session_id not in self._sessions:
            raise SessionError(f"unknown session: {session_id!r}")
        q = query.lower()
        return [
            e for e in self.list(session_id)
            if q in e["key"].lower() or q in str(e["value"]).lower()
        ]

    # -- persistence ------------------------------------------------

    def to_state(self):
        return {"sessions": self._sessions, "started": self._started}

    def from_state(self, state):
        self._sessions = dict(state.get("sessions", {}))
        self._started = dict(state.get("started", {}))
