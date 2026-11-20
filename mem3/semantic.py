"""Tier 3: semantic recall adapter.

This tier does NOT rebuild semantic recall. The Spark already has it:
the common memory service (PARA + bge-m3 embeddings, SQLite + FTS5)
behind an HTTP API on 127.0.0.1:18792. The adapter calls that service's
real interface, read from the service source
(``services/memory-service/memory-service.py``):

    POST /recall  {"text": ..., "top_k": ...}
      -> {"results": [{"id", "type", "title", "text", "path", "date",
                       "author", "source", "salience", "score",
                       "relevance", "recency", "importance"}]}

    GET /health -> {"ok": true}

The adapter is optional: the engine works standalone without the
service running. When the service is unreachable (connection error,
timeout, non-200, malformed body) the adapter falls back to a local
keyword search over records the engine itself stored
(:meth:`SemanticAdapter.add_local_record`), scored by token overlap.
Fallback results are marked ``"source": "local-fallback"`` so callers
can always tell which leg answered.
"""

import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:18792"
DEFAULT_TIMEOUT = 5.0

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "which", "how", "why", "when", "who",
    "did", "do", "does", "i", "you", "we", "it", "its", "this", "that",
    "with", "by", "at", "from", "as", "be", "been", "have", "has", "had",
}


def _tokens(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if t not in _STOPWORDS
    ]


class SemanticAdapter:
    """Thin client for the common memory service + local fallback."""

    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT,
                 local_records=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Local fallback corpus: list of dicts {id, title, text, ...}.
        self.local_records = list(local_records or [])
        self._next_local_id = (
            max((r.get("id", 0) for r in self.local_records), default=0) + 1
        )
        self.last_source = None  # "service" | "local-fallback" | None

    # -- service leg ------------------------------------------------

    def is_service_up(self):
        try:
            with urllib.request.urlopen(
                self.base_url + "/health", timeout=self.timeout
            ) as resp:
                return resp.status == 200 and json.loads(resp.read()).get("ok")
        except Exception:
            return False

    def _service_recall(self, query, top_k):
        req = urllib.request.Request(
            self.base_url + "/recall",
            data=json.dumps({"text": query, "top_k": top_k}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(
                    resp.geturl(), resp.status, "non-200", None, None)
            body = json.loads(resp.read())
        results = body.get("results")
        if not isinstance(results, list):
            raise ValueError("service /recall returned no results list")
        out = []
        for r in results:
            out.append({
                "id": r.get("id"),
                "type": r.get("type"),
                "title": r.get("title"),
                "text": r.get("text", ""),
                "path": r.get("path"),
                "date": r.get("date"),
                "author": r.get("author"),
                "source": "service",
                "score": r.get("score"),
            })
        return out

    # -- local fallback leg -----------------------------------------

    def add_local_record(self, text, title=None, record_type="note",
                         author="mem3", source=None, salience=5):
        """Seed the fallback corpus (persisted by the engine)."""
        rec = {
            "id": self._next_local_id,
            "type": record_type,
            "title": title or (str(text)[:60].strip() or "note"),
            "text": text,
            "author": author,
            "source": source,
            "salience": salience,
            "created_at": time.time(),
        }
        self._next_local_id += 1
        self.local_records.append(rec)
        return rec["id"]

    def _fallback_recall(self, query, top_k):
        qtok = set(_tokens(query))
        if not qtok:
            return []
        scored = []
        for rec in self.local_records:
            hay = set(_tokens(rec.get("title", ""))) | set(
                _tokens(rec.get("text", "")))
            overlap = len(qtok & hay)
            if overlap:
                scored.append((overlap / len(qtok), rec))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{
            "id": rec["id"],
            "type": rec.get("type"),
            "title": rec.get("title"),
            "text": rec.get("text", ""),
            "path": None,
            "date": None,
            "author": rec.get("author"),
            "source": "local-fallback",
            "score": round(score, 3),
        } for score, rec in scored[:top_k]]

    # -- public ------------------------------------------------------

    def recall(self, query, top_k=5):
        """Recall via the service; fall back to local keyword search.

        Returns ``(results, source)`` where source is ``"service"`` or
        ``"local-fallback"``. Never raises for service outages. The
        fallback is the point.
        """
        try:
            results = self._service_recall(query, top_k)
            self.last_source = "service"
            return results, "service"
        except Exception:
            self.last_source = "local-fallback"
            return self._fallback_recall(query, top_k), "local-fallback"

    def to_state(self):
        return {"local_records": self.local_records,
                "next_local_id": self._next_local_id}

    def from_state(self, state):
        self.local_records = list(state.get("local_records", []))
        self._next_local_id = state.get(
            "next_local_id",
            max((r.get("id", 0) for r in self.local_records), default=0) + 1,
        )
