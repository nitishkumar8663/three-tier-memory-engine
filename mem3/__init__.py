"""3-Tier Memory Engine.

A stdlib-only 3-tier memory engine for agents:

- Tier 1, working scratchpad: short-lived, per-session facts the
  agent is actively using. Dies with the session.
- Tier 2, episodic store: deterministic, extractive compression of a
  finished session (scratchpad + session events) into a compact
  summary record. TTL-evicted, searchable by session id / time /
  keywords. No LLM required.
- Tier 3, semantic recall: a thin adapter over the Spark's existing
  common memory service (PARA + bge-m3, POST /recall on
  127.0.0.1:18792). The semantic store is NOT rebuilt here; when the
  service is unreachable the adapter falls back to keyword search
  over a small local record set, so the engine works standalone.

Unified API lives in :mod:`mem3.engine` (``MemoryEngine`` with
``remember`` / ``recall``); the tiers are usable independently.
"""

from mem3.engine import MemoryEngine, RecallResult
from mem3.episodic import compress_session, EpisodicRecord
from mem3.errors import MemoryError, SessionError
from mem3.scratchpad import Scratchpad
from mem3.semantic import SemanticAdapter

__all__ = [
    "MemoryEngine",
    "RecallResult",
    "Scratchpad",
    "EpisodicRecord",
    "compress_session",
    "SemanticAdapter",
    "MemoryError",
    "SessionError",
]

__version__ = "0.1.0"
