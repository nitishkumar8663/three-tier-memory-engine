# 3-Tier Memory Engine

Agents that actually remember: a working scratchpad for the session
in progress, compressed episodic summaries of finished sessions, and
semantic recall, with automated TTL eviction across the tiers.
Stdlib-only Python, no models, no cloud.

## The three tiers

| Tier | What it is | Lifetime |
|------|-----------|----------|
| 1: Working scratchpad | Per-session key/value facts the agent is actively using (`put`/`get`/`clear`, scoped to a session/run id) | Dies with the session |
| 2: Episodic store | Deterministic, extractive compression of a finished session (scratchpad + session events) into a compact summary record | TTL-evicted (default 30 days, configurable per record) |
| 3: Semantic recall | Thin adapter over the Spark's existing common memory service (PARA + bge-m3) | **Never evicted by this engine**. The records belong to the service |

## Relation to the existing memory service

The Spark already has a working common memory service: a thin
wrapper over PARA + bge-m3 semantic recall, HTTP on
`127.0.0.1:18792` (`services/memory-service/memory-service.py`,
`scripts/memory-store.py`). **That part is not rebuilt here.** This
project adds only the missing pieces:

- **Tier 1 scratchpad**: new; the service has no session-scoped
  working area.
- **Tier 2 episodic compression**: new; the service stores records
  but never compresses a session into a summary.
- **TTL eviction + sweeper**: new; the service has no retention
  policy.
- **Tier 3 adapter**: a thin client for the service's real
  `POST /recall` interface (verified against the service source:
  `{"text", "top_k"}` → `{"results": [{id, type, title, text, path,
  date, author, source, salience, score, relevance, recency,
  importance}]}`), with a local keyword fallback when the service is
  unreachable. The adapter is optional: the engine works standalone
  without the service running.

The engine is a standalone library. It is **not** wired into the
live memory service (that is a separate decision for later) and it
uses its own data directory, never the live memory store.

## How episodic compression works (no LLM)

Deterministic, extractive rules:

- **Dedupe**: same normalised text kept once (first occurrence wins).
- **Keep**: kinds `decision`, `fact`, `outcome`.
- **Keep (conditional)**: `note` entries only if fact-like (a
  decision/fact marker word, a number, or a mid-sentence capitalised
  token); bare small-talk notes are dropped.
- **Drop**: kind `chatter` and filler one-liners (`ok`, `thanks`,
  `lol`, …).
- **Order**: decisions first, then facts, then outcomes, each in
  first-seen order; `key_points` is the merged list.

A session that produces nothing keepable leaves no episodic record
at all (the scratchpad is still cleared).

## Unified API

```python
from mem3 import MemoryEngine

eng = MemoryEngine("data/")                    # own data dir
eng.start_session("sprint-planning")
eng.remember("sprint-planning", "launch-date",
             "Launch is 2026-12-01", kind="fact")
eng.note("sprint-planning", "Ravi owns the migration", kind="note")
record = eng.end_session("sprint-planning")    # -> episodic summary
print(record.key_points)

hits = eng.recall("launch")                    # scratchpad -> episodic -> semantic
for h in hits:
    print(h.tier, h.score, h.text)

eng.add_semantic_record("PARA notes live in the memory service",
                        title="memory service")  # Tier 3 fallback corpus
events = eng.sweep()                           # enforce TTLs, log evictions
print(eng.eviction_log())
```

`recall()` returns hits in tier order (live scratchpad first, then
episodic summaries, then semantic), each naming its tier and score;
semantic hits also name their leg (`"service"` vs
`"local-fallback"`). `recall_episodic(query, session_id, since,
until)` searches Tier 2 by keyword / session / time window.

## Retention (TTL) policy

- **Scratchpad**: expires with the session. `end_session` clears
  it; the sweeper also purges scratchpads idle longer than
  `session_idle_ttl` (default 24 h). Those entries were never
  compressed, so the purge is logged as a loss.
- **Episodic**: each record carries `expires_at` (default 30 days,
  per-record override at `end_session`). The sweeper evicts expired
  records; every eviction is appended to `eviction.jsonl`.
- **Semantic**: never evicted by this engine.

## Persistence

Per-engine data directory (default layout under the dir you pass in):

```
data/
    scratchpad.json   # live sessions (atomic JSON writes)
    episodic.json     # summary records
    semantic.json     # local fallback corpus
    events.json       # session events awaiting compression
    eviction.jsonl    # append-only eviction log
```

JSON writes are atomic (temp file + fsync + `os.replace` + directory
fsync, as in durable-agent-state-machine). A SIGKILL leaves the old or the new file,
never a torn one. Torn JSONL lines are skipped on read.

## Run the demo

```bash
python demo/memory_demo.py
```

Runs end to end and writes the transcript to `demo/evidence.txt`:
scratchpad isolation, session compression (chatter dropped,
decisions/facts kept), keyword recall of the summary, a short-TTL
record evicted by the sweeper while a live one survives, and the
semantic adapter answering via the live service or, when the
service is down, via the local fallback.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

Covers: scratchpad scoping/isolation/clearing, compression
keep/drop/dedupe rules, TTL eviction (expired gone, live kept,
eviction logged, semantic never evicted), the semantic adapter
(fallback with the service down; the real `/recall` shape against a
fake in-process service), unified recall ordering, episodic search
filters, and persistence round-trips.

## Layout

```
mem3/
    __init__.py     # public API
    errors.py       # MemoryError, SessionError
    store.py        # atomic JSON + JSONL persistence
    scratchpad.py   # Tier 1: session-scoped working area
    episodic.py     # Tier 2: deterministic session compression
    semantic.py     # Tier 3: service adapter + local fallback
    engine.py       # unified remember/recall, sessions, sweeper
demo/
    memory_demo.py  # live end-to-end proof
    evidence.txt    # transcript of the last verified demo run
tests/
    test_scratchpad.py
    test_episodic.py
    test_ttl.py
    test_semantic.py
    test_recall.py
```

## Limits (v1)

- Episodic compression is extractive and rule-based: it keeps what
  was explicitly marked fact/decision/outcome (or looks fact-like);
  it cannot infer what an unmarked utterance *meant*, and a session
  of pure chatter leaves no record.
- The semantic fallback is keyword overlap over a small local
  corpus: a degraded leg, not a replacement for bge-m3 recall.
- The adapter assumes the service's documented `POST /recall`
  shape (verified against the service source, not against a live
  service. The service was down when this was built; the fake-service
  test covers the shape). If the service's response shape changes,
  the adapter's field mapping needs updating.
- Single process, no locking: one engine instance per data directory
  at a time. No cross-session dedupe in Tier 2 (each session
  compresses independently).
- Scratchpad values must be JSON-serialisable (enforced by the
  atomic store at save time).
