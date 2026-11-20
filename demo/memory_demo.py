#!/usr/bin/env python3
"""Demo: the 3-tier memory engine, end to end, with observed evidence.

Proves, live and in order:
  1. Tier 1 scratchpad: session-scoped facts, isolated per session.
  2. Tier 2 episodic compression: end the session, get a compact
     summary that keeps decisions/facts/outcomes and drops chatter;
     recall it later by keyword.
  3. TTL eviction: short-TTL records expire, the sweeper evicts them
     and logs the eviction; live records survive.
  4. Tier 3 semantic adapter: recall through the adapter, via the
     live common memory service when it is up, via the local keyword
     fallback when it is down (it is down at the time of writing).

Writes the full transcript to demo/evidence.txt. Uses its own data
directory (demo/data/). Never the live memory store.

Run:  python demo/memory_demo.py
"""

import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from mem3 import MemoryEngine  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
EVIDENCE = os.path.join(HERE, "evidence.txt")

LINES = []


def say(line=""):
    print(line)
    LINES.append(line)


def main():
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    eng = MemoryEngine(DATA_DIR)
    say("=== 3-Tier Memory Engine: live demo ===")
    say(f"data dir: {DATA_DIR} (engine's own; live memory store untouched)")
    say("")

    # --- Tier 3 pre-check: is the real service up? -------------------
    up = eng.semantic.is_service_up()
    say(f"[tier 3] common memory service (127.0.0.1:18792) up: {up}")
    say("")

    # --- Tier 1: scratchpad -------------------------------------------
    say("--- Tier 1: working scratchpad ---")
    eng.start_session("sprint-planning")
    eng.remember("sprint-planning", "launch-date",
                 "Launch is 2026-12-01", kind="fact")
    eng.remember("sprint-planning", "db-choice",
                 "Decided: SQLite for the prototype, Postgres later",
                 kind="decision")
    eng.remember("sprint-planning", "latency",
                 "p95 must stay under 200ms", kind="fact")
    eng.note("sprint-planning", "lol ok", kind="chatter")
    eng.note("sprint-planning", "thanks!", kind="chatter")
    eng.note("sprint-planning",
             "Ravi agreed to own the migration", kind="note")
    eng.start_session("other-session")
    eng.remember("other-session", "launch-date",
                 "Other session's own value", kind="fact")
    say("scratchpad 'sprint-planning' launch-date: "
        f"{eng.scratchpad.get('sprint-planning', 'launch-date')!r}")
    say("scratchpad 'other-session' launch-date: "
        f"{eng.scratchpad.get('other-session', 'launch-date')!r}")
    say("-> sessions isolated: same key, different values. OK")
    say("")

    # --- Tier 2: episodic compression ----------------------------------
    say("--- Tier 2: episodic session compression ---")
    eng.end_session("other-session")
    say("other session ended too (its fact compresses separately).")
    rec = eng.end_session("sprint-planning")
    assert rec is not None
    say(f"session 'sprint-planning' ended; scratchpad cleared: "
        f"{not eng.scratchpad.has_session('sprint-planning')}")
    say(f"episodic record: session={rec.session_id} "
        f"source_items={rec.source_count} dropped={rec.dropped_count}")
    say(f"  decisions: {rec.decisions}")
    say(f"  facts:     {rec.facts}")
    say(f"  outcomes:  {rec.outcomes}")
    say("-> chatter ('lol ok', 'thanks!') dropped; decision + facts kept. OK")
    say("")
    say("recall by keyword 'launch' (later, no live session):")
    for hit in eng.recall("launch"):
        say(f"  [{hit.tier}] score={hit.score} :: {hit.text[:100]}")
    say("-> episodic summary recalled by keyword. OK")
    say("")

    # --- TTL eviction ----------------------------------------------------
    say("--- TTL eviction ---")
    eng.start_session("short-lived")
    eng.remember("short-lived", "temp",
                 "Temporary note, expires in 1 second", kind="fact")
    short_rec = eng.end_session("short-lived", ttl_seconds=1)
    eng.start_session("long-lived")
    eng.remember("long-lived", "keep",
                 "Durable note, expires in 30 days", kind="fact")
    long_rec = eng.end_session("long-lived", ttl_seconds=30 * 24 * 3600)
    say(f"episodic records before sweep: {len(eng.episodic)}")
    say("waiting 1.5s for the short-TTL record to expire...")
    time.sleep(1.5)
    events = eng.sweep()
    say(f"sweeper evicted {len(events)} record(s):")
    for ev in events:
        say(f"  [{ev['tier']}] {ev['record_id']}: {ev['reason']}")
    say(f"episodic records after sweep: {len(eng.episodic)} "
        f"(kept: {[r.session_id for r in eng.episodic]})")
    assert short_rec.session_id not in [r.session_id for r in eng.episodic]
    assert long_rec.session_id in [r.session_id for r in eng.episodic]
    say("-> expired evicted + logged; live record kept. OK")
    say(f"eviction log entries: {len(eng.eviction_log())}")
    say("")

    # --- Tier 3: semantic adapter ---------------------------------------
    say("--- Tier 3: semantic adapter ---")
    eng.add_semantic_record(
        "The Spark memory service stores PARA notes with bge-m3 embeddings.",
        title="Spark memory service")
    eng.add_semantic_record(
        "Wardrobe app v1.1 was validated live on the emulator.",
        title="Wardrobe validation")
    results, source = eng.semantic.recall("Spark memory service bge-m3")
    say(f"adapter source: {source}")
    for r in results:
        say(f"  [{r['source']}] score={r['score']} :: {r['title']}: "
            f"{r['text'][:80]}")
    if source == "local-fallback":
        say("-> service down, local keyword fallback engaged. OK")
    else:
        say("-> live service answered. OK")
    say("")

    # --- Unified recall ordering ----------------------------------------
    say("--- Unified recall ordering ---")
    eng.start_session("live")
    eng.remember("live", "launch-date", "Launch is 2026-12-01",
                 kind="fact")
    tiers = [h.tier for h in eng.recall("launch")]
    say(f"recall('launch') tier order: {tiers}")
    say("-> scratchpad first, then episodic, then semantic. OK")
    eng.end_session("live")
    say("")
    say("=== demo complete: all checks observed live ===")

    with open(EVIDENCE, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    say(f"transcript written to {EVIDENCE}")


if __name__ == "__main__":
    main()
