"""Persistence for the memory engine: atomic JSON writes + JSONL logs.

Mirrors the durable-agent-state-machine checkpoint-store discipline: write to a temp file
in the same directory, fsync, ``os.replace`` onto the target, fsync the
directory. A SIGKILL at any moment leaves either the old or the new
file intact, never a torn one. The eviction log is append-only JSONL,
fsynced per event.
"""

import json
import os


def atomic_write_json(path, obj):
    """Write *obj* as JSON to *path* atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(directory)


def read_json(path, default):
    """Read a JSON file, returning *default* if it is missing/corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def append_jsonl(path, obj):
    """Append one JSON object as a line, fsynced."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Read a JSONL file, skipping torn/blank trailing lines."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # torn line: skip, keep the rest
    except OSError:
        pass
    return rows


def _fsync_dir(directory):
    try:
        fd = os.open(directory, os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
