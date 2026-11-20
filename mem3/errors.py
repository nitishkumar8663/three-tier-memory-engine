"""Errors for the 3-tier memory engine."""


class MemoryError(Exception):
    """Base error for the memory engine."""


class SessionError(MemoryError):
    """Raised for invalid session operations (unknown session, etc.)."""
