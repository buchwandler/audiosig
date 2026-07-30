"""Exception types raised by AudioSig."""

from __future__ import annotations


class AudioSignalError(Exception):
    """Base exception for AudioSig operations."""


class InvalidParameterError(AudioSignalError, ValueError):
    """A parameter is outside the supported domain."""


class AudioShapeError(AudioSignalError, ValueError):
    """The audio array shape or dtype is unsupported."""
