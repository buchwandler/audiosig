"""NumPy stride-based signal framing."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_integer


def frame(
    audio: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
) -> np.ndarray:
    """Return a read-only view shaped ``(..., frames, frame_length)``.

    The sample axis is moved to the end before framing. Overlapping frames are
    intentionally read-only so callers cannot accidentally corrupt the source
    array through a strided view.
    """
    source, normalized_axis = validate_audio(audio, axis=axis)
    length = validate_integer(frame_length, "frame_length")
    hop = validate_integer(hop_length, "hop_length")
    moved = np.moveaxis(source, normalized_axis, -1)
    sample_count = moved.shape[-1]
    frame_count = 1 + (sample_count - length) // hop if sample_count >= length else 0
    shape = (*moved.shape[:-1], frame_count, length)
    if frame_count == 0:
        return np.empty(shape, dtype=source.dtype)
    strides = (*moved.strides[:-1], hop * moved.strides[-1], moved.strides[-1])
    result = np.lib.stride_tricks.as_strided(moved, shape=shape, strides=strides)
    result.setflags(write=False)
    return result
