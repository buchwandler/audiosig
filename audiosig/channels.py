"""Channel-layout and channel-reduction helpers."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_axis
from .exceptions import AudioShapeError


def downmix_to_mono(
    audio: np.ndarray,
    *,
    channel_axis: int = 0,
) -> np.ndarray:
    """Downmix channels by arithmetic mean without modifying ``audio``.

    One-dimensional input is treated as already mono and copied. For higher
    dimensional input, ``channel_axis`` identifies the channel dimension and
    is removed by a dtype-preserving arithmetic mean. No clipping or amplitude
    normalization is performed.
    """
    validated, _ = validate_audio(audio, allow_empty=True)
    if validated.ndim == 1:
        return np.array(validated, dtype=validated.dtype, copy=True)

    normalized_axis = validate_axis(channel_axis, validated.ndim)
    if validated.shape[normalized_axis] == 0:
        raise AudioShapeError("the channel axis cannot be empty")
    result_shape = validated.shape[:normalized_axis] + validated.shape[normalized_axis + 1 :]
    if 0 in result_shape:
        return np.empty(result_shape, dtype=validated.dtype)
    mixed = np.mean(validated, axis=normalized_axis, dtype=validated.dtype)
    return np.array(mixed, dtype=validated.dtype, copy=True)
