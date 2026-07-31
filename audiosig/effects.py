"""Speech-oriented time and pitch effects."""

from __future__ import annotations

import numpy as np

from ._resampling import resample_to_length
from ._spectral import istft, phase_vocoder, stft
from ._validation import (
    validate_audio,
    validate_filter,
    validate_finite,
    validate_integer,
    validate_positive,
)
from .exceptions import InvalidParameterError


def time_stretch(
    audio: np.ndarray,
    rate: float,
    *,
    axis: int = -1,
    n_fft: int = 2048,
    hop_length: int | None = None,
) -> np.ndarray:
    """Change duration while approximately preserving pitch.

    Rates above one are faster and shorter; rates below one are slower and
    longer. This phase-vocoder implementation is intended for speech and
    moderate prosody changes.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    stretch = validate_positive(rate, "rate")
    fft_size = validate_integer(n_fft, "n_fft", minimum=2)
    hop = validate_integer(hop_length if hop_length is not None else fft_size // 4, "hop_length")
    if hop > fft_size:
        raise InvalidParameterError("hop_length must not exceed n_fft")
    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    requested_length = input_length / stretch
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested stretched length is too large")
    target_length = max(1, round(requested_length))
    if stretch == 1.0:
        return np.array(source, dtype=source.dtype, copy=True)
    moved = np.moveaxis(source, normalized_axis, -1)
    padded_length = max(input_length, fft_size)
    if padded_length != input_length:
        moved = np.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(0, padded_length - input_length)])
    spectrum = stft(moved, n_fft=fft_size, hop_length=hop, center=True)
    transformed = phase_vocoder(spectrum, rate=stretch, hop_length=hop, n_fft=fft_size)
    result = istft(
        transformed,
        n_fft=fft_size,
        hop_length=hop,
        length=target_length,
        center=True,
        dtype=source.dtype,
    )
    return np.moveaxis(result, -1, normalized_axis).astype(source.dtype, copy=False)


def pitch_shift(
    audio: np.ndarray,
    *,
    sample_rate: int,
    semitones: float,
    bins_per_octave: int = 12,
    axis: int = -1,
    n_fft: int = 2048,
    hop_length: int | None = None,
    filter_width: int = 32,
) -> np.ndarray:
    """Shift pitch by semitones while retaining the exact input duration."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    validate_positive(sample_rate, "sample_rate")
    bins = validate_integer(bins_per_octave, "bins_per_octave")
    shift = validate_finite(semitones, "semitones")
    fft_size = validate_integer(n_fft, "n_fft", minimum=2)
    hop = validate_integer(hop_length if hop_length is not None else fft_size // 4, "hop_length")
    if hop > fft_size:
        raise InvalidParameterError("hop_length must not exceed n_fft")
    validate_filter(filter_width, 0.945)
    if source.shape[normalized_axis] == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    if shift == 0.0:
        return np.array(source, dtype=source.dtype, copy=True)
    octaves = shift / bins
    max_octaves = np.log2(np.finfo(np.float64).max)
    min_octaves = np.log2(np.nextafter(0.0, 1.0))
    if not min_octaves <= octaves <= max_octaves:
        raise InvalidParameterError("semitones produces an unrepresentable pitch ratio")
    ratio = float(np.exp2(octaves))
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise InvalidParameterError("semitones produces an invalid pitch ratio")
    rate = 1.0 / ratio
    if not np.isfinite(rate) or rate <= 0.0:
        raise InvalidParameterError("semitones produces an invalid pitch rate")
    stretched = time_stretch(
        source,
        rate=rate,
        axis=normalized_axis,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return resample_to_length(
        stretched,
        source.shape[normalized_axis],
        axis=normalized_axis,
        filter_width=filter_width,
    )
