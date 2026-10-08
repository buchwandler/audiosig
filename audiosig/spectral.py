"""Public NumPy spectral and cepstral analysis features."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from . import _spectral
from ._validation import (
    validate_boolean,
    validate_finite,
    validate_integer,
    validate_positive,
)
from .exceptions import AudioShapeError, InvalidParameterError

_DEFAULT_DTYPE = np.dtype(np.float64)


def stft(
    audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    center: bool = True,
) -> np.ndarray:
    """Return the private Hann-windowed STFT as ``(..., bins, frames)``."""
    validate_boolean(center, "center")
    return _spectral.stft(audio, n_fft=n_fft, hop_length=hop_length, center=center)


def istft(
    spectrum: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    length: int | None = None,
    center: bool = True,
    dtype: np.dtype = _DEFAULT_DTYPE,
) -> np.ndarray:
    """Invert a public STFT using the existing overlap-add implementation."""
    validate_boolean(center, "center")
    try:
        output_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise InvalidParameterError("dtype must be float32 or float64") from error
    if output_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise InvalidParameterError("dtype must be float32 or float64")
    return _spectral.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        length=length,
        center=center,
        dtype=output_dtype,
    )


def power_spectrogram(
    audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    center: bool = True,
    power: float = 2.0,
) -> np.ndarray:
    """Compute ``abs(STFT) ** power`` in float64, without normalization or dB conversion."""
    exponent = validate_positive(power, "power")
    spectrum = stft(audio, n_fft=n_fft, hop_length=hop_length, center=center)
    return np.power(np.abs(spectrum).astype(np.float64, copy=False), exponent)


def _hz_to_mel_slaney(frequencies_hz: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = frequencies / f_sp
    logarithmic = frequencies >= min_log_hz
    mel[logarithmic] = min_log_mel + np.log(frequencies[logarithmic] / min_log_hz) / logstep
    return mel


def _mel_to_hz_slaney(mel_values: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel_values, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    frequencies = mel * f_sp
    logarithmic = mel >= min_log_mel
    frequencies[logarithmic] = min_log_hz * np.exp(logstep * (mel[logarithmic] - min_log_mel))
    return frequencies


def _frequency_limits(
    *, sample_rate: int, f_min: float, f_max: float | None
) -> tuple[int, float, float]:
    rate = validate_integer(sample_rate, "sample_rate")
    minimum = validate_finite(f_min, "f_min")
    if minimum < 0.0:
        raise InvalidParameterError("f_min must be nonnegative")
    nyquist = rate / 2.0
    maximum = nyquist if f_max is None else validate_finite(f_max, "f_max")
    if maximum <= minimum or maximum > nyquist:
        raise InvalidParameterError("frequency limits must satisfy 0 <= f_min < f_max <= Nyquist")
    return rate, minimum, maximum


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    mel_scale: Literal["slaney"] = "slaney",
    norm: Literal["slaney"] | None = "slaney",
) -> np.ndarray:
    """Build Slaney mel triangles against exact real-FFT bin-center frequencies."""
    _, minimum, maximum = _frequency_limits(sample_rate=sample_rate, f_min=f_min, f_max=f_max)
    fft_size = validate_integer(n_fft, "n_fft", minimum=2)
    mel_count = validate_integer(n_mels, "n_mels")
    if mel_scale != "slaney":
        raise InvalidParameterError("mel_scale must be 'slaney'")
    if norm not in ("slaney", None):
        raise InvalidParameterError("norm must be 'slaney' or None")

    mel_edges = np.linspace(
        _hz_to_mel_slaney(np.array([minimum]))[0],
        _hz_to_mel_slaney(np.array([maximum]))[0],
        mel_count + 2,
        dtype=np.float64,
    )
    hz_edges = _mel_to_hz_slaney(mel_edges)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    lower = hz_edges[:-2, np.newaxis]
    center = hz_edges[1:-1, np.newaxis]
    upper = hz_edges[2:, np.newaxis]
    weights = np.maximum(
        0.0,
        np.minimum(
            (frequencies[np.newaxis, :] - lower) / (center - lower),
            (upper - frequencies[np.newaxis, :]) / (upper - center),
        ),
    )
    if norm == "slaney":
        weights *= (2.0 / (upper[:, 0] - lower[:, 0]))[:, np.newaxis]
    return cast(np.ndarray, np.maximum(weights, 0.0).astype(np.float64, copy=False))


def mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    center: bool = True,
    power: float = 2.0,
    f_min: float = 0.0,
    f_max: float | None = None,
    mel_scale: Literal["slaney"] = "slaney",
    norm: Literal["slaney"] | None = "slaney",
    log: bool = False,
    log_floor: float = 1e-10,
) -> np.ndarray:
    """Return linear mel power, or its natural logarithm when ``log=True``."""
    validate_boolean(log, "log")
    floor = validate_positive(log_floor, "log_floor")
    power_spec = power_spectrogram(
        audio, n_fft=n_fft, hop_length=hop_length, center=center, power=power
    )
    bank = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        mel_scale=mel_scale,
        norm=norm,
    )
    mel_power: np.ndarray = np.einsum("mf,...ft->...mt", bank, power_spec, optimize=True)
    if log:
        log_mel: np.ndarray = np.log(np.maximum(mel_power, floor))
        return log_mel
    return mel_power


def _dct_ii_ortho_matrix(n_mels: int) -> np.ndarray:
    indices = np.arange(n_mels, dtype=np.float64)
    coefficients = indices[:, np.newaxis]
    matrix = np.cos(np.pi / n_mels * (indices[np.newaxis, :] + 0.5) * coefficients)
    matrix[0] *= np.sqrt(1.0 / n_mels)
    if n_mels > 1:
        matrix[1:] *= np.sqrt(2.0 / n_mels)
    return matrix


def mfcc(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mfcc: int = 13,
    n_mels: int = 40,
    center: bool = True,
    f_min: float = 0.0,
    f_max: float | None = None,
    log_floor: float = 1e-10,
) -> np.ndarray:
    """Compute natural-log mel-power MFCCs with an orthonormal DCT-II."""
    coefficient_count = validate_integer(n_mfcc, "n_mfcc")
    mel_count = validate_integer(n_mels, "n_mels")
    if coefficient_count > mel_count:
        raise InvalidParameterError("n_mfcc must not exceed n_mels")
    log_mel = mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_count,
        center=center,
        f_min=f_min,
        f_max=f_max,
        log=True,
        log_floor=log_floor,
    )
    matrix = _dct_ii_ortho_matrix(mel_count)[:coefficient_count]
    return cast(np.ndarray, np.einsum("km,...mt->...kt", matrix, log_mel, optimize=True))


def frame_times(
    n_frames: int,
    *,
    sample_rate: int,
    hop_length: int,
    frame_length: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """Return frame-center times in seconds for centered or uncentered framing."""
    count = validate_integer(n_frames, "n_frames", minimum=0)
    rate = validate_integer(sample_rate, "sample_rate")
    hop = validate_integer(hop_length, "hop_length")
    validate_boolean(center, "center")
    if frame_length is None:
        if not center:
            raise InvalidParameterError("frame_length is required when center=False")
        length = None
    else:
        length = validate_integer(frame_length, "frame_length")
    offsets = np.arange(count, dtype=np.float64) * hop
    if not center:
        assert length is not None
        offsets += length / 2.0
    return offsets / rate


def match_frames(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    frame_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact frame correspondence; never crop or align arrays."""
    if not isinstance(reference, np.ndarray) or not isinstance(estimate, np.ndarray):
        raise AudioShapeError("reference and estimate must be NumPy arrays")
    if reference.ndim == 0 or estimate.ndim == 0:
        raise AudioShapeError("frame arrays must have at least one dimension")
    if reference.ndim != estimate.ndim:
        raise AudioShapeError("reference and estimate must have equal dimensionality")
    axis = frame_axis
    if not isinstance(axis, (int, np.integer)):
        raise AudioShapeError("frame_axis must be an integer")
    axis = int(axis)
    if axis < 0:
        axis += reference.ndim
    if not 0 <= axis < reference.ndim:
        raise AudioShapeError(f"frame_axis {frame_axis} is invalid for {reference.ndim}-D arrays")
    if reference.shape != estimate.shape:
        for index, (ref_size, est_size) in enumerate(
            zip(reference.shape, estimate.shape, strict=True)
        ):
            if index != axis and ref_size != est_size:
                raise AudioShapeError("reference and estimate must match on non-frame axes")
        raise AudioShapeError("reference and estimate must have identical frame counts")
    return reference, estimate
