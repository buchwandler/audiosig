"""Small NumPy-only STFT and phase-vocoder primitives."""

from __future__ import annotations

import numpy as np

from ._framing import frame
from ._validation import validate_audio, validate_choices, validate_integer, validate_positive
from .exceptions import AudioShapeError, InvalidParameterError

_DEFAULT_DTYPE = np.dtype(np.float64)


def _fft_settings(n_fft: int, hop_length: int) -> tuple[int, int]:
    fft_size = validate_integer(n_fft, "n_fft", minimum=2)
    hop = validate_integer(hop_length, "hop_length")
    if hop > fft_size:
        raise InvalidParameterError("hop_length must not exceed n_fft")
    return fft_size, hop


def _pad_for_stft(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    *,
    center: bool,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Pad short arrays safely and add centered analysis padding."""
    source = audio
    if center:
        padding = n_fft // 2
        mode = pad_mode if source.shape[-1] > 1 or pad_mode == "constant" else "constant"
        source = np.pad(
            source,
            [(0, 0)] * (source.ndim - 1) + [(padding, padding)],
            mode=mode,
        )
    if source.shape[-1] < n_fft:
        source = np.pad(source, [(0, 0)] * (source.ndim - 1) + [(0, n_fft - source.shape[-1])])
    remainder = (source.shape[-1] - n_fft) % hop_length
    if remainder:
        source = np.pad(source, [(0, 0)] * (source.ndim - 1) + [(0, hop_length - remainder)])
    return source


def stft(
    audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    center: bool = True,
    win_length: int | None = None,
    pad_mode: str = "reflect",
    window: str = "symmetric",
    frame_count: int | None = None,
) -> np.ndarray:
    """Compute a centered Hann-windowed real-input STFT.

    The returned shape is ``(..., frequency_bins, frames)``. ``window`` may be
    ``'symmetric'`` (the historical NumPy Hann) or ``'periodic'`` (the
    ``0.5 - 0.5*cos(2*pi*i/win_length)`` convention used by some frontends).
    When ``win_length < n_fft``, the window is centered in the FFT frame.
    """
    source, axis = validate_audio(audio)
    if axis != source.ndim - 1:
        source = np.moveaxis(source, axis, -1)
    fft_size, hop = _fft_settings(n_fft, hop_length)
    pad = validate_choices(pad_mode, ("reflect", "constant"), "pad_mode")
    window_mode = validate_choices(window, ("symmetric", "periodic"), "window")
    window_size = fft_size if win_length is None else validate_integer(win_length, "win_length")
    if window_size > fft_size:
        raise InvalidParameterError("win_length must not exceed n_fft")
    requested_frames = (
        None if frame_count is None else validate_integer(frame_count, "frame_count", minimum=0)
    )
    padded = _pad_for_stft(source, fft_size, hop, center=center, pad_mode=pad)
    frames = frame(padded, frame_length=fft_size, hop_length=hop)
    sample_indices = np.arange(window_size, dtype=np.float64)
    if window_mode == "periodic":
        analysis_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * sample_indices / window_size)
    else:
        analysis_window = np.hanning(window_size)
    left = (fft_size - window_size) // 2
    right = fft_size - window_size - left
    analysis_window = np.pad(analysis_window, (left, right)).astype(source.dtype, copy=False)
    spectrum = np.fft.rfft(frames * analysis_window, n=fft_size, axis=-1)
    spectrum = np.moveaxis(spectrum, -1, -2)
    if requested_frames is not None:
        if spectrum.shape[-1] > requested_frames:
            spectrum = spectrum[..., :requested_frames]
        elif spectrum.shape[-1] < requested_frames:
            padding = requested_frames - spectrum.shape[-1]
            spectrum = np.pad(spectrum, [(0, 0)] * (spectrum.ndim - 1) + [(0, padding)])
    return spectrum


def istft(
    spectrum: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    length: int | None = None,
    center: bool = True,
    dtype: np.dtype = _DEFAULT_DTYPE,
) -> np.ndarray:
    """Invert a spectrum with overlap-add and squared-window normalization."""
    fft_size, hop = _fft_settings(n_fft, hop_length)
    if not isinstance(spectrum, np.ndarray) or spectrum.ndim < 2:
        raise AudioShapeError("spectrum must be an array shaped (..., frequencies, frames)")
    expected_bins = fft_size // 2 + 1
    if spectrum.shape[-2] != expected_bins or spectrum.shape[-1] == 0:
        raise AudioShapeError("spectrum has incompatible frequency or frame dimensions")
    target_length = validate_integer(length, "length") if length is not None else None
    frames = np.fft.irfft(np.moveaxis(spectrum, -2, -1), n=fft_size, axis=-1)
    window = np.hanning(fft_size).astype(dtype, copy=False)
    frame_count = frames.shape[-2]
    base_length = (frame_count - 1) * hop + fft_size
    output = np.zeros((*frames.shape[:-2], base_length), dtype=dtype)
    envelope = np.zeros(base_length, dtype=dtype)
    windowed = frames * window
    for index in range(frame_count):
        start = index * hop
        output[..., start : start + fft_size] += windowed[..., index, :]
        envelope[start : start + fft_size] += window * window
    safe_envelope = np.where(envelope > np.finfo(dtype).eps, envelope, 1.0)
    output /= safe_envelope
    if center:
        padding = fft_size // 2
        if output.shape[-1] > 2 * padding:
            output = output[..., padding:-padding]
        else:
            output = output[..., 0:0]
    if target_length is not None:
        if output.shape[-1] < target_length:
            output = np.pad(
                output,
                [(0, 0)] * (output.ndim - 1) + [(0, target_length - output.shape[-1])],
            )
        output = output[..., :target_length]
    return output.astype(dtype, copy=False)


def phase_vocoder(
    spectrum: np.ndarray,
    *,
    rate: float,
    hop_length: int,
    n_fft: int,
) -> np.ndarray:
    """Time-scale an STFT by interpolating magnitudes and accumulating phase."""
    stretch = validate_positive(rate, "rate")
    fft_size, hop = _fft_settings(n_fft, hop_length)
    if not isinstance(spectrum, np.ndarray) or spectrum.ndim < 2:
        raise AudioShapeError("spectrum must be an array shaped (..., frequencies, frames)")
    expected_bins = fft_size // 2 + 1
    if spectrum.shape[-2] != expected_bins or spectrum.shape[-1] == 0:
        raise AudioShapeError("spectrum has incompatible frequency or frame dimensions")
    frame_count = spectrum.shape[-1]
    time_steps = np.arange(0, frame_count, stretch, dtype=np.float64)
    output = np.empty((*spectrum.shape[:-1], time_steps.size), dtype=spectrum.dtype)
    phase_advance = 2.0 * np.pi * hop * np.arange(expected_bins, dtype=np.float64) / fft_size
    phase = np.angle(spectrum[..., 0])
    for output_index, time in enumerate(time_steps):
        left = min(int(np.floor(time)), frame_count - 1)
        right = min(left + 1, frame_count - 1)
        fraction = time - left
        left_spectrum = spectrum[..., left]
        right_spectrum = spectrum[..., right]
        magnitude = (1.0 - fraction) * np.abs(left_spectrum) + fraction * np.abs(right_spectrum)
        output[..., output_index] = magnitude * np.exp(1j * phase)
        if right != left:
            delta = np.angle(right_spectrum) - np.angle(left_spectrum) - phase_advance
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            phase += phase_advance + delta
    return output
