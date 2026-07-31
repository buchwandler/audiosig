"""Portable, dependency-light audio signal processing for NumPy arrays."""

from __future__ import annotations

try:
    from ._version import __version__  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only used in an unbuilt source checkout
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("audiosig")
    except PackageNotFoundError:
        __version__ = "0.1.dev0"

from ._resampling import resample, resample_speed, resample_to_length
from .amplitude import apply_gain_db, peak_normalize
from .effects import pitch_shift, time_stretch
from .exceptions import AudioShapeError, AudioSignalError, InvalidParameterError
from .speech import apply_speech_effects
from .silence import (
    abs2,
    activity_to_intervals,
    amplitude_to_db,
    energy_based_vad,
    find_speech_bounds,
    find_speech_start,
    frame_rms,
    frame_signal,
    frames_to_samples,
    median_filter_numpy,
    minmax_normalize,
    non_silent_frames,
    normalized_energy_vad,
    power_to_db,
    relative_db_vad,
    rms,
    short_time_energy,
    spectral_flux,
    split,
    trim,
    zero_crossing_rate,
)

__all__ = [
    "AudioShapeError",
    "AudioSignalError",
    "InvalidParameterError",
    "__version__",
    "abs2",
    "apply_speech_effects",
    "activity_to_intervals",
    "amplitude_to_db",
    "apply_gain_db",
    "energy_based_vad",
    "find_speech_bounds",
    "find_speech_start",
    "frame_rms",
    "frame_signal",
    "frames_to_samples",
    "median_filter_numpy",
    "minmax_normalize",
    "non_silent_frames",
    "normalized_energy_vad",
    "peak_normalize",
    "pitch_shift",
    "power_to_db",
    "relative_db_vad",
    "resample",
    "resample_speed",
    "resample_to_length",
    "rms",
    "short_time_energy",
    "spectral_flux",
    "split",
    "time_stretch",
    "trim",
    "zero_crossing_rate",
]
