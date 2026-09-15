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
from .basic import downmix_to_mono, generate_silence
from .boundaries import find_smooth_cut_point
from .effects import PitchShiftMethod, TimeStretchMethod, pitch_shift, time_stretch
from .exceptions import AudioShapeError, AudioSignalError, InvalidParameterError
from .loudness import (
    LoudnessMetrics,
    integrated_loudness,
    measure_loudness,
    sample_peak_dbfs,
    true_peak_dbtp,
)
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
from .speech import SpeechEffectsMethod, apply_speech_effects

__all__ = [
    "AudioShapeError",
    "AudioSignalError",
    "InvalidParameterError",
    "LoudnessMetrics",
    "PitchShiftMethod",
    "SpeechEffectsMethod",
    "TimeStretchMethod",
    "__version__",
    "abs2",
    "activity_to_intervals",
    "amplitude_to_db",
    "apply_gain_db",
    "apply_speech_effects",
    "downmix_to_mono",
    "energy_based_vad",
    "find_smooth_cut_point",
    "find_speech_bounds",
    "find_speech_start",
    "frame_rms",
    "frame_signal",
    "frames_to_samples",
    "generate_silence",
    "integrated_loudness",
    "measure_loudness",
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
    "sample_peak_dbfs",
    "short_time_energy",
    "spectral_flux",
    "split",
    "time_stretch",
    "trim",
    "true_peak_dbtp",
    "zero_crossing_rate",
]
