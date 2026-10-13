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

from ._resampling import resample, resample_speed, resample_to_length, resample_windowed_sinc
from .amplitude import (
    apply_gain_db,
    fade_edges,
    gain_for_level,
    join_with_edge_fades,
    peak_normalize,
    soft_limit,
)
from .basic import downmix_to_mono, generate_silence
from .boundaries import (
    PauseCropConfig,
    crop_on_pause,
    energy_fraction,
    energy_onset,
    find_smooth_cut_point,
    speech_onset,
    trim_lead,
    trim_trail,
)
from .effects import PitchShiftMethod, TimeStretchMethod, pitch_shift, time_stretch
from .exceptions import AudioShapeError, AudioSignalError, InvalidParameterError
from .loudness import (
    LoudnessAnalysis,
    LoudnessMetrics,
    analyze_loudness,
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
from .spectral import (
    frame_times,
    istft,
    match_frames,
    mel_filterbank,
    mel_spectrogram,
    mfcc,
    power_spectrogram,
    stft,
)
from .speech import (
    SpeechEffectsEnvelopeMethod,
    SpeechEffectsMethod,
    apply_speech_effects,
    apply_speech_effects_envelope,
    speech_effects_output_frames,
)
from .speech_level import (
    ReferenceLevelResult,
    SpeechLevel,
    normalize_reference_level,
    speech_level,
)

__all__ = [
    "AudioShapeError",
    "AudioSignalError",
    "InvalidParameterError",
    "LoudnessAnalysis",
    "LoudnessMetrics",
    "PauseCropConfig",
    "PitchShiftMethod",
    "ReferenceLevelResult",
    "SpeechEffectsEnvelopeMethod",
    "SpeechEffectsMethod",
    "SpeechLevel",
    "TimeStretchMethod",
    "__version__",
    "abs2",
    "activity_to_intervals",
    "amplitude_to_db",
    "analyze_loudness",
    "apply_gain_db",
    "apply_speech_effects",
    "apply_speech_effects_envelope",
    "crop_on_pause",
    "downmix_to_mono",
    "energy_based_vad",
    "energy_fraction",
    "energy_onset",
    "fade_edges",
    "find_smooth_cut_point",
    "find_speech_bounds",
    "find_speech_start",
    "frame_rms",
    "frame_signal",
    "frame_times",
    "frames_to_samples",
    "gain_for_level",
    "generate_silence",
    "integrated_loudness",
    "istft",
    "join_with_edge_fades",
    "match_frames",
    "measure_loudness",
    "median_filter_numpy",
    "mel_filterbank",
    "mel_spectrogram",
    "mfcc",
    "minmax_normalize",
    "non_silent_frames",
    "normalize_reference_level",
    "normalized_energy_vad",
    "peak_normalize",
    "pitch_shift",
    "power_spectrogram",
    "power_to_db",
    "relative_db_vad",
    "resample",
    "resample_speed",
    "resample_to_length",
    "resample_windowed_sinc",
    "rms",
    "sample_peak_dbfs",
    "short_time_energy",
    "soft_limit",
    "spectral_flux",
    "speech_effects_output_frames",
    "speech_level",
    "speech_onset",
    "split",
    "stft",
    "time_stretch",
    "trim",
    "trim_lead",
    "trim_trail",
    "true_peak_dbtp",
    "zero_crossing_rate",
]
