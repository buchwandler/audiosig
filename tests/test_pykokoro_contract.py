"""Numeric contracts consumed by PyKokoro without importing PyKokoro."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import audiosig
from audiosig import (
    AudioShapeError,
    AudioSignalError,
    InvalidParameterError,
    activity_to_intervals,
    apply_gain_db,
    downmix_to_mono,
    energy_based_vad,
    find_smooth_cut_point,
    frame_rms,
    generate_silence,
    pitch_shift,
    resample,
    resample_speed,
    trim,
)

pytestmark = pytest.mark.downstream


def test_contract_does_not_import_forbidden_dsp_packages() -> None:
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
import audiosig
forbidden = {'librosa', 'scipy', 'sklearn', 'audiomentations',
             'torch', 'numba', 'signalsmith_stretch'}
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parents[1])],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_smooth_cut_contract_for_continuous_audio() -> None:
    sample = np.arange(1_200, dtype=np.float32)
    audio = np.sin(2.0 * np.pi * 180.0 * sample / 24_000.0)
    first = find_smooth_cut_point(audio, start=120, end=1_080, anchor=640, window_length=120)
    second = find_smooth_cut_point(audio, start=120, end=1_080, anchor=640, window_length=120)

    assert isinstance(first, int)
    assert 120 <= first < 1_080
    assert first == second


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trim_contract(dtype: type[np.floating]) -> None:
    empty = np.empty(0, dtype=dtype)
    trimmed, interval = trim(empty)
    assert trimmed.shape == (0,)
    np.testing.assert_array_equal(interval, [0, 0])

    source = np.zeros(800, dtype=dtype)
    source[200:600] = 0.5
    trimmed, interval = trim(source, frame_length=32, hop_length=8)
    np.testing.assert_array_equal(trimmed, source[interval[0] : interval[1]])
    assert 0 <= interval[0] <= 200 <= 600 <= interval[1] <= source.size

    silence, silence_interval = trim(np.zeros(100, dtype=dtype), ref=1.0)
    assert silence.size == 0
    np.testing.assert_array_equal(silence_interval, [0, 0])


@pytest.mark.parametrize("length", [0, 1, 119, 120, 121, 241])
def test_normalized_vad_contract_for_short_inputs(length: int) -> None:
    source = np.zeros(length, dtype=np.float32)
    if length:
        source[-min(20, length) :] = 1.0
    activity = energy_based_vad(
        source,
        24_000,
        frame_duration_ms=5.0,
        energy_threshold=0.02,
        pad_end=True,
    )
    assert activity.dtype == np.bool_
    assert activity.ndim == 1
    if length == 0:
        assert activity.size == 0


def test_energy_valley_feature_matches_manual_padded_rms() -> None:
    source = np.array([0.0, 1.0, 0.0, 0.0, 0.5], dtype=np.float64)
    values = frame_rms(
        source,
        frame_length=4,
        hop_length=4,
        center=False,
        pad_end=True,
        normalize=True,
        dtype=np.float64,
    )
    padded = np.pad(source, (0, 3))
    frames = padded.reshape(2, 4)
    raw = np.sqrt(np.mean(frames**2, axis=-1))
    expected = (raw - raw.min()) / (raw.max() - raw.min())
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rate", [0.5, 0.75, 1.0, 1.25, 1.5])
def test_prosody_rate_contract(rate: float) -> None:
    source = np.sin(2 * np.pi * 220 * np.arange(2400) / 24_000).astype(np.float32)
    original = source.copy()
    result = resample_speed(source, rate)
    assert result.shape == (max(1, round(source.size / rate)),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("semitones", [-4.0, -2.0, 0.0, 2.0, 4.0])
def test_prosody_pitch_and_gain_contract(semitones: float) -> None:
    source = np.sin(2 * np.pi * 180 * np.arange(2400) / 24_000).astype(np.float32)
    shifted = pitch_shift(source, sample_rate=24_000, semitones=semitones, n_fft=256, hop_length=64)
    gained = apply_gain_db(source, semitones * 3.0)
    assert shifted.shape == source.shape
    assert shifted.dtype == gained.dtype == source.dtype
    assert np.isfinite(shifted).all() and np.isfinite(gained).all()
    assert not np.shares_memory(shifted, source)


def test_audio_annotation_numeric_sequence_has_expected_length() -> None:
    source = np.arange(100, dtype=np.float32)
    clipped = source[10:50]
    sped = resample_speed(clipped, 2.0)
    repeated = np.tile(sped, 2)
    gained = apply_gain_db(repeated, 6.0)
    output = resample(gained, source_rate=100, target_rate=200)

    assert output.dtype == np.float32
    assert output.shape == (80,)
    assert np.isfinite(output).all()


def test_ttsforge_audio_primitives_contract() -> None:
    silence = generate_silence(0.5, 24_000)
    assert silence.shape == (12_000,)
    assert silence.dtype == np.float32

    stereo = np.column_stack([np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32)])
    mono = downmix_to_mono(stereo, channel_axis=1)
    np.testing.assert_allclose(mono, 0.5)


def test_quiet_intervals_and_typed_exception_contract() -> None:
    speech = np.array([False, True, True, False, False, True], dtype=bool)
    quiet = activity_to_intervals(~speech, hop_length=10, sample_count=60, min_frames=2)
    np.testing.assert_array_equal(quiet, [[30, 50]])

    invalid_calls = (
        lambda: resample_speed(np.ones(4, dtype=np.float32), 0.0),
        lambda: activity_to_intervals(speech, hop_length=10, sample_count=60, min_frames=0),
        lambda: trim(np.ones(4, dtype=np.int16)),
    )
    for call in invalid_calls:
        with pytest.raises(AudioSignalError):
            call()
    assert issubclass(AudioShapeError, AudioSignalError)
    assert issubclass(InvalidParameterError, AudioSignalError)
    assert callable(audiosig.resample_to_length)
