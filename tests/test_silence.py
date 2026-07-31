from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, energy_based_vad, find_speech_start, trim
from audiosig import silence as silence_module
from audiosig.silence import (
    amplitude_to_db,
    median_filter_numpy,
    power_to_db,
    rms,
    spectral_flux,
    zero_crossing_rate,
)


def test_find_speech_start_evaluates_vad_once(monkeypatch: pytest.MonkeyPatch) -> None:
    source = np.zeros(5000, dtype=np.float32)
    source[1500:3500] = 0.5
    original_vad = silence_module.energy_based_vad
    call_count = 0

    def counting_vad(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_vad(*args, **kwargs)

    monkeypatch.setattr(silence_module, "energy_based_vad", counting_vad)

    find_speech_start(source, frame_length=256, hop_length=64)

    assert call_count == 1


def test_trim_and_vad_boundaries() -> None:
    source = np.zeros(5000, dtype=np.float32)
    source[1500:3500] = 0.5
    trimmed, interval = trim(source, frame_length=256, hop_length=64)
    assert 0 <= interval[0] <= 1500
    assert 3500 <= interval[1] <= 5000
    assert trimmed.size == interval[1] - interval[0]
    mask = energy_based_vad(source, frame_length=256, hop_length=64)
    assert mask.any()
    assert 0 <= find_speech_start(source, frame_length=256, hop_length=64) <= 1500


def test_silence_only_and_multichannel() -> None:
    source = np.zeros((2, 1000), dtype=np.float64)
    trimmed, interval = trim(source, frame_length=128, hop_length=32)
    assert trimmed.shape == source.shape
    np.testing.assert_array_equal(interval, [0, 1000])
    stereo = source.copy()
    stereo[1, 300:500] = 0.25
    assert energy_based_vad(stereo, frame_length=128, hop_length=32).shape[0] == 2


def test_silence_analysis_helpers_and_validation() -> None:
    source = np.sin(np.linspace(0, 4 * np.pi, 512)).astype(np.float32)
    assert np.isfinite(power_to_db(np.ones(3))).all()
    assert np.isfinite(amplitude_to_db(source)).all()
    assert rms(source).shape == ()
    assert zero_crossing_rate(source, frame_length=64, hop_length=32).size > 0
    assert spectral_flux(source, frame_length=64, hop_length=32).size > 0
    assert median_filter_numpy(np.arange(5), 3).shape == (5,)
    with pytest.raises(InvalidParameterError):
        median_filter_numpy(np.ones(3), 2)
    with pytest.raises(InvalidParameterError):
        energy_based_vad(source, threshold_db=-1)
    with pytest.raises(InvalidParameterError):
        trim(source, top_db=-1)
    numeric_trim, numeric_interval = trim(source, ref=1.0, frame_length=64, hop_length=32)
    assert numeric_trim.ndim == 1
    assert numeric_interval.shape == (2,)
