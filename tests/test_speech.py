from __future__ import annotations

import numpy as np
import pytest

import audiosig.speech as speech_module
from audiosig import AudioSignalError, InvalidParameterError, apply_speech_effects


def tone(length: int = 2400, dtype: type = np.float32) -> np.ndarray:
    return np.sin(2 * np.pi * 180 * np.arange(length) / 24_000).astype(dtype)


def test_speech_effects_neutral_copy_and_empty_contract() -> None:
    source = tone()
    result = apply_speech_effects(source, sample_rate=24_000)
    assert result is not source
    assert not np.shares_memory(result, source)
    np.testing.assert_array_equal(result, source)

    empty = np.empty((2, 0), dtype=np.float64)
    empty_result = apply_speech_effects(
        empty,
        sample_rate=24_000,
        rate=0.75,
        semitones=2.0,
        gain_db=-6.0,
        axis=1,
    )
    assert empty_result.shape == empty.shape
    assert empty_result.dtype == empty.dtype
    assert not np.shares_memory(empty_result, empty)


def test_speech_effects_apply_all_effects_and_exact_length() -> None:
    source = tone()
    original = source.copy()
    result = apply_speech_effects(
        source,
        sample_rate=24_000,
        rate=1.25,
        semitones=2.0,
        gain_db=6.0,
        n_fft=256,
        hop_length=64,
    )
    assert result.shape == (round(source.size / 1.25),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    assert not np.array_equal(result, source[: result.size])
    np.testing.assert_array_equal(source, original)


def test_speech_effects_operation_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_pitch(audio: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append("pitch")
        return audio

    def fake_rate(audio: np.ndarray, rate: float, **kwargs: object) -> np.ndarray:
        calls.append("rate")
        return audio

    def fake_gain(audio: np.ndarray, db: float, **kwargs: object) -> np.ndarray:
        calls.append("gain")
        return audio

    monkeypatch.setattr(speech_module, "pitch_shift", fake_pitch)
    monkeypatch.setattr(speech_module, "time_stretch", fake_rate)
    monkeypatch.setattr(speech_module, "apply_gain_db", fake_gain)

    apply_speech_effects(
        tone(128),
        sample_rate=24_000,
        rate=1.1,
        semitones=2.0,
        gain_db=3.0,
        n_fft=32,
        hop_length=8,
    )
    assert calls == ["pitch", "rate", "gain"]


def test_speech_effects_clipping_and_validation() -> None:
    source = np.ones(16, dtype=np.float32)
    clipped = apply_speech_effects(source, sample_rate=24_000, gain_db=20.0, clip=True)
    np.testing.assert_array_equal(clipped, 1.0)
    muted = apply_speech_effects(source, sample_rate=24_000, gain_db=-np.inf)
    np.testing.assert_array_equal(muted, 0.0)
    with pytest.raises(AudioSignalError):
        apply_speech_effects(source, sample_rate=0)
    with pytest.raises(InvalidParameterError):
        apply_speech_effects(source, sample_rate=24_000, rate=0)
    with pytest.raises(InvalidParameterError):
        apply_speech_effects(source, sample_rate=24_000, semitones=np.inf)
    with pytest.raises(InvalidParameterError):
        apply_speech_effects(source, sample_rate=24_000, hop_length=33, n_fft=32)
