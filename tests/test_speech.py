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
        method="phase_vocoder",
        n_fft=256,
        hop_length=64,
    )
    assert result.shape == (round(source.size / 1.25),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    assert not np.array_equal(result, source[: result.size])
    np.testing.assert_array_equal(source, original)


def test_speech_effects_combined_plan_uses_one_tsm_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_stretch(audio: np.ndarray, rate: float, **kwargs: object) -> np.ndarray:
        calls.append(f"stretch:{rate:.6f}")
        return audio

    def fake_gain(audio: np.ndarray, db: float, **kwargs: object) -> np.ndarray:
        calls.append("gain")
        return audio

    monkeypatch.setattr(speech_module, "time_stretch", fake_stretch)
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
    assert calls == ["stretch:0.979989", "gain"]


def test_speech_effects_matching_rate_and_pitch_uses_only_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_stretch(audio: np.ndarray, rate: float, **kwargs: object) -> np.ndarray:
        calls.append("stretch")
        return audio

    def fake_resample(audio: np.ndarray, length: int, **kwargs: object) -> np.ndarray:
        calls.append(f"resample:{length}")
        return np.zeros(length, dtype=audio.dtype)

    monkeypatch.setattr(speech_module, "time_stretch", fake_stretch)
    monkeypatch.setattr(speech_module, "resample_to_length", fake_resample)
    apply_speech_effects(
        tone(128), sample_rate=24_000, rate=2.0, semitones=12.0, method="phase_vocoder"
    )
    assert calls == ["resample:64"]


def test_speech_effects_passes_rolloff_to_pitch_resampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_resample(audio: np.ndarray, length: int, **kwargs: object) -> np.ndarray:
        captured.update(kwargs)
        return np.zeros(length, dtype=audio.dtype)

    monkeypatch.setattr(speech_module, "resample_to_length", fake_resample)
    apply_speech_effects(
        tone(128), sample_rate=24_000, semitones=2.0, method="phase_vocoder", rolloff=0.8
    )
    assert captured["rolloff"] == 0.8


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


def test_speech_effects_supports_esola_and_keeps_one_tsm_pass() -> None:
    source = tone(4800, dtype=np.float64)
    result = apply_speech_effects(
        source,
        sample_rate=24_000,
        rate=1.25,
        semitones=2.0,
        method="esola",
    )
    assert result.shape == (round(source.size / 1.25),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()

    with pytest.raises(InvalidParameterError, match="interval"):
        apply_speech_effects(source, sample_rate=24_000, rate=2.0, semitones=-12.0, method="esola")
