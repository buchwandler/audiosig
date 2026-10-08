from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import audiosig.spectral as public_spectral
from audiosig import AudioShapeError, InvalidParameterError
from audiosig._spectral import istft as private_istft
from audiosig._spectral import stft as private_stft


def _tone(sample_rate: int = 16_000, frequency: float = 1_000.0) -> np.ndarray:
    time = np.arange(sample_rate // 8, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time)


@pytest.mark.parametrize("n_fft,hop_length", [(16, 4), (15, 3)])
@pytest.mark.parametrize("center", [True, False])
def test_public_stft_is_private_wrapper(n_fft: int, hop_length: int, center: bool) -> None:
    source = np.random.default_rng(40).normal(size=(2, 111)).astype(np.float32)
    actual = public_spectral.stft(source, n_fft=n_fft, hop_length=hop_length, center=center)
    expected = private_stft(source, n_fft=n_fft, hop_length=hop_length, center=center)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape[-2] == n_fft // 2 + 1


@pytest.mark.parametrize("n_fft,hop_length", [(32, 8), (31, 7)])
def test_public_istft_round_trip(n_fft: int, hop_length: int) -> None:
    source = np.random.default_rng(n_fft).normal(size=257).astype(np.float64)
    spectrum = public_spectral.stft(source, n_fft=n_fft, hop_length=hop_length)
    actual = public_spectral.istft(spectrum, n_fft=n_fft, hop_length=hop_length, length=source.size)
    expected = private_istft(spectrum, n_fft=n_fft, hop_length=hop_length, length=source.size)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, source, atol=1e-8)


def test_public_stft_short_input_and_power_semantics() -> None:
    silence = np.zeros(1, dtype=np.float32)
    spectrum = public_spectral.stft(silence, n_fft=16, hop_length=4)
    assert spectrum.shape == (9, 2)
    power = public_spectral.power_spectrogram(silence, n_fft=16, hop_length=4)
    assert power.dtype == np.float64
    np.testing.assert_array_equal(power, np.zeros_like(power))

    source = _tone()
    actual = public_spectral.power_spectrogram(source, n_fft=64, hop_length=16, power=1.5)
    expected = np.abs(public_spectral.stft(source, n_fft=64, hop_length=16)) ** 1.5
    np.testing.assert_allclose(actual, expected)


def test_mel_filterbank_shape_bounds_and_slaney_points() -> None:
    bank = public_spectral.mel_filterbank(sample_rate=16_000, n_fft=31, n_mels=8)
    nyquist_bank = public_spectral.mel_filterbank(
        sample_rate=16_000, n_fft=31, n_mels=8, f_max=8_000
    )
    assert bank.shape == (8, 16)
    np.testing.assert_array_equal(bank, nyquist_bank)
    assert bank.dtype == np.float64
    assert np.all(bank >= 0.0)
    assert public_spectral._hz_to_mel_slaney(np.array([0.0, 1_000.0]))[0] == 0.0
    assert public_spectral._hz_to_mel_slaney(np.array([1_000.0]))[0] == pytest.approx(15.0)
    np.testing.assert_allclose(
        public_spectral._mel_to_hz_slaney(np.array([0.0, 15.0])), [0.0, 1_000.0]
    )


def test_mel_filterbank_validates_limits_and_scale() -> None:
    with pytest.raises(InvalidParameterError):
        public_spectral.mel_filterbank(sample_rate=16_000, n_fft=64, n_mels=4, f_min=-1)
    with pytest.raises(InvalidParameterError):
        public_spectral.mel_filterbank(
            sample_rate=16_000, n_fft=64, n_mels=4, f_min=1_000, f_max=1_000
        )
    with pytest.raises(InvalidParameterError):
        public_spectral.mel_filterbank(sample_rate=16_000, n_fft=64, n_mels=4, f_max=8_001)
    with pytest.raises(InvalidParameterError):
        public_spectral.mel_filterbank(
            sample_rate=16_000, n_fft=64, n_mels=4, mel_scale=cast(Any, "htk")
        )


def test_mel_spectrogram_preserves_lanes_and_log_floor() -> None:
    silence = np.zeros((2, 1, 80), dtype=np.float32)
    mel = public_spectral.mel_spectrogram(
        silence, sample_rate=16_000, n_fft=32, hop_length=8, n_mels=5
    )
    assert mel.shape == (2, 1, 5, 11)
    assert mel.dtype == np.float64
    np.testing.assert_array_equal(mel, 0.0)
    floor = 1e-7
    log_mel = public_spectral.mel_spectrogram(
        silence,
        sample_rate=16_000,
        n_fft=32,
        hop_length=8,
        n_mels=5,
        log=True,
        log_floor=floor,
    )
    np.testing.assert_array_equal(log_mel, np.log(floor))


def test_mel_tone_and_mfcc_dct_contract() -> None:
    sample_rate = 16_000
    source = _tone(sample_rate, 1_000.0)
    bank = public_spectral.mel_filterbank(sample_rate=sample_rate, n_fft=512, n_mels=20)
    mel = public_spectral.mel_spectrogram(
        source, sample_rate=sample_rate, n_fft=512, hop_length=128, n_mels=20
    )
    peak_bin = np.argmax(np.mean(mel, axis=-1))
    mel_peak_frequency = np.fft.rfftfreq(512, 1.0 / sample_rate)[np.argmax(bank[peak_bin])]
    assert mel_peak_frequency == pytest.approx(1_000.0, abs=400.0)

    dct = public_spectral._dct_ii_ortho_matrix(20)
    np.testing.assert_allclose(dct @ dct.T, np.eye(20), atol=1e-14)
    coefficients = public_spectral.mfcc(
        source,
        sample_rate=sample_rate,
        n_fft=512,
        hop_length=128,
        n_mfcc=8,
        n_mels=20,
    )
    assert coefficients.shape == (8, mel.shape[-1])
    assert coefficients.dtype == np.float64
    with pytest.raises(InvalidParameterError):
        public_spectral.mfcc(source, sample_rate=sample_rate, n_mfcc=21, n_mels=20)


def test_frame_times_formulas_and_zero_frames() -> None:
    np.testing.assert_array_equal(
        public_spectral.frame_times(3, sample_rate=10, hop_length=2), [0.0, 0.2, 0.4]
    )
    np.testing.assert_array_equal(
        public_spectral.frame_times(3, sample_rate=10, hop_length=2, frame_length=4, center=False),
        [0.2, 0.4, 0.6],
    )
    assert public_spectral.frame_times(0, sample_rate=10, hop_length=2).shape == (0,)
    with pytest.raises(InvalidParameterError):
        public_spectral.frame_times(1, sample_rate=10, hop_length=2, center=False)
    with pytest.raises(InvalidParameterError):
        public_spectral.frame_times(-1, sample_rate=10, hop_length=2)


def test_match_frames_is_strict_and_never_truncates() -> None:
    reference = np.zeros((2, 4, 3))
    estimate = reference.copy()
    actual_reference, actual_estimate = public_spectral.match_frames(reference, estimate)
    assert actual_reference is reference
    assert actual_estimate is estimate
    with pytest.raises(AudioShapeError):
        public_spectral.match_frames(reference, np.zeros((2, 4, 2)))
    with pytest.raises(AudioShapeError):
        public_spectral.match_frames(reference, np.zeros((3, 4, 3)))
    with pytest.raises(AudioShapeError):
        public_spectral.match_frames(reference, np.zeros(3))


def test_public_spectral_rejects_bad_audio_and_parameters() -> None:
    with pytest.raises(AudioShapeError):
        public_spectral.stft(np.ones(16, dtype=np.int32), n_fft=8, hop_length=2)
    with pytest.raises(AudioShapeError):
        public_spectral.stft(np.array([np.nan]), n_fft=8, hop_length=2)
    with pytest.raises(InvalidParameterError):
        public_spectral.power_spectrogram(np.ones(16), n_fft=8, hop_length=2, power=0.0)
    with pytest.raises(InvalidParameterError):
        public_spectral.istft(np.zeros((5, 2)), n_fft=8, hop_length=2, dtype=np.dtype(np.int64))
