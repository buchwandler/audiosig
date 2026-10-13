from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from audiosig import InvalidParameterError
from audiosig._resampling import resample_windowed_sinc
from audiosig.spectral import mel_filterbank, mel_spectrogram, stft
from audiosig.speech_level import normalize_reference_level


def _htk_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _htk_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def test_periodic_hann_and_independent_win_length_match_direct_formula() -> None:
    source = np.random.default_rng(11).normal(size=40).astype(np.float64)
    n_fft, win_length, hop_length = 16, 8, 4
    actual = stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
        win_length=win_length,
        window="periodic",
        pad_mode="constant",
        frame_count=3,
    )

    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)
    left = (n_fft - win_length) // 2
    window = np.pad(window, (left, n_fft - win_length - left))
    expected = np.stack(
        [
            np.fft.rfft(source[index * hop_length : index * hop_length + n_fft] * window)
            for index in range(3)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.shape == (n_fft // 2 + 1, 3)


def test_stft_padding_mode_and_explicit_frame_count() -> None:
    source = np.array([0.2, 0.4, -0.7, 0.1], dtype=np.float64)
    reflected = stft(source, n_fft=8, hop_length=4, pad_mode="reflect", frame_count=1)
    constant = stft(source, n_fft=8, hop_length=4, pad_mode="constant", frame_count=1)
    assert not np.allclose(reflected, constant)

    longer = stft(np.arange(20, dtype=np.float64), n_fft=8, hop_length=2, frame_count=14)
    assert longer.shape[-1] == 14
    np.testing.assert_array_equal(longer[..., -3:], 0.0)
    shorter = stft(np.arange(20, dtype=np.float64), n_fft=8, hop_length=2, frame_count=2)
    assert shorter.shape[-1] == 2


def test_htk_filterbank_matches_direct_triangle_formula_and_norm_option() -> None:
    sample_rate, n_fft, n_mels = 24_000, 32, 6
    bank = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=20.0,
        f_max=7_600.0,
        mel_scale="htk",
        norm=None,
    )
    edges = _htk_hz(
        np.linspace(
            _htk_mel(np.array([20.0]))[0],
            _htk_mel(np.array([7_600.0]))[0],
            n_mels + 2,
        )
    )
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    expected = np.maximum(
        0.0,
        np.minimum(
            (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
            (edges[2:, None] - frequencies[None, :]) / (edges[2:, None] - edges[1:-1, None]),
        ),
    )
    np.testing.assert_allclose(bank, expected, rtol=1e-13, atol=1e-13)

    normalized = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=20.0,
        f_max=7_600.0,
        mel_scale="htk",
        norm="slaney",
    )
    np.testing.assert_allclose(normalized, bank * (2.0 / (edges[2:] - edges[:-2]))[:, None])
    np.testing.assert_allclose(
        _htk_hz(_htk_mel(np.array([0.0, 700.0, 1_000.0]))),
        [0.0, 700.0, 1_000.0],
    )


def test_sopro_semantic_speaker_and_acoustic_mel_configurations_are_supported() -> None:
    rng = np.random.default_rng(7)
    semantic = rng.normal(size=16_000).astype(np.float32)
    semantic_mel = mel_spectrogram(
        semantic,
        sample_rate=16_000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        mel_scale="slaney",
        norm="slaney",
        power=2.0,
        window="periodic",
    )
    assert semantic_mel.shape[0] == 80
    assert np.isfinite(semantic_mel).all()

    speaker_mel = mel_spectrogram(
        semantic,
        sample_rate=16_000,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        n_mels=80,
        f_min=20.0,
        f_max=7_600.0,
        mel_scale="slaney",
        norm="slaney",
        power=2.0,
        window="periodic",
    )
    assert speaker_mel.shape[0] == 80
    assert np.isfinite(speaker_mel).all()

    acoustic = rng.normal(size=24_000).astype(np.float32)
    acoustic_mel = mel_spectrogram(
        acoustic,
        sample_rate=24_000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=100,
        mel_scale="htk",
        norm=None,
        power=1.0,
        window="periodic",
    )
    assert acoustic_mel.shape[0] == 100
    assert np.isfinite(acoustic_mel).all()


def test_mel_frame_count_and_invalid_spectral_options() -> None:
    source = np.ones(200, dtype=np.float32)
    mel = mel_spectrogram(
        source,
        sample_rate=16_000,
        n_fft=32,
        win_length=16,
        hop_length=8,
        n_mels=4,
        frame_count=5,
    )
    assert mel.shape == (4, 5)
    with pytest.raises(InvalidParameterError, match="win_length"):
        stft(source, n_fft=16, win_length=17, hop_length=4)
    with pytest.raises(InvalidParameterError, match="pad_mode"):
        stft(source, n_fft=16, hop_length=4, pad_mode=cast(Any, "edge"))
    with pytest.raises(InvalidParameterError, match="window"):
        stft(source, n_fft=16, hop_length=4, window=cast(Any, "hamming"))
    with pytest.raises(InvalidParameterError, match="frame_count"):
        stft(source, n_fft=16, hop_length=4, frame_count=-1)
    with pytest.raises(InvalidParameterError, match="mel_scale"):
        mel_filterbank(sample_rate=16_000, n_fft=64, n_mels=8, mel_scale=cast(Any, "slaney_linear"))


def test_pinned_sopro_semantic_speaker_acoustic_mel_golden() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sopro_reference_golden.json"
    golden = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = golden["tones"]["steady"]
    config = golden["frontend_config"]
    sample_rate = expected["sample_rate"]
    indices = np.arange(expected["samples"], dtype=np.float64)
    source = (
        0.08 * np.sin(2.0 * np.pi * 220.0 * indices / sample_rate)
        + 0.025 * np.sin(2.0 * np.pi * 791.0 * indices / sample_rate)
    ).astype(np.float32)
    prepared = normalize_reference_level(source, sample_rate=sample_rate).audio
    rate16 = resample_windowed_sinc(
        prepared,
        source_rate=sample_rate,
        target_rate=16_000,
        radius=16,
        alignment="half_sample",
        window="raised_cosine",
        antialias=False,
        normalize_weights=True,
        boundary="zero",
        length_mode="round",
    )

    semantic_cfg = config["semantic_encoder"]
    semantic_frames = int(np.ceil(rate16.size / semantic_cfg["hop_length"]))
    semantic_audio = np.pad(rate16, (0, semantic_cfg["n_fft"]))
    semantic = mel_spectrogram(
        semantic_audio,
        sample_rate=semantic_cfg["sample_rate"],
        n_fft=semantic_cfg["n_fft"],
        win_length=semantic_cfg["n_fft"],
        hop_length=semantic_cfg["hop_length"],
        n_mels=semantic_cfg["n_mels"],
        power=2.0,
        mel_scale="slaney",
        norm="slaney",
        window="periodic",
        frame_count=semantic_frames + 2,
    )
    semantic_log = np.log10(np.maximum(1.0e-10, semantic)).astype(np.float32)
    maximum = float(np.max(semantic_log))
    semantic = ((np.maximum(semantic_log, maximum - 8.0) + 4.0) / 4.0).astype(np.float32)

    speaker_cfg = config["speaker_encoder"]
    speaker = mel_spectrogram(
        rate16,
        sample_rate=speaker_cfg["sample_rate"],
        n_fft=speaker_cfg["n_fft"],
        win_length=speaker_cfg["win_length"],
        hop_length=speaker_cfg["hop_length"],
        n_mels=speaker_cfg["n_mels"],
        f_min=speaker_cfg["f_min"],
        f_max=speaker_cfg["f_max"],
        power=2.0,
        mel_scale="slaney",
        norm="slaney",
        window="periodic",
    )
    speaker = np.log(np.maximum(speaker_cfg["mel_log_floor"], speaker)).astype(np.float32)
    for frame_index in range(speaker.shape[1]):
        column = speaker[:, frame_index]
        mean = float(np.sum(column, dtype=np.float64)) / speaker_cfg["n_mels"]
        variance = (
            float(np.sum((column.astype(np.float64) - mean) ** 2, dtype=np.float64))
            / speaker_cfg["n_mels"]
        )
        speaker[:, frame_index] = ((column - mean) / np.sqrt(variance + 1.0e-5)).astype(np.float32)

    acoustic = mel_spectrogram(
        prepared,
        sample_rate=sample_rate,
        n_fft=config["vocoder"]["n_fft"],
        win_length=config["vocoder"]["n_fft"],
        hop_length=config["model"]["acoustic_mel_hop_length"],
        n_mels=config["model"]["acoustic_mel_n_mels"],
        power=1.0,
        mel_scale="htk",
        norm=None,
        window="periodic",
    )
    acoustic = np.log(np.maximum(1.0e-7, acoustic)).astype(np.float32)

    features = {
        "semantic": semantic,
        "speaker": speaker,
        "acoustic": acoustic,
    }
    assert list(semantic.shape) == expected["features"]["semantic_shape"]
    assert list(speaker.shape) == expected["features"]["speaker_shape"]
    assert list(acoustic.shape) == expected["features"]["acoustic_shape"]
    for name, feature in features.items():
        for index, value in expected["features"][name].items():
            np.testing.assert_allclose(feature.reshape(-1)[int(index)], value, rtol=1e-5, atol=2e-5)
