from __future__ import annotations

import numpy as np

import audiosig


def test_new_dsp_primitives_are_available_from_the_package_root() -> None:
    names = (
        "resample_windowed_sinc",
        "SpeechLevel",
        "ReferenceLevelResult",
        "speech_level",
        "normalize_reference_level",
        "PauseCropConfig",
        "crop_on_pause",
        "speech_onset",
        "energy_onset",
        "energy_fraction",
        "trim_lead",
        "trim_trail",
        "fade_edges",
        "join_with_edge_fades",
        "soft_limit",
        "gain_for_level",
        "stft",
        "istft",
        "power_spectrogram",
        "mel_filterbank",
        "mel_spectrogram",
        "mfcc",
        "frame_times",
        "match_frames",
    )
    assert len(audiosig.__all__) == len(set(audiosig.__all__))
    for name in names:
        assert name in audiosig.__all__
        assert getattr(audiosig, name) is not None


def test_public_compatibility_and_spectral_example() -> None:
    source = np.sin(2 * np.pi * 440 * np.arange(240) / 24_000).astype(np.float32)
    resampled = audiosig.resample_windowed_sinc(
        source,
        source_rate=24_000,
        target_rate=16_000,
        radius=16,
        alignment="half_sample",
        window="raised_cosine",
        antialias=False,
    )
    assert resampled.shape == (160,)
    mel = audiosig.mel_spectrogram(
        resampled,
        sample_rate=16_000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        mel_scale="slaney",
        norm="slaney",
        window="periodic",
    )
    assert mel.shape[0] == 80
