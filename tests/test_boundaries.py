from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, AudioSignalError, InvalidParameterError, find_smooth_cut_point


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_silence_chooses_anchor(dtype: type[np.floating]) -> None:
    audio = np.zeros(500, dtype=dtype)

    cut = find_smooth_cut_point(audio, start=100, end=300, anchor=220, window_length=32)

    assert cut == 220
    assert isinstance(cut, int)


@pytest.mark.parametrize("anchor, expected", [(20, 100), (400, 299)])
def test_silence_with_external_anchor_chooses_nearest_legal_sample(
    anchor: int, expected: int
) -> None:
    audio = np.zeros(500, dtype=np.float32)

    assert find_smooth_cut_point(audio, start=100, end=300, anchor=anchor) == expected


@pytest.mark.quality
def test_near_zero_waveform_boundary_beats_high_amplitude_anchor() -> None:
    sample_count = 400
    phase = np.linspace(0.0, 8.0 * np.pi, sample_count, endpoint=False)
    audio = np.sin(phase).astype(np.float64)
    anchor = 75

    cut = find_smooth_cut_point(audio, start=40, end=140, anchor=anchor, window_length=9)
    anchor_amplitude = max(abs(audio[anchor - 1]), abs(audio[anchor]))
    cut_amplitude = max(abs(audio[cut - 1]), abs(audio[cut]))

    assert 40 <= cut < 140
    assert cut_amplitude < anchor_amplitude
    assert cut == find_smooth_cut_point(audio, start=40, end=140, anchor=anchor, window_length=9)


@pytest.mark.quality
def test_energy_valley_is_preferred_to_high_energy_region() -> None:
    audio = np.ones(320, dtype=np.float64)
    audio[160:260] = 0.05

    cut = find_smooth_cut_point(audio, start=100, end=280, anchor=210, window_length=24)

    assert 160 <= cut < 260


@pytest.mark.quality
@pytest.mark.parametrize(
    "fixture",
    ["steady_tone", "amplitude_modulated_tone", "harmonic_mixture", "noise", "transient"],
)
def test_representative_quality_fixtures_produce_finite_legal_boundaries(
    fixture: str,
) -> None:
    sample = np.arange(320, dtype=np.float64)
    tone = np.sin(2.0 * np.pi * 7.0 * sample / 320.0)
    if fixture == "steady_tone":
        audio = tone
    elif fixture == "amplitude_modulated_tone":
        audio = (0.2 + 0.8 * (0.5 + 0.5 * tone)) * tone
    elif fixture == "harmonic_mixture":
        audio = tone + 0.4 * np.sin(2.0 * np.pi * 19.0 * sample / 320.0)
    elif fixture == "noise":
        audio = np.random.default_rng(20260914).standard_normal(sample.size)
    else:
        audio = np.zeros(sample.size, dtype=np.float64)
        audio[157:163] = 1.0

    cut = find_smooth_cut_point(audio, start=80, end=240, anchor=160, window_length=24)
    repeat = find_smooth_cut_point(audio, start=80, end=240, anchor=160, window_length=24)

    assert isinstance(cut, int)
    assert 80 <= cut < 240
    assert cut == repeat


def test_continuous_non_silent_signal_still_returns_a_legal_cut() -> None:
    sample = np.arange(800, dtype=np.float64)
    audio = np.sin(2.0 * np.pi * 220.0 * sample / 24_000.0)

    cut = find_smooth_cut_point(audio, start=100, end=700, anchor=420)

    assert isinstance(cut, int)
    assert 100 <= cut < 700


def test_multichannel_scoring_is_conservative() -> None:
    audio = np.zeros((2, 220), dtype=np.float32)
    audio[1, 90] = 1.0

    cut = find_smooth_cut_point(audio, start=60, end=140, anchor=90, window_length=16)

    assert cut != 90
    assert 60 <= cut < 140


def test_axis_support_matches_channels_first_and_samples_first() -> None:
    channels_first = np.zeros((2, 220), dtype=np.float64)
    channels_first[0, 90] = 1.0
    samples_first = channels_first.T

    first = find_smooth_cut_point(channels_first, start=60, end=140, anchor=90, axis=1)
    second = find_smooth_cut_point(samples_first, start=60, end=140, anchor=90, axis=0)

    assert first == second


def test_input_is_not_modified() -> None:
    audio = np.linspace(-1.0, 1.0, 200, dtype=np.float64)
    original = audio.copy()

    find_smooth_cut_point(audio, start=20, end=180, anchor=100)

    np.testing.assert_array_equal(audio, original)


def test_short_legal_interval_returns_a_cut() -> None:
    audio = np.sin(np.arange(32, dtype=np.float32))

    cut = find_smooth_cut_point(audio, start=7, end=8, anchor=100, window_length=1000)

    assert cut == 7


def test_empty_audio_has_explicit_none_contract() -> None:
    assert find_smooth_cut_point(np.empty(0, dtype=np.float32), start=0, end=0) is None


@pytest.mark.parametrize(
    "audio, kwargs, exception",
    [
        ([0.0, 1.0], {"start": 0, "end": 2}, AudioShapeError),
        (np.ones(2, dtype=np.int16), {"start": 0, "end": 2}, AudioShapeError),
        (np.array([0.0, np.nan]), {"start": 0, "end": 2}, AudioShapeError),
        (np.array([0.0, np.inf]), {"start": 0, "end": 2}, AudioShapeError),
        (np.ones(2, dtype=np.float32), {"start": 0, "end": 2, "axis": 2}, AudioShapeError),
        (np.ones(2, dtype=np.float32), {"start": -1, "end": 2}, InvalidParameterError),
        (np.ones(2, dtype=np.float32), {"start": 1, "end": 1}, InvalidParameterError),
        (np.ones(2, dtype=np.float32), {"start": 0, "end": 3}, InvalidParameterError),
        (np.ones(2, dtype=np.float32), {"start": 0.5, "end": 2}, InvalidParameterError),
        (np.ones(2, dtype=np.float32), {"start": 0, "end": 2.0}, InvalidParameterError),
        (
            np.ones(2, dtype=np.float32),
            {"start": 0, "end": 2, "anchor": 1.5},
            InvalidParameterError,
        ),
        (
            np.ones(2, dtype=np.float32),
            {"start": 0, "end": 2, "window_length": 0},
            InvalidParameterError,
        ),
        (
            np.ones(2, dtype=np.float32),
            {"start": 0, "end": 2, "window_length": -1},
            InvalidParameterError,
        ),
    ],
)
def test_invalid_inputs_raise_typed_exceptions(
    audio: object, kwargs: dict[str, object], exception: type[AudioSignalError]
) -> None:
    with pytest.raises(exception):
        find_smooth_cut_point(audio, **kwargs)  # type: ignore[arg-type]


def test_empty_audio_rejects_non_empty_range() -> None:
    with pytest.raises(InvalidParameterError):
        find_smooth_cut_point(np.empty(0, dtype=np.float32), start=0, end=1)
