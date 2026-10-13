from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audiosig import InvalidParameterError
from audiosig.boundaries import PauseCropConfig, crop_on_pause
from audiosig.speech_level import normalize_reference_level

_SAMPLE_RATE = 1_000


def _config(**overrides: float) -> PauseCropConfig:
    values: dict[str, float] = {
        "frame_seconds": 0.025,
        "hop_seconds": 0.010,
        "floor_quantile": 0.10,
        "quiet_multiplier": 4.0,
        "pause_min_seconds": 0.10,
        "pause_keep_seconds": 0.15,
        "search_forward_seconds": 5.0,
        "search_backward_seconds": 5.0,
        "min_keep_fraction": 0.75,
        "room_tone_seconds": 0.25,
        "fade_seconds": 0.02,
    }
    values.update(overrides)
    return PauseCropConfig(**values)


def _active_without_long_pause(length: int) -> np.ndarray:
    phase = np.arange(length) % 100
    return np.where(phase < 40, 0.001, 0.3).astype(np.float32)


def test_crop_on_pause_selects_forward_pause_and_keeps_pause_tail() -> None:
    source = np.concatenate([np.full(1_200, 0.2), np.zeros(250), np.full(550, 0.2)]).astype(
        np.float32
    )
    actual = crop_on_pause(source, target_seconds=1.0, sample_rate=_SAMPLE_RATE, config=_config())
    assert actual.shape == (1_375,)
    np.testing.assert_array_equal(actual[:1_200], source[:1_200])
    assert np.max(np.abs(actual[1_200:])) == 0.0


def test_crop_on_pause_selects_nearby_backward_pause() -> None:
    source = np.concatenate([np.full(600, 0.2), np.zeros(400), np.full(1_000, 0.2)]).astype(
        np.float64
    )
    actual = crop_on_pause(source, target_seconds=1.0, sample_rate=_SAMPLE_RATE, config=_config())
    assert actual.shape[0] < 1_000
    assert actual.shape[0] > 600
    assert np.max(np.abs(actual[600:])) == 0.0


def test_crop_on_pause_no_pause_uses_seeded_room_tone() -> None:
    source = _active_without_long_pause(2_000)
    first = crop_on_pause(
        source,
        target_seconds=1.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_seed=19,
    )
    repeated = crop_on_pause(
        source,
        target_seconds=1.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_seed=19,
    )
    other = crop_on_pause(
        source,
        target_seconds=1.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_seed=20,
    )
    assert first.shape == (1_250,)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert first[999] == 0.0
    assert np.std(first[1_000:]) == pytest.approx(0.001, rel=0.2)


def test_short_and_already_short_reference_contracts() -> None:
    short = np.linspace(-0.1, 0.1, 80, dtype=np.float32)
    unchanged = crop_on_pause(
        short, target_seconds=0.05, sample_rate=_SAMPLE_RATE, config=_config()
    )
    np.testing.assert_array_equal(unchanged, short)
    assert not np.shares_memory(unchanged, short)

    already_short_quiet = np.concatenate([np.full(1_200, 0.2), np.zeros(800)]).astype(np.float32)
    quiet_result = crop_on_pause(
        already_short_quiet,
        target_seconds=3.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_seed=19,
    )
    np.testing.assert_array_equal(quiet_result, already_short_quiet)

    already_short_active = _active_without_long_pause(1_000).astype(np.float64)
    completed = crop_on_pause(
        already_short_active,
        target_seconds=2.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_seed=19,
    )
    assert completed.shape == (1_250,)
    assert completed[999] == 0.0


def test_crop_on_pause_supports_multichannel_axis_and_injected_rng() -> None:
    source = np.broadcast_to(_active_without_long_pause(2_000), (2, 2_000)).copy()
    rng = np.random.default_rng(5)
    result = crop_on_pause(
        source,
        target_seconds=1.0,
        sample_rate=_SAMPLE_RATE,
        config=_config(),
        room_tone_rng=rng,
        axis=1,
    )
    assert result.shape == (2, 1_250)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()


def test_crop_on_pause_validates_configuration_and_seed() -> None:
    source = np.ones(200, dtype=np.float32)
    with pytest.raises(InvalidParameterError, match="room_tone_seed"):
        crop_on_pause(source, target_seconds=1.0, sample_rate=_SAMPLE_RATE, room_tone_seed=-1)
    with pytest.raises(InvalidParameterError, match="either"):
        crop_on_pause(
            source,
            target_seconds=1.0,
            sample_rate=_SAMPLE_RATE,
            room_tone_rng=np.random.default_rng(1),
            room_tone_seed=1,
        )
    with pytest.raises(InvalidParameterError, match="min_keep_fraction"):
        crop_on_pause(
            source,
            target_seconds=1.0,
            sample_rate=_SAMPLE_RATE,
            config=_config(min_keep_fraction=1.2),
        )


def test_pinned_sopro_room_tone_reference_golden() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sopro_reference_golden.json"
    golden = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = golden["tones"]["room_tone"]
    sample_rate = expected["sample_rate"]
    indices = np.arange(expected["samples"], dtype=np.float64)
    source = np.zeros(expected["samples"], dtype=np.float32)
    active_from = expected["active_from_sample"]
    source[active_from:] = (
        0.12 * np.sin(2.0 * np.pi * 220.0 * indices[active_from:] / sample_rate)
    ).astype(np.float32)

    cropped = crop_on_pause(
        source,
        target_seconds=10.0,
        sample_rate=sample_rate,
        room_tone_seed=1,
    )
    assert cropped.size == expected["prepared_samples"]
    normalized = normalize_reference_level(cropped, sample_rate=sample_rate)
    assert normalized.resulting_db == pytest.approx(expected["level_db"], abs=1e-6)
    for index, value in expected["prepared"].items():
        np.testing.assert_allclose(normalized.audio[int(index)], value, rtol=1e-5, atol=1e-8)
