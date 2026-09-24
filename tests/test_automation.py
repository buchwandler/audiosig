from __future__ import annotations

import numpy as np
import pytest

from audiosig._automation import _LinearEnvelope, _RateMap, _validate_points
from audiosig.exceptions import InvalidParameterError


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (((0.0, 3.0),), [3.0, 3.0, 3.0]),
        (((0.0, 1.0), (2.0, 5.0)), [1.0, 3.0, 5.0]),
        (((0.0, -2.0), (1.0, 0.0), (3.0, 4.0)), [-2.0, 1.0, 4.0]),
    ],
)
def test_linear_envelope_interpolates_and_holds_final_value(
    points: tuple[tuple[float, float], ...], expected: list[float]
) -> None:
    envelope = _LinearEnvelope(points)
    times = (0.0, points[-1][0] / 2.0, points[-1][0] + 10.0)
    np.testing.assert_allclose([envelope.value_at(time) for time in times], expected)


def test_omitted_envelope_uses_its_constant_default() -> None:
    assert _LinearEnvelope(default=1.0).value_at(5.0) == 1.0


@pytest.mark.parametrize(
    "points",
    [
        ((1.0, 2.0),),
        ((0.0, 1.0), (0.0, 2.0)),
        ((0.0, 1.0), (2.0, 2.0), (1.0, 3.0)),
        ((-0.1, 1.0),),
        ((np.nan, 1.0),),
        ((np.inf, 1.0),),
        ((0.0, np.nan),),
        ((0.0, np.inf),),
        ((0.0, 0.0),),
        ((0.0, -1.0),),
    ],
)
def test_rate_points_reject_invalid_times_and_values(
    points: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(InvalidParameterError):
        _validate_points(points, name="rate_points", positive_values=True)


@pytest.mark.parametrize("points", [((0.5, 1.0),), ((0.0, 1.0, 2.0),), ((0.0,),)])
def test_control_points_require_zero_start_and_two_values(
    points: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(InvalidParameterError):
        _validate_points(points, name="control_points")


def test_linear_envelope_rejects_nonfinite_or_negative_evaluation_time() -> None:
    envelope = _LinearEnvelope(((0.0, 1.0),))
    for time in (np.nan, np.inf, -1.0):
        with pytest.raises(InvalidParameterError):
            envelope.value_at(time)


@pytest.mark.parametrize(
    ("points", "times", "expected"),
    [
        (((0.0, 1.0),), (0.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
        (((0.0, 0.5),), (0.0, 1.0, 2.0), (0.0, 0.5, 1.0)),
        (((0.0, 2.0),), (0.0, 1.0, 2.0), (0.0, 2.0, 4.0)),
        (((0.0, 1.0), (2.0, 0.5)), (0.0, 1.0, 2.0), (0.0, 0.875, 1.5)),
        (((0.0, 0.5), (2.0, 1.0)), (0.0, 1.0, 2.0), (0.0, 0.625, 1.5)),
    ],
)
def test_rate_map_integrates_constant_and_linear_segments(
    points: tuple[tuple[float, float], ...],
    times: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    rate_map = _RateMap(points)
    np.testing.assert_allclose(
        [rate_map.source_time_at(time) for time in times], expected, rtol=0.0, atol=1e-14
    )


def test_rate_map_integrates_multiple_segments_and_holds_final_rate() -> None:
    rate_map = _RateMap(((0.0, 1.0), (2.0, 0.5), (3.0, 2.0)))
    np.testing.assert_allclose(
        [rate_map.source_time_at(time) for time in (0.0, 2.0, 3.0, 4.0)],
        (0.0, 1.5, 2.75, 4.75),
        rtol=0.0,
        atol=1e-14,
    )


def test_rate_map_inverts_constant_linear_and_multi_segment_times() -> None:
    rate_map = _RateMap(((0.0, 1.0), (2.0, 0.5), (3.0, 2.0)))
    for output_time in (0.0, 0.2, 1.0, 2.0, 2.5, 3.0, 4.0, 12.0):
        source_time = rate_map.source_time_at(output_time)
        assert rate_map.output_time_for_source_time(source_time) == pytest.approx(
            output_time, rel=1e-13, abs=1e-13
        )


def test_rate_map_inverse_at_exact_control_point_source_times() -> None:
    rate_map = _RateMap(((0.0, 1.0), (2.0, 0.5), (3.0, 2.0)))
    assert rate_map.output_time_for_source_time(1.5) == pytest.approx(2.0)
    assert rate_map.output_time_for_source_time(2.75) == pytest.approx(3.0)


def test_rate_map_output_frame_count_uses_inverse_map_and_rounding() -> None:
    assert _RateMap(((0.0, 0.5),)).output_frames_for_input_frames(24_000, 24_000) == 48_000
    assert _RateMap(((0.0, 2.0),)).output_frames_for_input_frames(24_000, 24_000) == 12_000

    rate_map = _RateMap(((0.0, 1.0), (2.0, 0.5)))
    expected_seconds = 4.0 - np.sqrt(8.0)
    assert rate_map.output_frames_for_input_frames(24_000, 24_000) == round(
        expected_seconds * 24_000
    )
    assert rate_map.output_frames_for_input_frames(0, 24_000) == 0


def test_rate_map_rejects_invalid_times_and_frame_arguments() -> None:
    rate_map = _RateMap()
    for value in (np.nan, np.inf, -1.0):
        with pytest.raises(InvalidParameterError):
            rate_map.source_time_at(value)
        with pytest.raises(InvalidParameterError):
            rate_map.output_time_for_source_time(value)
    for frames, sample_rate in ((-1, 24_000), (1.5, 24_000), (1, 0), (1, 24_000.5)):
        with pytest.raises(InvalidParameterError):
            rate_map.output_frames_for_input_frames(frames, sample_rate)
