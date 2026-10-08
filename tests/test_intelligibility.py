from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig.intelligibility import estoi, stoi
from audiosig.metrics import MetricResult

IntelligibilityMetric = Callable[..., MetricResult]


def _oracle_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261008)
    reference = rng.normal(size=40_000)
    estimate = reference + 0.2 * rng.normal(size=40_000)
    return reference, estimate


def test_stoi_estoi_against_independent_scientific_reference_values() -> None:
    # Oracle: PySTOI 0.4.1 (github.com/mpariente/pystoi), queried as a black box;
    # its test_stoi_octave.py compares against the original Octave implementation.
    # Algorithm sources: Taal et al. (ICASSP 2010, doi:10.1109/ICASSP.2010.5495701)
    # and Jensen & Taal (2016, doi:10.1109/TASLP.2016.2585878). This implementation
    # follows the papers' equations; no PySTOI source is used or vendored.
    reference, estimate = _oracle_fixture()
    stoi_result = stoi(reference, estimate, sample_rate=10_000)
    estoi_result = estoi(reference, estimate, sample_rate=10_000)

    assert stoi_result.value == pytest.approx(0.9589928689124777, abs=2e-4, rel=0.0)
    assert estoi_result.value == pytest.approx(0.9558169218900828, abs=2e-4, rel=0.0)
    assert isinstance(stoi_result, MetricResult)
    assert isinstance(estoi_result, MetricResult)
    assert stoi_result.name == "stoi"
    assert estoi_result.name == "estoi"
    assert stoi_result.unit == estoi_result.unit == "score"
    assert stoi_result.higher_is_better and estoi_result.higher_is_better
    assert stoi_result.parameters["sample_rate"] == 10_000
    assert stoi_result.parameters["n_fft"] == 512
    assert stoi_result.parameters["segment_frames"] == 30
    assert stoi_result.parameters["window"] == "hann_symmetric"
    assert stoi_result.parameters["center"] is False
    assert stoi_result.parameters["epsilon"] == 1e-12
    assert stoi_result.parameters["band_spacing_octaves"] == pytest.approx(1.0 / 3.0)
    assert stoi_result.parameters["beta_db"] == -15.0
    assert estoi_result.parameters["segment_frames"] == 30
    assert stoi_result.diagnostics["segments"] == 282


def test_intelligibility_identity_determinism_and_input_ownership() -> None:
    reference, _ = _oracle_fixture()
    original = reference.copy()
    first = stoi(reference, reference.copy(), sample_rate=10_000)
    second = stoi(reference, reference.copy(), sample_rate=10_000)
    extended = estoi(reference, reference.copy(), sample_rate=10_000)
    assert first.value == pytest.approx(1.0, abs=1e-12)
    assert second.value == first.value
    assert extended.value == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(reference, original)


def test_intelligibility_accepts_float32_and_float64() -> None:
    reference, estimate = _oracle_fixture()
    scores = []
    for dtype in (np.float32, np.float64):
        result = estoi(reference.astype(dtype), estimate.astype(dtype), sample_rate=10_000)
        assert result.value is not None and np.isfinite(result.value)
        scores.append(result.value)
    assert scores[0] == pytest.approx(scores[1], abs=1e-6)


@pytest.mark.parametrize("metric", [stoi, estoi])
def test_intelligibility_rejects_non_10khz_unequal_length_and_stereo(
    metric: IntelligibilityMetric,
) -> None:
    reference, estimate = _oracle_fixture()
    with pytest.raises(InvalidParameterError, match="10000 Hz"):
        metric(reference, estimate, sample_rate=16_000)
    with pytest.raises(InvalidParameterError, match="10000 Hz"):
        metric(reference, estimate, sample_rate=10_000, estimate_sample_rate=8_000)
    with pytest.raises(AudioShapeError, match="sample counts"):
        metric(reference, estimate[:-1], sample_rate=10_000)
    with pytest.raises(AudioShapeError, match="mono"):
        metric(
            np.stack([reference, reference]),
            np.stack([estimate, estimate]),
            sample_rate=10_000,
        )


@pytest.mark.parametrize("metric", [stoi, estoi])
def test_intelligibility_rejects_bad_types_alignment_and_insufficient_speech(
    metric: IntelligibilityMetric,
) -> None:
    reference, estimate = _oracle_fixture()
    with pytest.raises(AudioShapeError):
        metric(reference.astype(np.int16), estimate.astype(np.int16), sample_rate=10_000)
    with pytest.raises(InvalidParameterError, match="alignment"):
        metric(reference, estimate, sample_rate=10_000, alignment=cast(Any, "shift"))
    with pytest.raises(InvalidParameterError, match="short"):
        metric(np.ones(200), np.ones(200), sample_rate=10_000)
    with pytest.raises(InvalidParameterError, match="no active frames"):
        metric(np.zeros(40_000), np.zeros(40_000), sample_rate=10_000)
