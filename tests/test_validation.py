from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError, resample, time_stretch, trim
from audiosig._validation import validate_axis, validate_choices, validate_positive


@pytest.mark.parametrize(
    "value", [np.array(1.0), np.array([], dtype=np.float32), np.array([1], dtype=np.int16)]
)
def test_rejects_invalid_shapes_and_dtypes(value: np.ndarray) -> None:
    with pytest.raises(AudioShapeError):
        time_stretch(value, 1.1)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_audio(value: float) -> None:
    with pytest.raises(AudioShapeError):
        time_stretch(np.array([value], dtype=np.float32), 1.1)


def test_rejects_invalid_parameters() -> None:
    audio = np.ones(32, dtype=np.float32)
    with pytest.raises(AudioShapeError):
        time_stretch(audio, 1.1, axis=2)
    with pytest.raises(InvalidParameterError):
        time_stretch(audio, 0)
    with pytest.raises(InvalidParameterError):
        time_stretch(audio, 1.1, n_fft=1)
    with pytest.raises(InvalidParameterError):
        resample(audio, source_rate=0, target_rate=1)
    with pytest.raises(InvalidParameterError):
        resample(audio, source_rate=1, target_rate=1, filter_width=0)
    with pytest.raises(InvalidParameterError):
        resample(audio, source_rate=1, target_rate=1, rolloff=0)
    with pytest.raises(InvalidParameterError):
        validate_positive(np.nan, "value")
    with pytest.raises(InvalidParameterError):
        validate_positive(True, "value")
    with pytest.raises(InvalidParameterError, match="ref callable"):
        trim(audio, ref=lambda values: np.array([1.0, 2.0]))
    with pytest.raises(InvalidParameterError, match="pad_mode"):
        trim(audio, pad_mode="not-a-mode")
    with pytest.raises(AudioShapeError):
        validate_axis(4, 2)
    with pytest.raises(AudioShapeError):
        validate_axis("x", 2)  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        validate_choices("x", ("a", "b"), "choice")
