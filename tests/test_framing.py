from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig._framing import frame


def test_frame_shape_axis_and_read_only_view() -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 12)
    frames = frame(source, frame_length=4, hop_length=2)
    assert frames.shape == (2, 5, 4)
    assert not frames.flags.writeable
    np.testing.assert_array_equal(frames[0, 1], source[0, 2:6])


def test_frame_rejects_invalid_axis_and_sizes() -> None:
    source = np.ones(8, dtype=np.float32)
    with pytest.raises(AudioShapeError):
        frame(source, frame_length=2, hop_length=1, axis=2)
    with pytest.raises(InvalidParameterError):
        frame(source, frame_length=0, hop_length=1)
    assert frame(source, frame_length=16, hop_length=2).shape == (0, 16)
    with pytest.raises(InvalidParameterError):
        frame(source, frame_length=2, hop_length=0)
