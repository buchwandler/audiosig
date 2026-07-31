from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, split, time_stretch, trim


def test_advanced_empty_audio_documentation_contract() -> None:
    empty = np.array([], dtype=np.float32)

    trimmed, interval = trim(empty)
    assert len(trimmed) == 0
    np.testing.assert_array_equal(interval, [0, 0])

    intervals = split(empty)
    assert len(intervals) == 0

    with pytest.raises(AudioShapeError):
        time_stretch(empty, rate=1.0)


def test_advanced_silent_audio_documentation_example() -> None:
    silence = np.zeros(24000, dtype=np.float32)

    trimmed, interval = trim(silence, top_db=40.0, ref=np.max)
    assert len(trimmed) == 24000
    np.testing.assert_array_equal(interval, [0, 24000])

    trimmed, interval = trim(silence, top_db=40.0, ref=1.0)
    assert len(trimmed) == 0
    np.testing.assert_array_equal(interval, [0, 0])
