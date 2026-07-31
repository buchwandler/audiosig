from __future__ import annotations

import numpy as np

from audiosig import apply_speech_effects


def test_short_speech_like_segments_remain_finite_and_sized() -> None:
    for sample_count in (0, 1, 120, 480, 2400):
        source = np.zeros(sample_count, dtype=np.float32)
        if sample_count:
            source[: max(1, sample_count // 3)] = 0.5
        result = apply_speech_effects(
            source,
            sample_rate=24_000,
            rate=0.75,
            semitones=2.0,
            gain_db=-3.0,
            n_fft=256,
            hop_length=64,
        )
        expected = 0 if sample_count == 0 else round(sample_count / 0.75)
        assert result.size == expected
        assert np.isfinite(result).all()
