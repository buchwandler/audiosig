"""Demonstrate non-silent interval extraction with deterministic noise bursts."""

from __future__ import annotations

import numpy as np

from audiosig import split


def main() -> None:
    rng = np.random.default_rng(11)
    audio = np.zeros(16_000, dtype=np.float32)
    audio[2_000:4_500] = rng.normal(0.0, 0.2, 2_500).astype(np.float32)
    audio[9_000:12_500] = rng.normal(0.0, 0.15, 3_500).astype(np.float32)

    intervals = split(audio, frame_length=320, hop_length=160, top_db=25.0)

    print("Noise is used here as a deterministic stand-in for speech activity.")
    print(f"non-silent intervals: {intervals.tolist()}")
    assert intervals.shape[1] == 2
    assert np.all((intervals >= 0) & (intervals <= audio.size))


if __name__ == "__main__":
    main()
