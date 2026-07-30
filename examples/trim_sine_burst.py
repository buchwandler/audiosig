"""Trim synthetic leading and trailing silence around a sine-wave burst."""

from __future__ import annotations

import numpy as np

from audiosig import trim


def main() -> None:
    sample_rate = 24_000
    leading = np.zeros(sample_rate // 4, dtype=np.float32)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.25 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    trailing = np.zeros(sample_rate // 3, dtype=np.float32)
    audio = np.concatenate([leading, tone, trailing])

    trimmed, interval = trim(
        audio,
        top_db=40.0,
        frame_length=512,
        hop_length=128,
    )

    print(f"input samples:   {audio.size}")
    print(f"trim interval:   {interval.tolist()}")
    print(f"trimmed samples: {trimmed.size}")
    print(f"trimmed seconds: {trimmed.size / sample_rate:.3f}")


if __name__ == "__main__":
    main()
