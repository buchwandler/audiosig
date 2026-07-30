"""Demonstrate normalized-energy and decibel VAD with synthetic noise bursts."""

from __future__ import annotations

import numpy as np

from audiosig import find_speech_bounds, normalized_energy_vad, relative_db_vad


def active_frame_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a boolean frame mask into half-open active frame ranges."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, mask.size))
    return ranges


def main() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(7)
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    audio[4_000:10_000] = rng.normal(0.0, 0.12, 6_000).astype(np.float32)
    audio[18_000:25_000] = rng.normal(0.0, 0.2, 7_000).astype(np.float32)

    normalized_mask = normalized_energy_vad(
        audio,
        sample_rate,
        frame_duration_ms=10,
        energy_threshold=0.15,
    )
    db_mask = relative_db_vad(
        audio,
        frame_length=320,
        hop_length=160,
        threshold_db=30.0,
    )

    print("Noise is used here as a deterministic stand-in for speech activity.")
    print(f"normalized active ranges: {active_frame_ranges(normalized_mask)}")
    print(f"dB active ranges:         {active_frame_ranges(db_mask)}")
    print(
        "normalized speech bounds:",
        find_speech_bounds(
            audio,
            sample_rate,
            frame_duration_ms=10,
            energy_threshold=0.15,
        ).tolist(),
    )


if __name__ == "__main__":
    main()
