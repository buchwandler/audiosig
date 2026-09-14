"""Measure smooth cut-point selection on representative NumPy signals."""

from __future__ import annotations

import argparse
import time

import numpy as np

from audiosig import find_smooth_cut_point


def measure(
    audio: np.ndarray,
    *,
    start: int,
    end: int,
    anchor: int,
    repeats: int,
) -> float:
    for _ in range(3):
        find_smooth_cut_point(audio, start=start, end=end, anchor=anchor)
    started = time.perf_counter()
    for _ in range(repeats):
        find_smooth_cut_point(audio, start=start, end=end, anchor=anchor)
    return (time.perf_counter() - started) * 1_000_000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(20260914)
    for seconds in (1, 3, 10):
        sample_count = seconds * args.sample_rate
        for interval_ms in (20, 40, 80):
            interval = interval_ms * args.sample_rate // 1000
            start = sample_count // 2 - interval // 2
            end = start + interval
            for label, audio in (
                ("mono", rng.standard_normal(sample_count).astype(np.float32)),
                ("stereo", rng.standard_normal((2, sample_count)).astype(np.float32)),
            ):
                micros = measure(
                    audio,
                    start=start,
                    end=end,
                    anchor=(start + end) // 2,
                    repeats=args.repeats,
                )
                print(f"{label:7s} {seconds:2d}s {interval_ms:2d}ms {micros:9.2f} us/call")


if __name__ == "__main__":
    main()
