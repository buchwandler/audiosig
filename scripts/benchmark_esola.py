#!/usr/bin/env python3
"""Benchmark ESOLA across the implementation brief's duration/rate matrix."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from time import perf_counter

import numpy as np

from audiosig import time_stretch
from audiosig._esola import _estimate_trend_window_ms, _extract_epochs_lane


def _speech_like(length: int, sample_rate: int, lanes: int) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    mono = (
        0.35 * np.sin(2.0 * np.pi * 180.0 * time) + 0.15 * np.sin(2.0 * np.pi * 90.0 * time) ** 3
    ).astype(np.float64)
    return np.stack([mono * (1.0 - 0.1 * index) for index in range(lanes)])


def benchmark(
    durations: tuple[int, ...] = (1, 10, 60),
    sample_rates: tuple[int, ...] = (16_000, 24_000, 48_000),
    lanes_values: tuple[int, ...] = (1, 4),
    rates: tuple[float, ...] = (0.75, 1.25, 1.5, 2.0),
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for duration in durations:
        for sample_rate in sample_rates:
            for lanes in lanes_values:
                source = _speech_like(duration * sample_rate, sample_rate, lanes)
                for rate in rates:
                    tracemalloc.start()
                    started = perf_counter()
                    result = time_stretch(
                        source,
                        rate,
                        sample_rate=sample_rate,
                        method="esola",
                        axis=-1,
                    )
                    elapsed = perf_counter() - started
                    _, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    audio_seconds = result.shape[-1] / sample_rate
                    records.append(
                        {
                            "duration_seconds": duration,
                            "sample_rate": sample_rate,
                            "lanes": lanes,
                            "rate": rate,
                            "samples": result.shape[-1],
                            "runtime_seconds": elapsed,
                            "real_time_factor": (
                                audio_seconds / elapsed if elapsed else float("inf")
                            ),
                            "peak_memory_bytes": peak_memory,
                            "finite": bool(np.isfinite(result).all()),
                        }
                    )
    return records


def benchmark_synthetic_epochs(
    sample_rate: int = 16_000,
    duration_seconds: float = 2.0,
    frequencies: tuple[float, ...] = (60.0, 100.0, 180.0, 300.0, 500.0),
) -> list[dict[str, object]]:
    """Report adaptive-window and one/two-pass epoch timing on known tones."""
    length = max(8, round(sample_rate * duration_seconds))
    time = np.arange(length, dtype=np.float64) / sample_rate
    records: list[dict[str, object]] = []
    for frequency in frequencies:
        source = np.sin(2.0 * np.pi * frequency * time)
        trend_window_ms = _estimate_trend_window_ms(source, sample_rate)
        for detrend_passes in (1, 2):
            started = perf_counter()
            epochs = _extract_epochs_lane(
                source,
                sample_rate,
                trend_window_ms=trend_window_ms,
                detrend_passes=detrend_passes,
            )
            elapsed = perf_counter() - started
            spacing = np.diff(epochs)
            records.append(
                {
                    "frequency_hz": frequency,
                    "sample_rate": sample_rate,
                    "trend_window_ms": trend_window_ms,
                    "detrend_passes": detrend_passes,
                    "epoch_count": int(epochs.size),
                    "median_period_samples": (float(np.median(spacing)) if spacing.size else None),
                    "expected_period_samples": sample_rate / frequency,
                    "runtime_seconds": elapsed,
                    "finite": bool(np.isfinite(epochs).all()),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="esola-benchmark.json")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run the fast known-tone epoch/ZFR parameter benchmark",
    )
    args = parser.parse_args()
    records = benchmark_synthetic_epochs() if args.synthetic else benchmark()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    print(f"wrote {len(records)} benchmark records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
