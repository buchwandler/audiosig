#!/usr/bin/env python3
"""Benchmark the experimental TD-PSOLA speech-effects path on deterministic fixtures."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from audiosig import apply_speech_effects


def _fixture(sample_rate: int, seconds: float, lanes: int) -> np.ndarray:
    samples = max(1, round(sample_rate * seconds))
    time = np.arange(samples, dtype=np.float64) / sample_rate
    voiced = 0.16 * np.sin(2.0 * np.pi * 140.0 * time)
    voiced += 0.08 * np.sin(2.0 * np.pi * 280.0 * time)
    envelope = np.minimum(1.0, time * 40.0) * np.minimum(1.0, (seconds - time) * 40.0)
    noise = np.random.default_rng(20260731).normal(0.0, 0.012, samples)
    source = (voiced * envelope + noise).astype(np.float32)
    if lanes == 1:
        return source
    return np.stack([source * (1.0 - 0.05 * lane) for lane in range(lanes)])


def benchmark(sample_rate: int, seconds: float, lanes: int) -> list[dict[str, object]]:
    source = _fixture(sample_rate, seconds, lanes)
    records: list[dict[str, object]] = []
    for rate, semitones in ((0.8, -4.0), (1.0, 4.0), (1.25, 4.0)):
        started = perf_counter()
        result = apply_speech_effects(
            source,
            sample_rate=sample_rate,
            rate=rate,
            semitones=semitones,
            method="td_psola",
        )
        elapsed = perf_counter() - started
        input_samples = source.shape[-1]
        output_samples = result.shape[-1]
        records.append(
            {
                "sample_rate": sample_rate,
                "input_seconds": seconds,
                "lanes": lanes,
                "rate": rate,
                "semitones": semitones,
                "input_samples": input_samples,
                "output_samples": output_samples,
                "length_error": output_samples - max(1, round(input_samples / rate)),
                "runtime_seconds": elapsed,
                "audio_seconds_per_compute_second": (
                    output_samples / sample_rate / elapsed if elapsed else float("inf")
                ),
                "finite": bool(np.isfinite(result).all()),
                "peak": float(np.max(np.abs(result))) if result.size else 0.0,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--seconds", type=float, nargs="+", default=(1.0, 10.0))
    parser.add_argument("--lanes", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    records = [
        item
        for seconds in args.seconds
        for lanes in args.lanes
        for item in benchmark(args.sample_rate, seconds, lanes)
    ]
    payload = json.dumps(records, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
