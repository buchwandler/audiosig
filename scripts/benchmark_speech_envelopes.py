#!/usr/bin/env python3
"""Measure static and time-varying speech effects on deterministic audio."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from time import perf_counter

import numpy as np

import audiosig
from audiosig import (
    apply_speech_effects,
    apply_speech_effects_envelope,
    speech_effects_output_frames,
)


def _speech_like(sample_rate: int, seconds: float) -> np.ndarray:
    sample_count = max(1, round(sample_rate * seconds))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    f0 = 140.0 + 12.0 * np.sin(2.0 * np.pi * 0.7 * time)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    voiced = (
        0.18 * np.sin(phase)
        + 0.08 * np.sin(2.0 * phase)
        + 0.04 * np.sin(3.0 * phase)
        + 0.02 * np.sin(4.0 * phase)
    )
    envelope = np.minimum(1.0, time * 40.0) * np.minimum(1.0, (seconds - time) * 40.0)
    return (voiced * envelope).astype(np.float32)


def _measure(
    source: np.ndarray,
    sample_rate: int,
    duration: float,
    mode: str,
    operation: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    started = perf_counter()
    result = operation(source)
    elapsed = perf_counter() - started
    output_frames = result.shape[-1]
    output_seconds = output_frames / sample_rate
    return {
        "input_seconds": duration,
        "input_frames": source.shape[-1],
        "sample_rate": sample_rate,
        "mode": mode,
        "output_frames": output_frames,
        "output_seconds": output_seconds,
        "runtime_seconds": elapsed,
        "real_time_factor": output_seconds / elapsed if elapsed else float("inf"),
        "finite": bool(np.isfinite(result).all()),
        "peak": float(np.max(np.abs(result))) if result.size else 0.0,
    }


def benchmark(durations: tuple[float, ...], sample_rate: int) -> list[dict[str, object]]:
    rate_points = ((0.0, 1.0), (0.45, 0.85))
    pitch_points = ((0.0, 0.0), (0.3, 2.0))
    records: list[dict[str, object]] = []
    for duration in durations:
        source = _speech_like(sample_rate, duration)
        input_frames = source.shape[-1]
        rate_frames = speech_effects_output_frames(
            input_frames, sample_rate=sample_rate, rate_points=rate_points
        )
        comparison_rate = input_frames / rate_frames
        operations = (
            (
                "static_rate",
                lambda audio, rate=comparison_rate: apply_speech_effects(
                    audio, sample_rate=sample_rate, rate=rate, method="wsola"
                ),
            ),
            (
                "variable_rate",
                lambda audio: apply_speech_effects_envelope(
                    audio, sample_rate=sample_rate, rate_points=rate_points
                ),
            ),
            (
                "static_pitch",
                lambda audio: apply_speech_effects(
                    audio, sample_rate=sample_rate, semitones=2.0, method="td_psola"
                ),
            ),
            (
                "variable_pitch",
                lambda audio: apply_speech_effects_envelope(
                    audio, sample_rate=sample_rate, pitch_points=pitch_points
                ),
            ),
            (
                "static_combined",
                lambda audio, rate=comparison_rate: apply_speech_effects(
                    audio,
                    sample_rate=sample_rate,
                    rate=rate,
                    semitones=2.0,
                    method="td_psola",
                ),
            ),
            (
                "combined_variable",
                lambda audio: apply_speech_effects_envelope(
                    audio,
                    sample_rate=sample_rate,
                    rate_points=rate_points,
                    pitch_points=pitch_points,
                ),
            ),
        )
        duration_records = [
            _measure(source, sample_rate, duration, mode, operation)
            for mode, operation in operations
        ]
        static_rate_frames = max(1, round(input_frames / comparison_rate))
        expected_by_mode = {
            "static_rate": static_rate_frames,
            "variable_rate": rate_frames,
            "static_pitch": input_frames,
            "variable_pitch": input_frames,
            "static_combined": static_rate_frames,
            "combined_variable": rate_frames,
        }
        for record in duration_records:
            expected_frames = expected_by_mode[str(record["mode"])]
            record["constant_comparison_rate"] = comparison_rate
            record["expected_output_frames"] = expected_frames
            record["length_error"] = int(record["output_frames"]) - expected_frames
        runtime_by_mode = {
            str(record["mode"]): float(record["runtime_seconds"]) for record in duration_records
        }
        for record in duration_records:
            mode = str(record["mode"])
            baseline = {
                "variable_rate": "static_rate",
                "variable_pitch": "static_pitch",
                "combined_variable": "static_combined",
            }.get(mode)
            record["baseline_mode"] = baseline
            record["relative_overhead"] = (
                float(record["runtime_seconds"]) / runtime_by_mode[baseline]
                if baseline is not None and runtime_by_mode[baseline] > 0.0
                else None
            )
        records.extend(duration_records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", type=float, nargs="+", default=(1.0, 10.0, 60.0))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    records = benchmark(tuple(args.durations), args.sample_rate)
    payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "audiosig": audiosig.__version__,
            "platform": platform.platform(),
        },
        "records": records,
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {len(records)} benchmark records to {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
