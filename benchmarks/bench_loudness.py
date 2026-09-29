"""Repeatable standalone benchmark for AudioSig loudness and peak analysis.

Run ``python benchmarks/bench_loudness.py --durations 186 --sample-rates 24000``
for a representative Readio programme. Pass ``--compare-generic-true-peak`` to
also time the former full-waveform windowed-sinc measurement path.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Iterable

import numpy as np

from audiosig import integrated_loudness, measure_loudness, sample_peak_dbfs, true_peak_dbtp
from audiosig._resampling import resample


def _peak_rss_mib() -> float | None:
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return None
    # Linux/Android reports KiB; macOS reports bytes.
    return peak / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)


def _timed(callable_: Callable[[], object], duration: float) -> tuple[float, float]:
    start = time.perf_counter()
    callable_()
    elapsed = time.perf_counter() - start
    return elapsed, duration / elapsed if elapsed else math.inf


def _csv_row(
    duration: int,
    sample_rate: int,
    metric: str,
    seconds: float,
    x_realtime: float,
) -> str:
    rss = _peak_rss_mib()
    rss_value = "" if rss is None else f"{rss:.1f}"
    return f"{duration},{sample_rate},{metric},{seconds:.6f},{x_realtime:.3f},{rss_value}"


def _run_case(duration: int, sample_rate: int, compare_generic: bool) -> Iterable[str]:
    frames = duration * sample_rate
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    audio = (0.05 * np.sin(2.0 * np.pi * 997.0 * time_axis)).astype(np.float32)
    del time_axis

    integrated_seconds, integrated_xrt = _timed(
        lambda: integrated_loudness(audio, sample_rate=sample_rate), duration
    )
    sample_seconds, sample_xrt = _timed(lambda: sample_peak_dbfs(audio), duration)
    true_peak_seconds, true_peak_xrt = _timed(
        lambda: true_peak_dbtp(audio, sample_rate=sample_rate), duration
    )
    full_seconds, full_xrt = _timed(
        lambda: measure_loudness(audio, sample_rate=sample_rate), duration
    )
    yield _csv_row(duration, sample_rate, "integrated_loudness", integrated_seconds, integrated_xrt)
    yield _csv_row(duration, sample_rate, "sample_peak_dbfs", sample_seconds, sample_xrt)
    yield _csv_row(duration, sample_rate, "true_peak_dbtp", true_peak_seconds, true_peak_xrt)
    yield _csv_row(duration, sample_rate, "measure_loudness", full_seconds, full_xrt)

    if compare_generic:
        start = time.perf_counter()
        oversampled = resample(
            audio.astype(np.float64),
            source_rate=sample_rate,
            target_rate=sample_rate * 4,
            filter_width=32,
        )
        _ = float(np.max(np.abs(oversampled)))
        elapsed = time.perf_counter() - start
        yield (
            f"{duration},{sample_rate},generic_resample_reference,{elapsed:.6f},"
            f"{duration / elapsed:.3f},{_peak_rss_mib() or ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", type=int, default=[10, 60, 186, 600])
    parser.add_argument("--sample-rates", nargs="+", type=int, default=[24_000, 44_100, 48_000])
    parser.add_argument(
        "--compare-generic-true-peak",
        action="store_true",
        help="also run the former full-waveform 4x generic resampler reference",
    )
    args = parser.parse_args()
    print("duration_s,sample_rate,metric,wall_seconds,x_realtime,peak_rss_mib")
    for duration in args.durations:
        if duration <= 0:
            parser.error("durations must be positive")
        for sample_rate in args.sample_rates:
            if sample_rate <= 0:
                parser.error("sample rates must be positive")
            for row in _run_case(duration, sample_rate, args.compare_generic_true_peak):
                print(row)


if __name__ == "__main__":
    main()
