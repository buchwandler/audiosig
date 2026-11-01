"""Repeatable standalone benchmark for AudioSig loudness and peak analysis.

Run ``python benchmarks/bench_loudness.py --durations 60 600 1355 --sample-rates 24000``
for audiobook-sized workloads. Each metric runs in an isolated child process so
``peak_rss_mib`` is a per-case high-water mark rather than process-lifetime RSS.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from audiosig import analyze_loudness  # noqa: E402
from audiosig._resampling import resample  # noqa: E402
from audiosig.loudness import _k_weight  # noqa: E402

_METRICS = (
    "integrated_only",
    "sample_peak_only",
    "true_peak_only",
    "full_analysis",
    "k_weight_core",
)
_WORKER_METRICS = (*_METRICS, "generic_resample_reference")
_K_WEIGHT_CORE_FRAMES = 1_000_000
_SIGNAL_CHUNK_SIZE = 1_048_576


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
    duration_s: float,
    sample_rate: int,
    metric: str,
    seconds: float,
    x_realtime: float,
    input_dtype: str,
) -> str:
    rss = _peak_rss_mib()
    rss_value = "" if rss is None else f"{rss:.1f}"
    return (
        f"{duration_s:.3f},{sample_rate},{metric},{seconds:.6f},"
        f"{x_realtime:.3f},{rss_value},{input_dtype}"
    )


def _make_audio(frames: int, sample_rate: int) -> np.ndarray:
    """Build deterministic float32 sine input without a waveform-sized float64 temporary."""
    audio = np.empty(frames, dtype=np.float32)
    angular_step = 2.0 * math.pi * 997.0 / sample_rate
    for start in range(0, frames, _SIGNAL_CHUNK_SIZE):
        stop = min(start + _SIGNAL_CHUNK_SIZE, frames)
        indices = np.arange(start, stop, dtype=np.float64)
        audio[start:stop] = 0.05 * np.sin(angular_step * indices)
    return audio


def _run_worker(duration: int, sample_rate: int, metric: str) -> str:
    frames = _K_WEIGHT_CORE_FRAMES if metric == "k_weight_core" else duration * sample_rate
    audio = _make_audio(frames, sample_rate)
    measured_duration = frames / sample_rate

    if metric == "integrated_only":
        operation = partial(
            analyze_loudness,
            audio,
            sample_rate=sample_rate,
            integrated=True,
            sample_peak=False,
            true_peak=False,
        )
        input_dtype = "float32"
    elif metric == "sample_peak_only":
        operation = partial(
            analyze_loudness,
            audio,
            sample_rate=sample_rate,
            integrated=False,
            sample_peak=True,
            true_peak=False,
        )
        input_dtype = "float32"
    elif metric == "true_peak_only":
        operation = partial(
            analyze_loudness,
            audio,
            sample_rate=sample_rate,
            integrated=False,
            sample_peak=False,
            true_peak=True,
        )
        input_dtype = "float32"
    elif metric == "full_analysis":
        operation = partial(analyze_loudness, audio, sample_rate=sample_rate)
        input_dtype = "float32"
    elif metric == "k_weight_core":
        operation = partial(_k_weight, audio, sample_rate)
        input_dtype = "float32"
    else:
        audio64 = audio.astype(np.float64)

        def operation() -> None:
            oversampled = resample(
                audio64,
                source_rate=sample_rate,
                target_rate=sample_rate * 4,
                filter_width=32,
            )
            _ = float(np.max(np.abs(oversampled)))

        input_dtype = "float64"

    seconds, x_realtime = _timed(operation, measured_duration)
    return _csv_row(
        measured_duration,
        sample_rate,
        metric,
        seconds,
        x_realtime,
        input_dtype,
    )


def _run_isolated_case(duration: int, sample_rate: int, metric: str) -> str:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--duration",
        str(duration),
        "--sample-rate",
        str(sample_rate),
        "--metric",
        metric,
    ]
    completed = subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_cases(
    durations: list[int], sample_rates: list[int], metrics: Iterable[str]
) -> Iterable[str]:
    selected_metrics = tuple(metrics)
    for duration in durations:
        for sample_rate in sample_rates:
            for metric in selected_metrics:
                if metric != "k_weight_core":
                    yield _run_isolated_case(duration, sample_rate, metric)
    if "k_weight_core" in selected_metrics:
        for sample_rate in sample_rates:
            yield _run_isolated_case(1, sample_rate, "k_weight_core")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", type=int, default=[10, 60, 186, 600])
    parser.add_argument("--sample-rates", nargs="+", type=int, default=[24_000, 44_100, 48_000])
    parser.add_argument(
        "--metric",
        choices=_WORKER_METRICS,
        help="run only this metric for each requested duration/sample-rate pair",
    )
    parser.add_argument(
        "--compare-generic-true-peak",
        action="store_true",
        help="also time the former full-waveform 4x generic resampler reference",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--duration", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sample-rate", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.duration is None or args.sample_rate is None or args.metric is None:
            parser.error("worker mode requires --duration, --sample-rate, and --metric")
        if args.duration <= 0 or args.sample_rate <= 0:
            parser.error("duration and sample rate must be positive")
        print(_run_worker(args.duration, args.sample_rate, args.metric))
        return

    if any(duration <= 0 for duration in args.durations):
        parser.error("durations must be positive")
    if any(sample_rate <= 0 for sample_rate in args.sample_rates):
        parser.error("sample rates must be positive")
    selected = [args.metric] if args.metric else list(_METRICS)
    if args.compare_generic_true_peak:
        selected.append("generic_resample_reference")

    print("duration_s,sample_rate,metric,wall_seconds,x_realtime,peak_rss_mib,input_dtype")
    for row in _run_cases(args.durations, args.sample_rates, selected):
        print(row)


if __name__ == "__main__":
    main()
