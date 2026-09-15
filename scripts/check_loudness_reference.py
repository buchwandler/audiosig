"""Run deterministic BS.1770 loudness reference checks.

The built-in vector is a 2-second, 440 Hz, -20 dBFS sine. Its expected value
was calculated from the BS.1770 K-weighting, 400 ms block, 100 ms hop, and
absolute/relative gate definitions used by the standards. External reference
WAVs are intentionally not bundled; developers can add locally supplied
vectors to this script without changing the runtime package dependencies.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from audiosig import integrated_loudness, sample_peak_dbfs, true_peak_dbtp

LOUDNESS_TOLERANCE_LU = 0.05
PEAK_TOLERANCE_DB = 0.02
_REFERENCE_SAMPLE_RATE = 24_000
_REFERENCE_LOUDNESS_LUFS = -23.73724


def _reference_signal(sample_rate: int = _REFERENCE_SAMPLE_RATE) -> np.ndarray:
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    return 0.1 * np.sin(2.0 * np.pi * 440.0 * time)


def _check_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(
            f"{name}: expected {expected:.5f} +/- {tolerance:.5f}, got {actual:.5f}"
        )


def main() -> int:
    audio = _reference_signal()
    measured = integrated_loudness(audio, sample_rate=_REFERENCE_SAMPLE_RATE)
    _check_close(
        "24 kHz integrated loudness",
        measured,
        _REFERENCE_LOUDNESS_LUFS,
        LOUDNESS_TOLERANCE_LU,
    )

    louder = integrated_loudness(audio * 10.0 ** (3.0 / 20.0), sample_rate=_REFERENCE_SAMPLE_RATE)
    _check_close("3 dB loudness gain", louder - measured, 3.0, LOUDNESS_TOLERANCE_LU)

    sample_peak = sample_peak_dbfs(audio)
    true_peak = true_peak_dbtp(audio, sample_rate=_REFERENCE_SAMPLE_RATE)
    if true_peak < sample_peak - PEAK_TOLERANCE_DB:
        raise AssertionError(
            f"true peak {true_peak:.5f} dBTP is below sample peak {sample_peak:.5f} dBFS"
        )

    for sample_rate in (24_000, 44_100, 48_000):
        rate_audio = _reference_signal(sample_rate)
        rate_loudness = integrated_loudness(rate_audio, sample_rate=sample_rate)
        if not math.isfinite(rate_loudness):
            raise AssertionError(f"no finite loudness result at {sample_rate} Hz")

    print(
        "PASS: BS.1770 deterministic reference checks "
        f"(24 kHz={measured:.5f} LUFS, sample peak={sample_peak:.5f} dBFS, "
        f"true peak={true_peak:.5f} dBTP; tolerances={LOUDNESS_TOLERANCE_LU:.2f} LU/"
        f"{PEAK_TOLERANCE_DB:.2f} dB)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
