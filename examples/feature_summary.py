"""Compute frame-level features for a synthetic sine/noise signal."""

from __future__ import annotations

import numpy as np

from audiosig import (
    frame_rms,
    median_filter_numpy,
    spectral_flux,
    zero_crossing_rate,
)


def main() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(17)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = (0.2 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    noise = rng.normal(0.0, 0.05, sample_rate).astype(np.float32)
    audio = np.concatenate([sine, noise])

    frame_length = 400
    hop_length = 160
    rms_values = frame_rms(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    zcr_values = zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        normalize=True,
    )
    flux_values = spectral_flux(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        window="hamming",
        normalize=True,
    )
    smoothed_rms = median_filter_numpy(rms_values, window_size=5)

    print(f"frames: {rms_values.size}")
    print(f"RMS range: {rms_values.min():.5f} .. {rms_values.max():.5f}")
    print(f"smoothed RMS range: {smoothed_rms.min():.5f} .. {smoothed_rms.max():.5f}")
    print(f"normalized ZCR range: {zcr_values.min():.3f} .. {zcr_values.max():.3f}")
    print(f"normalized flux range: {flux_values.min():.3f} .. {flux_values.max():.3f}")


if __name__ == "__main__":
    main()
