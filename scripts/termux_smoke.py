"""Small dependency-boundary smoke test for Termux and other portable hosts."""

from __future__ import annotations

import platform
import sys

import numpy as np

import audiosig
from audiosig import (
    activity_to_intervals,
    apply_gain_db,
    apply_speech_effects,
    frame_rms,
    pitch_shift,
    resample,
    resample_speed,
    resample_to_length,
    time_stretch,
    trim,
)


def main() -> None:
    sample_rate = 24_000
    samples = np.arange(sample_rate, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)).astype(np.float32)
    gained = apply_gain_db(audio, -3.0)
    stretched = time_stretch(audio, rate=1.1)
    shifted = pitch_shift(audio, sample_rate=sample_rate, semitones=2.0)
    converted = resample(audio, source_rate=sample_rate, target_rate=16_000)
    exact = resample_to_length(audio, 12_000)
    speed = resample_speed(audio, 1.25)
    empty = np.empty(0, dtype=np.float32)
    empty_result = apply_speech_effects(empty, sample_rate=sample_rate)
    energy = frame_rms(
        audio[:5000],
        frame_length=512,
        hop_length=512,
        center=False,
        pad_end=True,
        normalize=True,
    )
    quiet = activity_to_intervals(
        ~np.array([True, False, False, True]),
        hop_length=512,
        sample_count=2048,
        min_frames=2,
    )
    composed = apply_speech_effects(
        audio[:5000],
        sample_rate=sample_rate,
        rate=1.1,
        semitones=1.0,
        gain_db=-3.0,
    )
    trimmed, interval = trim(audio, top_db=60.0)
    outputs = [
        gained,
        stretched,
        shifted,
        converted,
        exact,
        speed,
        empty_result,
        energy,
        quiet,
        composed,
        trimmed,
    ]
    assert all(np.isfinite(value).all() for value in outputs)
    assert gained.shape == audio.shape
    assert stretched.shape == (round(audio.size / 1.1),)
    assert shifted.shape == audio.shape
    assert converted.shape == (16_000,)
    assert exact.shape == (12_000,)
    assert speed.shape == (round(audio.size / 1.25),)
    assert empty_result.shape == empty.shape
    assert energy.shape == (10,)
    assert quiet.shape == (1, 2)
    assert composed.shape == (round(5000 / 1.1),)
    assert interval.shape == (2,)
    forbidden = {
        "librosa",
        "scipy",
        "sklearn",
        "audiomentations",
        "torch",
        "numba",
        "signalsmith_stretch",
    }
    loaded = forbidden.intersection(sys.modules)
    assert not loaded, loaded
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print(f"AudioSig: {audiosig.__version__}")
    print(
        f"outputs: gain={gained.shape}, stretch={stretched.shape}, "
        f"pitch={shifted.shape}, resample={converted.shape}"
    )
    print("AudioSig Termux smoke test passed")


if __name__ == "__main__":
    main()
