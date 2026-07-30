"""Small dependency-boundary smoke test for Termux and other portable hosts."""

from __future__ import annotations

import platform
import sys

import numpy as np

import audiosig
from audiosig import apply_gain_db, pitch_shift, resample, time_stretch, trim


def main() -> None:
    sample_rate = 24_000
    samples = np.arange(sample_rate, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)).astype(np.float32)
    gained = apply_gain_db(audio, -3.0)
    stretched = time_stretch(audio, rate=1.1)
    shifted = pitch_shift(audio, sample_rate=sample_rate, semitones=2.0)
    converted = resample(audio, source_rate=sample_rate, target_rate=16_000)
    trimmed, interval = trim(audio, top_db=60.0)
    outputs = [gained, stretched, shifted, converted, trimmed]
    assert all(np.isfinite(value).all() for value in outputs)
    assert gained.shape == audio.shape
    assert stretched.shape == (round(audio.size / 1.1),)
    assert shifted.shape == audio.shape
    assert converted.shape == (16_000,)
    assert interval.shape == (2,)
    forbidden = {"librosa", "scipy", "sklearn", "audiomentations", "torch", "numba"}
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
