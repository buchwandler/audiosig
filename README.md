# AudioSig

Portable, dependency-light audio signal processing for NumPy arrays.

AudioSig provides the focused DSP operations needed by speech and audiobook
applications without requiring librosa, SciPy, scikit-learn, or native
extensions.

## Installation

```bash
pip install audiosig
```

## Basic usage

```python
import numpy as np
from audiosig import (
    apply_speech_effects,
    pitch_shift,
    resample_speed,
    resample_to_length,
    time_stretch,
)

sample_rate = 24_000
audio = np.zeros(sample_rate, dtype=np.float32)

faster = time_stretch(audio, rate=1.1)
higher = pitch_shift(audio, sample_rate=sample_rate, semitones=2.0)
exact = resample_to_length(audio, 12_000)
faster_playback = resample_speed(audio, speed=1.25)
speech_effects = apply_speech_effects(audio, sample_rate=sample_rate, rate=1.1)
```

AudioSig accepts `float32` and `float64` NumPy arrays shaped as `(samples,)`,
`(channels, samples)`, or arbitrary leading batch dimensions followed by the
sample axis. The public functions never mutate their input and preserve the
floating-point dtype.

`time_stretch(rate=1.1)` makes audio faster and shorter; rates below one make
it slower and longer. `pitch_shift(semitones=2)` raises pitch while preserving
the exact input length. `resample_speed` changes both duration and pitch,
while `resample_to_length` targets an exact sample count. Regular resampling
uses a finite windowed-sinc low-pass filter and produces
`round(input_length * target_rate / source_rate)` samples.

## Silence trimming and VAD

```python
from audiosig import normalized_energy_vad, split, trim

trimmed, interval = trim(
    audio,
    top_db=40.0,
    frame_length=512,
    hop_length=128,
)

activity = normalized_energy_vad(
    audio,
    sample_rate,
    frame_duration_ms=10,
    energy_threshold=0.15,
)

intervals = split(audio, top_db=40.0, frame_length=512, hop_length=128)
```

Passing `sample_rate` selects normalized-energy VAD compatible with the former
PyKokoro helper. Omitting it selects relative-dB VAD with overlapping frame
controls. The package also exports framing, frame RMS, short-time energy,
zero-crossing rate, spectral flux, median smoothing, decibel conversion, and
non-silent-frame detection.
`split` returns clipped half-open `[start, end)` intervals for every active
region, while `trim` returns the single outer interval.
The compatibility `energy_based_vad` wrapper remains available; explicit
`normalized_energy_vad` and `relative_db_vad` functions make the algorithm
choice visible. Normalized VAD preserves its historical `pad_end=False`
default, while `pad_end=True` analyzes trailing partial speech. Use
`find_speech_bounds` when `[0, 0]` must be distinguishable from speech at
sample zero.

Synthetic examples are available under [`examples/`](examples/README.md). They
use sine waves and seeded noise, so no input recordings or optional packages
are required.

The phase vocoder and resampler are optimized for speech, TTS, and moderate
prosody changes. They are portable numerical building blocks, not a claim of
transparent extreme music-production quality. Invalid arrays and parameters
raise typed `AudioSig` exceptions so applications can choose their own
fail-open or fail-fast policy.

AudioSig supports Python 3.10 through 3.14 and requires NumPy 1.24 or newer.
The package is licensed under Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the provenance boundary
of the clean-room silence/VAD implementation.

## PyKokoro integration

AudioSig provides numeric DSP only. PyKokoro should parse SSMD strings at its
application boundary, call AudioSig with numeric parameters, catch
`AudioSignalError`, and decide whether to log, retry, or return the original
audio. A stable dependency range such as `audiosig>=0.1.0,<0.2` should only be
declared after that release is published on the intended package index.

The relevant numeric operations are `apply_speech_effects`, `apply_gain_db`,
`energy_based_vad`, `frame_rms`, `resample`, `resample_speed`,
`resample_to_length`, `trim`, and `activity_to_intervals`.

This checkout contains AudioSig only; downstream source changes and release
verification require the PyKokoro and TTSForge repositories.

## Compatibility policy

Direct `resample` keeps AudioSig's historical rounded output length by default;
pass `length_mode="ceil"` when matching librosa's length policy is required.
`pitch_shift` continues to enforce the exact input length. The portable effect
implementation intentionally retains its symmetric Hann window and reflect
center padding, so time/pitch effects are covered by duration, axis, channel,
and finite-output invariants rather than sample-for-sample librosa equality.
