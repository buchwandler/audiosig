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
from audiosig import pitch_shift, time_stretch

sample_rate = 24_000
audio = np.zeros(sample_rate, dtype=np.float32)

faster = time_stretch(audio, rate=1.1)
higher = pitch_shift(audio, sample_rate=sample_rate, semitones=2.0)
```

AudioSig accepts `float32` and `float64` NumPy arrays shaped as `(samples,)`,
`(channels, samples)`, or arbitrary leading batch dimensions followed by the
sample axis. The public functions never mutate their input and preserve the
floating-point dtype.

`time_stretch(rate=1.1)` makes audio faster and shorter; rates below one make
it slower and longer. `pitch_shift(semitones=2)` raises pitch while preserving
the exact input length. Resampling uses a finite windowed-sinc low-pass filter
and produces `round(input_length * target_rate / source_rate)` samples.

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
are required. The full implementation and downstream replacement map are in
[`AUDIOSIG_TRIM_VAD_BRIEF.md`](AUDIOSIG_TRIM_VAD_BRIEF.md).

The phase vocoder and resampler are optimized for speech, TTS, and moderate
prosody changes. They are portable numerical building blocks, not a claim of
transparent extreme music-production quality. Invalid arrays and parameters
raise typed `AudioSig` exceptions so applications can choose their own
fail-open or fail-fast policy.

AudioSig supports Python 3.10 through 3.14 and requires NumPy 1.24 or newer.
The package is licensed under Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the provenance boundary
of the clean-room silence/VAD implementation.

## Downstream integration

PyKokoro is intended to depend on `audiosig>=0.1,<0.2` for volume, rate, pitch,
and silence/VAD processing. PyKokoro should keep parsing SSMD strings at its
application boundary, catch `AudioSignalError`, and return the original audio
when a prosody transform fails. TTSForge should depend on PyKokoro and should
not import AudioSig directly unless it develops an independent DSP use case.

This checkout contains AudioSig only; downstream source changes and release
verification require the PyKokoro and TTSForge repositories.

## Compatibility policy

Direct `resample` keeps AudioSig's historical rounded output length by default;
pass `length_mode="ceil"` when matching librosa's length policy is required.
`pitch_shift` continues to enforce the exact input length. The portable effect
implementation intentionally retains its symmetric Hann window and reflect
center padding, so time/pitch effects are covered by duration, axis, channel,
and finite-output invariants rather than sample-for-sample librosa equality.
