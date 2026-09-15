[![PyPI - Version](https://img.shields.io/pypi/v/audiosig)](https://pypi.org/project/audiosig/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiosig)
![PyPI - Downloads](https://img.shields.io/pypi/dm/audiosig)

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
    generate_silence,
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
pause = generate_silence(0.25, sample_rate)
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

## Waveform construction and channel downmixing

`generate_silence(duration, sample_rate)` returns a newly allocated,
one-dimensional mono buffer of zeros. Duration is in seconds, sample rate is
in samples per second, and its length is exactly `int(duration * sample_rate)`
using truncation/floor semantics. The default dtype is float32; any NumPy real
floating dtype, such as float16, float32, or float64, may be requested. Invalid
duration, sample rate, or dtype values raise `InvalidParameterError`. For long
silence files, generate bounded chunks in the application and stream those
chunks to the file layer; AudioSig does not provide file I/O or a streaming API.

`downmix_to_mono(audio, channel_axis=-1)` averages the explicitly selected
channel axis in the input dtype. It accepts finite real floating one- or
two-dimensional arrays, preserves frame order and dtype, performs no clipping
or normalization, and returns contiguous caller-owned storage. One-dimensional
audio is treated as already mono and copied. Invalid shape, axis, dtype, or
sample values raise `AudioShapeError`. SoundFile-style frames-first data uses
the default `channel_axis=-1`; channels-first data uses `channel_axis=0`:

```python
from audiosig import downmix_to_mono

silence = generate_silence(0.5, 24_000)  # (12_000,), float32
mono_frames = downmix_to_mono(frames, channel_axis=1)
mono_channels = downmix_to_mono(channels, channel_axis=0)
```

Both operations are NumPy-array primitives only. They do not decode or encode
WAV/FLAC/MP3 files, resolve URLs, play audio, or compose audiobook chapters.

## Loudness and peak measurement

AudioSig provides measurement primitives for comparing audio levels without modifying the waveform:

- `sample_peak_dbfs` reports the largest discrete sample in dBFS.
- `peak_normalize` scales to a requested sample peak; it is not perceived-loudness normalization.
- RMS and short-time energy describe signal energy over a selected window, but are not LUFS.
- `integrated_loudness` applies BS.1770-style K-weighting and absolute/relative gating to report LUFS.
- `true_peak_dbtp` oversamples to estimate inter-sample peak in dBTP.

The v1 loudness meter accepts finite one-dimensional mono `float32` or `float64` arrays. It uses 400 ms blocks with a 100 ms hop, returns `-math.inf` for digital silence, and does not pad signals shorter than one complete block:

```python
import audiosig

metrics = audiosig.measure_loudness(audio, sample_rate=24_000)
print(metrics.integrated_lufs)
print(metrics.sample_peak_dbfs)
print(metrics.true_peak_dbtp)

before = audiosig.integrated_loudness(audio, sample_rate=24_000)
louder = audiosig.apply_gain_db(audio, 3.0)
after = audiosig.integrated_loudness(louder, sample_rate=24_000)
assert abs((after - before) - 3.0) < 0.05
```

The stable PyKokoro handoff imports are `LoudnessMetrics`, `integrated_loudness`, `sample_peak_dbfs`, `true_peak_dbtp`, and `measure_loudness` from `audiosig`; the minimum release containing this API is `0.1.3`. AudioSig only measures and applies generic gain. It does not choose a target LUFS, normalize every segment, or implement voice calibration.

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

AudioSig provides speech-oriented NumPy WSOLA and experimental ESOLA backends
for moderate rate and prosody changes, plus a basic phase-vocoder backend for
generic numerical use. Select ESOLA with `method="esola"` and provide
`sample_rate`; its supported rate range is `0.5 <= rate <= 2.0`. The
speech-effects compositor uses WSOLA by default and combines pitch and rate in
one time-scale pass. ESOLA is not a general music stretcher, does not claim
formant preservation, and has not been validated as an extreme-speed solution.
The experimental `method="td_psola"` option is available only through
`pitch_shift` and `apply_speech_effects`: it directly modifies voiced speech
pulses and uses WSOLA for unvoiced duration changes. It is limited to
`-6 <= semitones <= 6` and `0.75 <= rate <= 1.5`, is not a music or polyphonic
pitch shifter, and does not guarantee formant preservation. It is not a
default; see the [TD-PSOLA listening protocol](docs/td-psola-listening-evaluation-2026-07-31.md)
before considering any promotion.
All public effects preserve exact output-length, dtype, axis, copy, and finite
input contracts. Invalid arrays and parameters raise typed `AudioSig`
exceptions so applications can choose their own fail-open or fail-fast policy.

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

`find_smooth_cut_point` selects a waveform-safe candidate inside a caller-provided numeric half-open interval. It uses the sample axis and optional preferred anchor supplied by the caller, but it does not decide whether that interval is semantically legal or whether a downstream application should retry.

```python
from audiosig import find_smooth_cut_point

candidate = find_smooth_cut_point(
    audio,
    start=search_start,
    end=search_end,
    anchor=preferred_index,
    window_length=120,
 )
```

This checkout contains AudioSig only; downstream source changes and release
verification require the PyKokoro and TTSForge repositories.

## Compatibility policy

Direct `resample` keeps AudioSig's historical rounded output length by default;
pass `length_mode="ceil"` when matching librosa's length policy is required.
`pitch_shift` continues to enforce the exact input length. The portable effect
implementation intentionally retains its symmetric Hann window and reflect
center padding, so time/pitch effects are covered by duration, axis, channel,
and finite-output invariants rather than sample-for-sample librosa equality.
