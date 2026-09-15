# AudioSig API Reference

## Core Functions

### Waveform Construction and Channels

#### `generate_silence(duration, sample_rate, *, dtype=np.float32)`

Return a newly allocated one-dimensional mono NumPy buffer filled with zeros.
Duration is in seconds, sample rate is in samples per second, and the sample
count is exactly `int(duration * sample_rate)`, so fractional sample counts
are truncated. `dtype` must be a real NumPy floating dtype; the default is
float32. Invalid duration, sample rate, and dtype values raise
`InvalidParameterError`. Long silence creation should use bounded
application-level chunks rather than one arbitrarily large array.

#### `downmix_to_mono(audio, *, channel_axis=-1)`

Return a caller-owned mono array by taking the arithmetic mean over the
explicit channel axis. Input must be a finite real floating one- or
two-dimensional NumPy array; the source dtype and frame order are preserved.
One-dimensional input is already mono and is copied. The operation does not
clip or normalize amplitude. Invalid shape, axis, dtype, or sample values
raise `AudioShapeError`; the result is contiguous.

```python
from audiosig import downmix_to_mono, generate_silence

silence = generate_silence(0.5, 24_000)
mono_soundfile = downmix_to_mono(frames_first, channel_axis=1)
mono_audiosig = downmix_to_mono(channels_first, channel_axis=0)
```

These functions do not perform file decoding/encoding, URL handling, playback,
streaming, or audiobook composition.

### Time and Pitch Effects

#### `time_stretch(audio, rate, *, sample_rate=None, method='phase_vocoder', axis=-1, n_fft=2048, hop_length=None)`

Change audio duration while approximately preserving pitch.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `rate` (float): Stretch factor. Values > 1.0 make audio faster/shorter, < 1.0 slower/longer
- `sample_rate` (int, optional): Required when `method='wsola'` or `method='esola'`; used for speech-time geometry
- `method` (`'phase_vocoder'`, `'wsola'`, or `'esola'`): Select the generic or speech-oriented backend. `'td_psola'` is intentionally not a `time_stretch` method.
- `axis` (int): Sample axis (default: -1)
- `n_fft` (int): FFT window size (default: 2048)
- `hop_length` (int, optional): Hop size. Defaults to n_fft // 4

**Returns:** np.ndarray - Stretched audio with same dtype as input

**Raises:**

- `InvalidParameterError`: If rate is not positive or parameters are invalid
- `AudioShapeError`: If audio array is invalid

ESOLA is an experimental speech backend with exact output length
`round(input_samples / rate)` and supported rates from 0.5 through 2.0. It
does not make a general music-quality or formant-preservation claim.

**Example:**

```python
import numpy as np
from audiosig import time_stretch

audio = np.random.randn(24000).astype(np.float32)
faster = time_stretch(audio, rate=1.2)  # 20% faster
slower = time_stretch(audio, rate=0.8)  # 20% slower
```

---

#### `pitch_shift(audio, *, sample_rate, semitones, bins_per_octave=12, method='phase_vocoder', axis=-1, n_fft=2048, hop_length=None, filter_width=32, rolloff=0.945)`

Shift pitch by semitones while preserving exact input duration.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `sample_rate` (int): Audio sample rate in Hz
- `semitones` (float): Pitch shift in semitones (positive = higher, negative = lower)
- `bins_per_octave` (int): Semitones per octave (default: 12)
- `axis` (int): Sample axis (default: -1)
- `n_fft` (int): FFT window size (default: 2048)
- `hop_length` (int, optional): Hop size
- `filter_width` (int): Resampling filter width (default: 32)
- `method` (`'phase_vocoder'`, `'wsola'`, `'esola'`, or `'td_psola'`): Pitch method. `td_psola` directly synthesizes voiced speech and does not resample the complete waveform.
- `rolloff` (float): Pitch-resampler rolloff (default: 0.945)

**Returns:** np.ndarray - Pitch-shifted audio with exact same length as input

**Example:**

```python
from audiosig import pitch_shift

# Raise pitch by 2 semitones
higher = pitch_shift(audio, sample_rate=24000, semitones=2.0)

# Lower pitch by 5 semitones
lower = pitch_shift(audio, sample_rate=24000, semitones=-5.0)
```

#### `apply_speech_effects(audio, *, sample_rate, rate=1.0, semitones=0.0, gain_db=0.0, axis=-1, clip=False, method='wsola', n_fft=2048, hop_length=None, filter_width=32, rolloff=0.945)`

Apply numeric speech effects using one planned pitch/rate time-scale pass,
optional resampling, and gain. WSOLA is the default speech backend;
`method='phase_vocoder'` selects the generic reference path,
`method='esola'` selects the experimental epoch-synchronous path, and
`method='td_psola'` selects the experimental direct speech pitch/prosody path.
The output length is exactly `round(input_samples / rate)` for non-empty input.
TD-PSOLA supports `0.75 <= rate <= 1.5` and `-6 <= semitones <= 6`; it does not
guarantee vocal-formant preservation. This compositor does not parse SSMD strings and raises typed
AudioSig exceptions for invalid input or parameters.

---

### Resampling

#### `resample(audio, *, source_rate, target_rate, axis=-1, filter_width=32, rolloff=0.945, length_mode='round')`

Resample audio with a windowed-sinc anti-aliasing filter.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `source_rate` (float): Source sample rate in Hz
- `target_rate` (float): Target sample rate in Hz
- `axis` (int): Sample axis (default: -1)
- `filter_width` (int): Filter kernel width (default: 32)
- `rolloff` (float): Filter rolloff frequency (default: 0.945)
- `length_mode` (str): Length calculation mode ('round' or 'ceil')

**Returns:** np.ndarray - Resampled audio

**Output Length:**

- `length_mode='round'`: `round(input_length * target_rate / source_rate)`
- `length_mode='ceil'`: `ceil(input_length * target_rate / source_rate)`

**Example:**

```python
from audiosig import resample

# Downsample from 48kHz to 16kHz
downsampled = resample(audio_48k, source_rate=48000, target_rate=16000)

# Upsample from 8kHz to 24kHz
upsampled = resample(audio_8k, source_rate=8000, target_rate=24000)
```

#### `resample_to_length(audio, length, *, axis=-1, filter_width=32, rolloff=0.945)`

Resample audio to an exact sample count along the selected sample axis. This
is a sample-count operation, not a sample-rate conversion. `length` may be
zero; an empty input can only be resampled to zero samples.

#### `resample_speed(audio, speed, *, axis=-1, filter_width=32, rolloff=0.945)`

Change playback speed by resampling. Values above one make audio shorter and
higher pitched; values below one make it longer and lower pitched. Use
`time_stretch` when pitch should remain approximately unchanged.

```python
from audiosig import resample_speed, resample_to_length

exact = resample_to_length(audio_48k, 12_000)
faster = resample_speed(audio_48k, speed=1.25)
```

---

### Amplitude Processing

#### `apply_gain_db(audio, db, *, clip=False)`

Apply a decibel gain to audio.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `db` (float): Gain in decibels (negative = attenuation)
- `clip` (bool): If True, clip output to [-1.0, 1.0] (default: False)

**Returns:** np.ndarray - Gain-adjusted audio

**Example:**

```python
from audiosig import apply_gain_db

louder = apply_gain_db(audio, db=6.0)    # +6 dB gain
quieter = apply_gain_db(audio, db=-3.0)  # -3 dB attenuation
clipped = apply_gain_db(audio, db=12.0, clip=True)
```

---

#### `peak_normalize(audio, *, peak=1.0, eps=1e-12)`

Scale audio so its absolute peak equals the target value.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `peak` (float): Target peak amplitude (default: 1.0)
- `eps` (float): Silence threshold (default: 1e-12)

**Returns:** np.ndarray - Normalized audio

**Example:**

```python
from audiosig import peak_normalize

# Normalize to 0.9 peak
normalized = peak_normalize(audio, peak=0.9)

# Silent audio is returned unchanged
silent = np.zeros(1000, dtype=np.float32)
still_silent = peak_normalize(silent)
```

---

### Loudness Measurement

#### `integrated_loudness(audio, *, sample_rate, axis=-1)`

Return BS.1770-style integrated programme loudness in LUFS. Audio is K-weighted, divided into complete 400 ms blocks with a 100 ms hop, then processed with the -70 LUFS absolute gate and -10 LU relative gate. Signals shorter than one complete block return `-math.inf` without invented padding. V1 accepts one-dimensional mono `float32`/`float64` arrays; the mandatory sample rate is 24,000 Hz, and 44,100 and 48,000 Hz are also supported. Digital silence returns `-math.inf`.

#### `sample_peak_dbfs(audio, *, axis=-1)`

Return `20 * log10(max(abs(audio)))` in dBFS. Digital silence returns `-math.inf`. This is a sample peak measurement, not a loudness normalization operation.

#### `true_peak_dbtp(audio, *, sample_rate, axis=-1, oversample=4)`

Estimate inter-sample peak in dBTP using AudioSig's NumPy windowed-sinc resampler. `oversample` must be a positive integer; `1` is equivalent to sample peak. The function never clips, limits, or modifies the source audio. Digital silence returns `-math.inf`.

#### `LoudnessMetrics` and `measure_loudness(audio, *, sample_rate, axis=-1, true_peak_oversample=4)`

`LoudnessMetrics` is an immutable dataclass containing `integrated_lufs`, `sample_peak_dbfs`, and `true_peak_dbtp`. `measure_loudness` composes the three measurements without applying policy or changing the input. Stable imports are available from `audiosig`; the PyKokoro minimum release is `0.1.3`.

## Silence Detection

### Voice Activity Detection

#### `normalized_energy_vad(audio, sample_rate, *, frame_duration_ms=5.0, energy_threshold=0.02, axis=-1, pad_end=False)`

Detect voice activity using normalized RMS energy.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `sample_rate` (int): Audio sample rate in Hz
- `frame_duration_ms` (float): Frame duration in milliseconds (default: 5.0)
- `energy_threshold` (float): Threshold in [0, 1] (default: 0.02)
- `axis` (int): Sample axis (default: -1)
- `pad_end` (bool): Include trailing partial frame (default: False)

**Returns:** np.ndarray - Boolean array indicating active frames

**Example:**

```python
from audiosig import normalized_energy_vad

activity = normalized_energy_vad(
    audio,
    sample_rate=24000,
    frame_duration_ms=10,
    energy_threshold=0.15,
)

# Count active frames
active_frames = np.sum(activity)
```

---

#### `relative_db_vad(audio, *, frame_length=2048, hop_length=512, threshold_db=40.0, top_db=None, axis=-1, pad_end=False)`

Detect voice activity using relative dB threshold.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `frame_length` (int): Analysis frame length (default: 2048)
- `hop_length` (int): Hop length (default: 512)
- `threshold_db` (float): dB below peak to consider active (default: 40.0)
- `top_db` (float, optional): Alias for threshold_db
- `axis` (int): Sample axis (default: -1)
- `pad_end` (bool): Include trailing partial frame (default: False)

**Returns:** np.ndarray - Boolean array indicating active frames

---

#### `energy_based_vad(audio, sample_rate=None, *, frame_duration_ms=5.0, energy_threshold=None, frame_length=2048, hop_length=512, threshold_db=40.0, top_db=None, axis=-1, pad_end=False)`

Unified voice activity detection interface.

**Algorithm Selection:**

- With `sample_rate`: Uses `normalized_energy_vad`
- Without `sample_rate`: Uses `relative_db_vad`

**Returns:** np.ndarray - Boolean array indicating active frames

---

### Speech Bounds

#### `find_speech_start(audio, sample_rate=None, **kwargs)`

Find the first active sample index.

**Returns:** int - First active sample index, or 0 if no activity

---

#### `find_speech_bounds(audio, sample_rate=None, **kwargs)`

Find speech start and end bounds.

**Returns:** np.ndarray - `[start, end]` sample indices, or `[0, 0]` if no speech

**Example:**

```python
from audiosig import find_speech_bounds

bounds = find_speech_bounds(audio, sample_rate=24000)
if not np.array_equal(bounds, [0, 0]):
    start, end = bounds
    speech = audio[start:end]
```

---

### Smooth Cut-Point Selection

#### `find_smooth_cut_point(audio, *, start, end, anchor=None, window_length=120, axis=-1)`

Select a deterministic waveform boundary in the half-open sample interval `[start, end)`. The candidate minimizes a fixed combination of local RMS, adjacent endpoint amplitude, cross-boundary slope, and distance from the preferred anchor. This is not a silence detector and returns a legal candidate for continuous voiced or noisy audio.

**Parameters:**

- `audio` (np.ndarray): Finite float32 or float64 audio array
- `start`, `end` (int): Legal half-open search bounds with `0 <= start < end <= sample_count`
- `anchor` (int, optional): Preferred sample index; defaults to the middle legal candidate and may lie outside the interval
- `window_length` (int): Local RMS analysis length in samples, at least 1
- `axis` (int): Sample axis, default `-1`; all other dimensions are conservatively aggregated

**Returns:** `int` for a legal candidate, or `None` only for the explicit empty case `start=0, end=0` on empty audio.

Invalid arrays, axes, bounds, anchors, and parameters raise `AudioShapeError` or `InvalidParameterError`. Ties are resolved by total score, anchor distance, then lower sample index. The caller remains responsible for semantic interval legality and retry policy.

---

### Silence Trimming

#### `trim(audio, *, top_db=60.0, ref=np.max, frame_length=2048, hop_length=512, aggregate=np.max, axis=-1, center=True, pad_mode='constant')`

Trim leading and trailing silence.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `top_db` (float): Silence threshold in dB (default: 60.0)
- `ref` (float or callable): Reference amplitude (default: np.max)
- `frame_length` (int): Analysis frame length (default: 2048)
- `hop_length` (int): Hop length (default: 512)
- `aggregate` (callable): Aggregation function (default: np.max)
- `axis` (int): Sample axis (default: -1)
- `center` (bool): Use centered frames (default: True)
- `pad_mode` (str): Padding mode (default: 'constant')

**Returns:** tuple[np.ndarray, np.ndarray] - (trimmed_audio, [start, end])

**Example:**

```python
from audiosig import trim

trimmed, interval = trim(audio, top_db=40.0)
print(f"Kept samples {interval[0]} to {interval[1]}")
```

---

#### `split(audio, *, top_db=60.0, ref=np.max, frame_length=2048, hop_length=512, aggregate=np.max, axis=-1, center=True, pad_mode='constant')`

Split audio into non-silent intervals.

**Parameters:** Same as `trim()`

**Returns:** np.ndarray - Array of `[start, end)` half-open intervals, shape `(n_intervals, 2)`

**Example:**

```python
from audiosig import split

intervals = split(audio, top_db=40.0, frame_length=512, hop_length=128)

for start, end in intervals:
    segment = audio[start:end]
    process(segment)
```

---

## Frame Analysis

### `frame_signal(audio, *, frame_length=None, hop_length=None, sample_rate=None, frame_ms=20.0, hop_ms=10.0, axis=-1, center=False, pad_mode='constant', pad_end=False)`

Split audio into overlapping frames.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `frame_length` (int, optional): Frame length in samples
- `hop_length` (int, optional): Hop length in samples
- `sample_rate` (int, optional): Sample rate (for ms conversion)
- `frame_ms` (float): Frame duration in ms (default: 20.0)
- `hop_ms` (float): Hop duration in ms (default: 10.0)
- `axis` (int): Sample axis (default: -1)
- `center` (bool): Center frames with padding (default: False)
- `pad_mode` (str): Padding mode (default: 'constant')
- `pad_end` (bool): Pad to include trailing samples (default: False)

**Returns:** np.ndarray - Framed audio with shape `(..., n_frames, frame_length)`

---

### `frame_rms(audio, *, frame_length=2048, hop_length=512, axis=-1, center=True, pad_mode='constant', pad_end=False, normalize=False, dtype=np.float32)`

Compute RMS amplitude for each frame.

**Returns:** np.ndarray - RMS values per frame

Set `pad_end=True` to include a zero-padded trailing partial frame. Set
`normalize=True` to scale each leading-dimension slice independently to
`[0, 1]`.

---

### `short_time_energy(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False, pad_end=False)`

Calculate mean-square energy for each frame.

**Returns:** np.ndarray - Energy values per frame

---

### `zero_crossing_rate(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False, pad_end=False, normalize=False)`

Calculate zero-crossing rate for each frame.

**Parameters:**

- `normalize` (bool): Normalize to [0, 1] range (default: False)

**Returns:** np.ndarray - Zero-crossing rates per frame

---

### `spectral_flux(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False, pad_end=False, window='hann', normalize=False)`

Calculate spectral flux between adjacent frames.

**Parameters:**

- `window` (str): Window function ('hann' or 'hamming')
- `normalize` (bool): Normalize to [0, 1] range (default: False)

**Returns:** np.ndarray - Spectral flux values per frame

---

## Utility Functions

### Decibel Conversion

#### `amplitude_to_db(amplitude, *, ref=1.0, amin=1e-5, top_db=None)`

Convert amplitude to decibels.

**Parameters:**

- `amplitude` (np.ndarray): Input amplitude values
- `ref` (float or callable): Reference amplitude (default: 1.0)
- `amin` (float): Minimum amplitude floor (default: 1e-5)
- `top_db` (float, optional): Clip to this range below peak

**Returns:** np.ndarray - Decibel values

---

#### `power_to_db(power, *, ref=1.0, amin=1e-10, top_db=None)`

Convert power to decibels.

**Parameters:**

- `power` (np.ndarray): Input power values
- `ref` (float or callable): Reference power (default: 1.0)
- `amin` (float): Minimum power floor (default: 1e-10)
- `top_db` (float, optional): Clip to this range below peak

**Returns:** np.ndarray - Decibel values

---

### Other Utilities

#### `rms(audio, *, axis=-1)`

Compute overall RMS amplitude.

**Returns:** np.ndarray - RMS value(s)

---

#### `abs2(values, *, dtype=None)`

Compute squared magnitude for real or complex values.

**Returns:** np.ndarray - Squared values

---

#### `frames_to_samples(frames, *, hop_length=512, n_fft=None)`

Convert frame indices to sample indices.

**Returns:** int or np.ndarray - Sample indices

---

#### `median_filter_numpy(values, size=3, *, window_size=None, mode='edge')`

Apply median filter along the final axis.

**Parameters:**

- `mode` (str): Boundary mode ('edge' or 'truncate')

**Returns:** np.ndarray - Filtered values

---

#### `non_silent_frames(audio, *, top_db=60.0, ref=np.max, frame_length=2048, hop_length=512, aggregate=np.max, axis=-1, center=True, pad_mode='constant')`

Return boolean mask of non-silent frames.

**Returns:** np.ndarray - Boolean mask

---

#### `activity_to_intervals(activity, *, hop_length, sample_count, min_frames=1)`

Convert frame activity mask to clipped sample intervals. `min_frames` filters
out active runs shorter than the requested number of frames.

**Returns:** np.ndarray - Half-open `[start, end)` intervals

---

## Exceptions

### `AudioSignalError`

Base exception for all AudioSig operations.

### `InvalidParameterError(AudioSignalError, ValueError)`

Raised when a parameter is outside the supported domain.

### `AudioShapeError(AudioSignalError, ValueError)`

Raised when audio array shape or dtype is unsupported.

---

## Type Annotations

All public functions include full type annotations. The package is PEP 561 compatible with a `py.typed` marker.

```python
import numpy as np
from audiosig import time_stretch

# Type checking works with mypy/pyright
result: np.ndarray = time_stretch(audio, rate=1.1)
```
