# AudioSig API Reference

## Core Functions

### Time and Pitch Effects

#### `time_stretch(audio, rate, *, axis=-1, n_fft=2048, hop_length=None)`

Change audio duration while approximately preserving pitch.

**Parameters:**

- `audio` (np.ndarray): Input audio array
- `rate` (float): Stretch factor. Values > 1.0 make audio faster/shorter, < 1.0 slower/longer
- `axis` (int): Sample axis (default: -1)
- `n_fft` (int): FFT window size (default: 2048)
- `hop_length` (int, optional): Hop size. Defaults to n_fft // 4

**Returns:** np.ndarray - Stretched audio with same dtype as input

**Raises:**

- `InvalidParameterError`: If rate is not positive or parameters are invalid
- `AudioShapeError`: If audio array is invalid

**Example:**

```python
import numpy as np
from audiosig import time_stretch

audio = np.random.randn(24000).astype(np.float32)
faster = time_stretch(audio, rate=1.2)  # 20% faster
slower = time_stretch(audio, rate=0.8)  # 20% slower
```

---

#### `pitch_shift(audio, *, sample_rate, semitones, bins_per_octave=12, axis=-1, n_fft=2048, hop_length=None, filter_width=32)`

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

**Returns:** np.ndarray - Pitch-shifted audio with exact same length as input

**Example:**

```python
from audiosig import pitch_shift

# Raise pitch by 2 semitones
higher = pitch_shift(audio, sample_rate=24000, semitones=2.0)

# Lower pitch by 5 semitones
lower = pitch_shift(audio, sample_rate=24000, semitones=-5.0)
```

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

### `frame_rms(audio, *, frame_length=2048, hop_length=512, axis=-1, center=True, pad_mode='constant', dtype=np.float32)`

Compute RMS amplitude for each frame.

**Returns:** np.ndarray - RMS values per frame

---

### `short_time_energy(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False)`

Calculate mean-square energy for each frame.

**Returns:** np.ndarray - Energy values per frame

---

### `zero_crossing_rate(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False, normalize=False)`

Calculate zero-crossing rate for each frame.

**Parameters:**

- `normalize` (bool): Normalize to [0, 1] range (default: False)

**Returns:** np.ndarray - Zero-crossing rates per frame

---

### `spectral_flux(audio, *, frame_length=2048, hop_length=512, axis=-1, center=False, window='hann', normalize=False)`

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

#### `activity_to_intervals(activity, *, hop_length, sample_count)`

Convert frame activity mask to sample intervals.

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
