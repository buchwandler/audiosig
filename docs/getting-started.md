# Getting Started with AudioSig

## Installation

Install AudioSig from PyPI:

```bash
pip install audiosig
```

For development with test dependencies:

```bash
pip install audiosig[dev]
```

## Quick Start

### Basic Audio Processing

```python
import numpy as np
from audiosig import pitch_shift, time_stretch, resample

# Create a sample audio signal (1 second at 24kHz)
sample_rate = 24_000
t = np.linspace(0, 1, sample_rate, dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Time stretching (make audio 20% faster)
faster = time_stretch(audio, rate=1.2)

# Pitch shifting (raise pitch by 2 semitones)
higher = pitch_shift(audio, sample_rate=sample_rate, semitones=2.0)

# Resampling (convert from 24kHz to 16kHz)
resampled = resample(audio, source_rate=24_000, target_rate=16_000)
```

### Constructing Silence and Downmixing Channels

Use `generate_silence` for deterministic mono silence. Its sample count uses
the existing TTS/PyKokoro-compatible floor rule,
`int(duration * sample_rate)`, and output is float32 unless float64 is
requested:

```python
from audiosig import generate_silence

pause = generate_silence(0.5, 24_000)
precise_pause = generate_silence(0.5, 24_000, dtype=np.float64)
```

Use `downmix_to_mono` when the decoder has already produced a NumPy array.
Declare the channel axis explicitly: SoundFile-style `(frames, channels)`
data uses `channel_axis=1`, while AudioSig-style `(channels, samples)` data
uses the default `channel_axis=0`.

```python
from audiosig import downmix_to_mono

mono = downmix_to_mono(frames_first, channel_axis=1)
mono = downmix_to_mono(channels_first)
```

Downmixing is an arithmetic mean in the source dtype with no clipping or
normalization, and both functions return caller-owned arrays. AudioSig does
not decode files or stream long silence buffers; applications creating long
files should write bounded silence chunks through their file layer.

### Silence Detection and Trimming

```python
from audiosig import trim, split, normalized_energy_vad

# Trim silence from beginning and end
trimmed, interval = trim(audio, top_db=40.0)

# Split audio into non-silent segments
intervals = split(audio, top_db=40.0)

# Voice Activity Detection
activity = normalized_energy_vad(
    audio,
    sample_rate,
    frame_duration_ms=10,
    energy_threshold=0.15,
)
```

### Amplitude Processing

```python
from audiosig import apply_gain_db, peak_normalize

# Apply gain in decibels
louder = apply_gain_db(audio, db=6.0)

# Normalize to peak amplitude
normalized = peak_normalize(audio, peak=0.9)
```

## Understanding the API

### Array Shapes

AudioSig accepts arrays in these shapes:

- **1D**: `(samples,)` - Single channel audio
- **2D**: `(channels, samples)` - Multi-channel audio
- **nD**: `(..., samples)` - Batch dimensions followed by sample axis

The sample axis defaults to the last dimension (`axis=-1`).

### Data types

- Audio inputs must be `float32` or `float64` NumPy arrays.
- Audio transforms such as gain, resampling, time stretching, pitch shifting,
  and trimming preserve the audio dtype.
- Feature and decibel functions may return `float32` or `float64` as documented.
- VAD functions return boolean masks.
- Interval and frame-index conversion functions return integer values.
- Public operations do not mutate the input audio.

### Time Stretching

`time_stretch(audio, rate)` changes duration while preserving pitch:

- `rate > 1.0`: Faster and shorter
- `rate < 1.0`: Slower and longer
- `rate = 1.0`: No change (returns copy)

```python
from audiosig import time_stretch

# Make audio 50% faster
faster = time_stretch(audio, rate=1.5)

# Make audio 30% slower
slower = time_stretch(audio, rate=0.7)
```

### Pitch Shifting

`pitch_shift(audio, sample_rate, semitones)` changes pitch while preserving duration:

- `semitones > 0`: Higher pitch
- `semitones < 0`: Lower pitch
- `semitones = 0`: No change (returns copy)

```python
from audiosig import pitch_shift

# Raise pitch by 3 semitones
higher = pitch_shift(audio, sample_rate=24000, semitones=3.0)

# Lower pitch by 5 semitones
lower = pitch_shift(audio, sample_rate=24000, semitones=-5.0)
```

For experimental speech-only direct pitch/prosody processing, opt in with
`method="td_psola"`. Moderate changes only are supported; unvoiced regions are
duration-scaled without pitch synthesis and formant preservation is not
guaranteed:

```python
from audiosig import apply_speech_effects, pitch_shift

voice_up = pitch_shift(audio, sample_rate=24_000, semitones=4.0, method="td_psola")
slower_voice = apply_speech_effects(
    audio,
    sample_rate=24_000,
    rate=0.85,
    semitones=-3.0,
    method="td_psola",
)
```

TD-PSOLA is experimental and intended for clean or mildly noisy speech, not
music, polyphony, strong noise, reverb, creaky voice, or extreme shifts.

### Resampling

`resample(audio, source_rate, target_rate)` changes sample rate:

```python
from audiosig import resample

# Downsample from 48kHz to 16kHz
downsampled = resample(audio_48k, source_rate=48000, target_rate=16000)

# Upsample from 8kHz to 24kHz
upsampled = resample(audio_8k, source_rate=8000, target_rate=24000)
```

## Voice Activity Detection (VAD)

AudioSig provides two VAD algorithms:

### Normalized Energy VAD

Best for speech detection with known sample rate:

```python
from audiosig import normalized_energy_vad

activity = normalized_energy_vad(
    audio,
    sample_rate=24000,
    frame_duration_ms=10,      # Frame size in milliseconds
    energy_threshold=0.15,     # Threshold in [0, 1]
    pad_end=False,             # Discard trailing partial frame
)
```

### Relative dB VAD

Best for general audio with relative threshold:

```python
from audiosig import relative_db_vad

activity = relative_db_vad(
    audio,
    frame_length=2048,
    hop_length=512,
    threshold_db=40.0,         # dB below peak
)
```

### Unified Interface

The `energy_based_vad` function selects the algorithm based on parameters:

```python
from audiosig import energy_based_vad

# With sample_rate → normalized energy VAD
activity = energy_based_vad(audio, sample_rate=24000)

# Without sample_rate → relative dB VAD
activity = energy_based_vad(audio, frame_length=2048, hop_length=512)
```

## Silence Trimming

### trim()

Removes leading and trailing silence:

```python
from audiosig import trim

trimmed, interval = trim(
    audio,
    top_db=40.0,          # Silence threshold in dB
    frame_length=2048,    # Analysis frame size
    hop_length=128,       # Hop size
)

# interval is [start, end] sample indices
print(f"Trimmed from sample {interval[0]} to {interval[1]}")
```

### split()

Returns all non-silent segments:

```python
from audiosig import split

intervals = split(
    audio,
    top_db=40.0,
    frame_length=512,
    hop_length=128,
)

# Each interval is [start, end) half-open
for start, end in intervals:
    segment = audio[start:end]
    process_segment(segment)
```

## Error Handling

AudioSig raises typed exceptions for invalid inputs:

```python
from audiosig import AudioSignalError, AudioShapeError, InvalidParameterError

try:
    result = time_stretch(audio, rate=-1.0)
except InvalidParameterError as e:
    print(f"Invalid parameter: {e}")
except AudioShapeError as e:
    print(f"Invalid audio shape: {e}")
except AudioSignalError as e:
    print(f"Audio processing error: {e}")
```

## Next Steps

- See the [API Reference](api-reference.md) for complete function documentation
- Check the [Examples](examples.md) for more usage patterns
- Read the [Advanced Topics](advanced.md) for performance tips and edge cases
