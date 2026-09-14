# AudioSig Examples

## Basic Audio Processing

### Creating Test Signals

```python
import numpy as np

def create_sine_wave(frequency=440, duration=1.0, sample_rate=24000):
    """Create a sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * frequency * t)

def create_chirp(f_start=220, f_end=880, duration=1.0, sample_rate=24000):
    """Create a frequency sweep for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    frequency = f_start + (f_end - f_start) * t / duration
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    return np.sin(phase).astype(np.float32)
```

### Silence and Channel Layouts

```python
import numpy as np
from audiosig import downmix_to_mono, generate_silence

silence = generate_silence(0.5, 24_000)
assert silence.shape == (12_000,)

left = np.ones(8, dtype=np.float32)
right = np.zeros(8, dtype=np.float32)
frames_first = np.column_stack([left, right])
mono_frames = downmix_to_mono(frames_first)

channels_first = np.stack([left, right])
mono_channels = downmix_to_mono(channels_first, channel_axis=0)
np.testing.assert_array_equal(mono_frames, 0.5)
np.testing.assert_array_equal(mono_channels, 0.5)
```

Silence length uses `int(duration * sample_rate)` truncation. Silence accepts
real floating NumPy dtypes; downmixing accepts finite real floating one- or
two-dimensional audio, preserves the source dtype, returns independent
contiguous storage, and does not clip or normalize. For a long file, produce
bounded silence chunks in the application rather than requesting one
unbounded array; AudioSig deliberately does not provide file I/O or streaming
wrappers.

### Time Stretching Examples

```python
from audiosig import time_stretch

# Create a 1-second tone
audio = create_sine_wave(440, duration=1.0)

# Speed up by 50% (shorter duration)
faster = time_stretch(audio, rate=1.5)
print(f"Original: {len(audio)} samples")
print(f"Faster: {len(faster)} samples")  # ~16000 samples

# Slow down by 50% (longer duration)
slower = time_stretch(audio, rate=0.5)
print(f"Slower: {len(slower)} samples")  # ~48000 samples

# Music tempo adjustment
def adjust_tempo(audio, tempo_factor):
    """Adjust audio tempo by a factor."""
    return time_stretch(audio, rate=tempo_factor)

# Speech rate adjustment
def slow_speech(audio, factor=0.8):
    """Make speech slower and clearer."""
    return time_stretch(audio, rate=factor)
```

### Pitch Shifting Examples

```python
from audiosig import pitch_shift

audio = create_sine_wave(440, duration=1.0, sample_rate=24000)

# Raise pitch by 1 octave (12 semitones)
octave_up = pitch_shift(audio, sample_rate=24000, semitones=12.0)

# Lower pitch by 1 octave
octave_down = pitch_shift(audio, sample_rate=24000, semitones=-12.0)

# Musical transposition
def transpose(audio, semitones, sample_rate=24000):
    """Transpose audio by semitones."""
    return pitch_shift(audio, sample_rate=sample_rate, semitones=semitones)

# A4 (440 Hz) to C5 (523 Hz) - about 3 semitones
c5 = transpose(audio, semitones=3.0)
```

For a speech-only experimental direct pitch path, keep changes moderate and
opt in explicitly:

```python
from audiosig import apply_speech_effects

speech_up = apply_speech_effects(
    audio,
    sample_rate=24_000,
    semitones=4.0,
    method="td_psola",
)
```

TD-PSOLA does not guarantee formant preservation, does not pitch unvoiced
regions, and is not intended for music or polyphonic material.

### Resampling Examples

```python
from audiosig import resample, resample_speed, resample_to_length

# Create audio at 48kHz
audio_48k = create_sine_wave(440, duration=1.0, sample_rate=48000)

# Downsample to 16kHz
audio_16k = resample(audio_48k, source_rate=48000, target_rate=16000)
print(f"48kHz: {len(audio_48k)} samples")
print(f"16kHz: {len(audio_16k)} samples")  # 1/3 of original

# Upsample to 96kHz
audio_96k = resample(audio_48k, source_rate=48000, target_rate=96000)
print(f"96kHz: {len(audio_96k)} samples")  # 2x original

# Exact sample-count and playback-speed operations
exact = resample_to_length(audio_48k, 12_000)
faster_playback = resample_speed(audio_48k, speed=1.25)
```

## Smooth Cut-Point Selection

This numeric example searches a synthetic waveform near a preferred sample. The result is deterministic and remains inside the caller's legal interval:

```python
import numpy as np
from audiosig import find_smooth_cut_point

sample_rate = 24_000
time = np.arange(sample_rate, dtype=np.float32) / sample_rate
audio = np.sin(2.0 * np.pi * 220.0 * time)
candidate = find_smooth_cut_point(
    audio, start=10_000, end=14_000, anchor=12_000, window_length=120
 )
assert 10_000 <= candidate < 14_000
```

AudioSig does not decide whether the interval is semantically legal or whether an application should retry. It only selects a numeric waveform boundary.

## Silence Detection and VAD

### Basic VAD Usage

```python
from audiosig import normalized_energy_vad, energy_based_vad

# Create audio with silence and speech
sample_rate = 24000
silence = np.zeros(sample_rate // 2, dtype=np.float32)  # 0.5s silence
speech = create_sine_wave(440, duration=0.5, sample_rate=sample_rate)
audio = np.concatenate([silence, speech, silence])

# Detect voice activity
activity = normalized_energy_vad(
    audio,
    sample_rate=sample_rate,
    frame_duration_ms=10,
    energy_threshold=0.1,
)

print(f"Active frames: {np.sum(activity)}")
print(f"Total frames: {len(activity)}")
```

### Tuning VAD Parameters

```python
from audiosig import normalized_energy_vad

def detect_speech(audio, sample_rate, sensitivity='medium'):
    """Detect speech with different sensitivity levels."""
    thresholds = {
        'low': 0.05,      # More permissive
        'medium': 0.15,   # Balanced
        'high': 0.30,     # More strict
    }

    return normalized_energy_vad(
        audio,
        sample_rate=sample_rate,
        frame_duration_ms=10,
        energy_threshold=thresholds[sensitivity],
    )

# Usage
activity = detect_speech(audio, sample_rate=24000, sensitivity='medium')
```

### Finding Speech Boundaries

```python
from audiosig import find_speech_bounds, find_speech_start

# Create test audio
sample_rate = 24000
silence = np.zeros(sample_rate // 2, dtype=np.float32)
speech = create_sine_wave(440, duration=1.0, sample_rate=sample_rate)
audio = np.concatenate([silence, speech, silence])

# Find speech bounds
bounds = find_speech_bounds(audio, sample_rate=sample_rate)
print(f"Speech from {bounds[0]} to {bounds[1]}")

# Extract just the speech
if not np.array_equal(bounds, [0, 0]):
    start, end = bounds
    speech_only = audio[start:end]

# Find speech start
start_idx = find_speech_start(audio, sample_rate=sample_rate)
print(f"Speech starts at sample {start_idx}")
```

## Silence Trimming

### Basic Trimming

```python
from audiosig import trim

# Create audio with leading/trailing silence
sample_rate = 24000
silence = np.zeros(sample_rate // 4, dtype=np.float32)  # 0.25s silence
speech = create_sine_wave(440, duration=0.5, sample_rate=sample_rate)
audio = np.concatenate([silence, speech, silence])

# Trim silence
trimmed, interval = trim(audio, top_db=40.0)

print(f"Original length: {len(audio)}")
print(f"Trimmed length: {len(trimmed)}")
print(f"Kept samples {interval[0]} to {interval[1]}")
```

### Splitting Audio Segments

```python
from audiosig import split

# Create audio with multiple speech segments
sample_rate = 24000
segment1 = create_sine_wave(440, duration=0.3, sample_rate=sample_rate)
segment2 = create_sine_wave(880, duration=0.3, sample_rate=sample_rate)
silence = np.zeros(sample_rate // 4, dtype=np.float32)

audio = np.concatenate([segment1, silence, segment2, silence, segment1])

# Split into non-silent segments
intervals = split(audio, top_db=40.0, frame_length=512, hop_length=128)

print(f"Found {len(intervals)} segments")
for i, (start, end) in enumerate(intervals):
    segment = audio[start:end]
    print(f"Segment {i}: samples {start}-{end} ({len(segment)} samples)")
```

### Custom Trimming Parameters

```python
from audiosig import trim

def aggressive_trim(audio, top_db=20.0):
    """More aggressive silence removal."""
    return trim(audio, top_db=top_db, frame_length=1024, hop_length=64)

def gentle_trim(audio, top_db=60.0):
    """More conservative silence removal."""
    return trim(audio, top_db=top_db, frame_length=4096, hop_length=256)

# Use different strategies
trimmed_aggressive, _ = aggressive_trim(audio)
trimmed_gentle, _ = gentle_trim(audio)
```

## Amplitude Processing

### Gain Control

```python
from audiosig import apply_gain_db

audio = create_sine_wave(440, duration=1.0, sample_rate=24000)

# Boost by 6 dB
boosted = apply_gain_db(audio, db=6.0)

# Attenuate by 10 dB
attenuated = apply_gain_db(audio, db=-10.0)

# With clipping protection
loud_clipped = apply_gain_db(audio, db=20.0, clip=True)

# Fade in/out effect
def fade_in(audio, duration_samples):
    """Apply linear fade in."""
    fade = np.linspace(0, 1, duration_samples, dtype=audio.dtype)
    result = audio.copy()
    result[:duration_samples] *= fade
    return result

def fade_out(audio, duration_samples):
    """Apply linear fade out."""
    fade = np.linspace(1, 0, duration_samples, dtype=audio.dtype)
    result = audio.copy()
    result[-duration_samples:] *= fade
    return result
```

### Normalization

```python
from audiosig import peak_normalize

audio = create_sine_wave(440, duration=1.0, sample_rate=24000) * 0.5

# Normalize to peak 0.9
normalized = peak_normalize(audio, peak=0.9)
print(f"Peak before: {np.max(np.abs(audio)):.3f}")
print(f"Peak after: {np.max(np.abs(normalized)):.3f}")

# Normalize to different targets
def normalize_to_target(audio, target_peak=0.95):
    """Normalize to specific peak level."""
    return peak_normalize(audio, peak=target_peak)
```

## Frame Analysis

### Basic Framing

```python
from audiosig import frame_signal, frame_rms

audio = create_sine_wave(440, duration=1.0, sample_rate=24000)

# Split into frames
frames = frame_signal(
    audio,
    frame_length=2048,
    hop_length=512,
)
print(f"Audio shape: {audio.shape}")
print(f"Frames shape: {frames.shape}")  # (n_frames, 2048)

# Compute RMS per frame
rms_values = frame_rms(
    audio,
    frame_length=2048,
    hop_length=512,
)
print(f"RMS shape: {rms_values.shape}")  # (n_frames,)
```

### Feature Extraction

```python
from audiosig import (
    short_time_energy,
    zero_crossing_rate,
    spectral_flux,
)

audio = create_sine_wave(440, duration=1.0, sample_rate=24000)

# Short-time energy
energy = short_time_energy(audio, frame_length=2048, hop_length=512)

# Zero-crossing rate
zcr = zero_crossing_rate(audio, frame_length=2048, hop_length=512)

# Spectral flux
flux = spectral_flux(audio, frame_length=2048, hop_length=512)

print(f"Energy shape: {energy.shape}")
print(f"ZCR shape: {zcr.shape}")
print(f"Flux shape: {flux.shape}")
```

## Real-World Examples

### Audio Preprocessing Pipeline

```python
import numpy as np
from audiosig import (
    trim,
    peak_normalize,
    resample,
    normalized_energy_vad,
)

def preprocess_audio(audio, source_rate, target_rate=16000):
    """Complete audio preprocessing pipeline."""
    # 1. Trim silence
    trimmed, _ = trim(audio, top_db=40.0)

    # 2. Normalize amplitude
    normalized = peak_normalize(trimmed, peak=0.95)

    # 3. Resample if needed
    if source_rate != target_rate:
        resampled = resample(
            normalized,
            source_rate=source_rate,
            target_rate=target_rate,
        )
    else:
        resampled = normalized

    return resampled

# Usage
audio_48k = load_audio("speech.wav")  # Your audio loading function
processed = preprocess_audio(audio_48k, source_rate=48000, target_rate=16000)
```

### Speech Segmentation

```python
import numpy as np
from audiosig import split, trim, normalized_energy_vad

def segment_speech(audio, sample_rate, min_duration_ms=100):
    """Segment audio into speech portions."""
    # Split into non-silent intervals
    intervals = split(
        audio,
        top_db=35.0,
        frame_length=512,
        hop_length=128,
    )

    # Filter by minimum duration
    min_samples = int(sample_rate * min_duration_ms / 1000)
    segments = []

    for start, end in intervals:
        if (end - start) >= min_samples:
            segments.append(audio[start:end])

    return segments

# Usage
audio = load_audio("long_speech.wav")
segments = segment_speech(audio, sample_rate=24000)
print(f"Found {len(segments)} speech segments")
```

### Batch Processing

```python
import numpy as np
from audiosig import time_stretch, pitch_shift

def process_batch(audio_list, rate=1.0, semitones=0.0, sample_rate=24000):
    """Process a batch of audio files."""
    results = []

    for audio in audio_list:
        # Apply time stretch if needed
        if rate != 1.0:
            audio = time_stretch(audio, rate=rate)

        # Apply pitch shift if needed
        if semitones != 0.0:
            audio = pitch_shift(
                audio,
                sample_rate=sample_rate,
                semitones=semitones,
            )

        results.append(audio)

    return results

# Usage
audio_files = [load_audio(f) for f in file_list]
processed = process_batch(audio_files, rate=1.1, semitones=2.0)
```

## Performance Tips

### Memory Efficiency

```python
import numpy as np

# Process in chunks for large files
def process_large_audio(audio, chunk_size=24000):
    """Process large audio in chunks."""
    results = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        processed = process_chunk(chunk)  # Your processing function
        results.append(processed)
    return np.concatenate(results)
```

### Using Appropriate dtypes

```python
# For storage efficiency
audio_int16 = (audio * 32767).astype(np.int16)

# For processing (always use float)
audio_float = audio_int16.astype(np.float32) / 32767.0

# AudioSig preserves dtype
from audiosig import time_stretch
result = time_stretch(audio_float, rate=1.1)  # Returns float32
```
