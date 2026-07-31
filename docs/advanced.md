# Advanced Topics

## Understanding the Phase Vocoder

AudioSig's time stretching uses a phase vocoder algorithm:

1. **STFT**: Convert audio to frequency domain using Short-Time Fourier Transform
2. **Phase Modification**: Adjust phase to change timing while preserving pitch
3. **ISTFT**: Convert back to time domain

### Parameters

```python
from audiosig import time_stretch

# Control FFT resolution
result = time_stretch(
    audio,
    rate=1.2,
    n_fft=4096,        # Higher = better frequency resolution, worse time resolution
    hop_length=1024,   # Controls overlap between frames
)
```

### Trade-offs

- **Larger n_fft**: Better frequency separation, worse time precision
- **Smaller n_fft**: Better time precision, worse frequency separation
- **hop_length**: Typically n_fft // 4 for good quality

## Resampling Internals

AudioSig uses windowed-sinc interpolation:

```python
from audiosig import resample

# Control filter characteristics
result = resample(
    audio,
    source_rate=48000,
    target_rate=16000,
    filter_width=64,     # Wider = sharper cutoff, slower
    rolloff=0.945,       # Anti-aliasing filter rolloff
    length_mode='round', # Output length calculation
)
```

### Length Modes

- `'round'`: AudioSig's default, matches historical behavior
- `'ceil'`: Matches librosa's output length semantics

```python
from audiosig import resample

# Compare length modes
result_round = resample(audio, source_rate=44100, target_rate=16000, length_mode='round')
result_ceil = resample(audio, source_rate=44100, target_rate=16000, length_mode='ceil')

print(f"Round: {len(result_round)}")
print(f"Ceil: {len(result_ceil)}")
```

## VAD Algorithm Details

### Normalized Energy VAD

The normalized energy VAD works by:

1. Framing the audio into non-overlapping segments
2. Computing RMS energy per frame
3. Normalizing to [0, 1] range
4. Thresholding to detect activity

```python
from audiosig import normalized_energy_vad

# The algorithm in detail
activity = normalized_energy_vad(
    audio,
    sample_rate=24000,
    frame_duration_ms=10,     # 10ms frames = 240 samples
    energy_threshold=0.15,    # 15% of max energy
    pad_end=False,            # Discard trailing partial frame
)
```

### Relative dB VAD

The relative dB VAD works by:

1. Computing short-time energy
2. Finding peak energy
3. Setting threshold relative to peak
4. Detecting frames above threshold

```python
from audiosig import relative_db_vad

activity = relative_db_vad(
    audio,
    frame_length=2048,
    hop_length=512,
    threshold_db=40.0,  # 40 dB below peak
)
```

## Edge Cases and Special Handling

### Empty Audio

```python
import numpy as np
from audiosig import trim, split

# Empty arrays are handled gracefully
empty = np.array([], dtype=np.float32)

# trim returns empty array and [0, 0]
trimmed, interval = trim(empty)
assert len(trimmed) == 0
assert np.array_equal(interval, [0, 0])

# split returns empty intervals
intervals = split(empty)
assert len(intervals) == 0
```

`trim`, `split`, and selected analysis helpers accept empty arrays. Audio
effects such as `time_stretch` and `pitch_shift` require a non-empty sample
axis and raise `AudioShapeError` for empty audio.

### Silent Audio

```python
import numpy as np
from audiosig import trim, peak_normalize

# All-zeros audio
silence = np.zeros(24000, dtype=np.float32)

# With the default peak-relative reference, uniform silence remains unchanged.
trimmed, interval = trim(silence, top_db=40.0, ref=np.max)
assert len(trimmed) == 24000
assert np.array_equal(interval, [0, 24000])

# A fixed nonzero reference classifies the all-zero signal as silence.
trimmed, interval = trim(silence, top_db=40.0, ref=1.0)
assert len(trimmed) == 0
assert np.array_equal(interval, [0, 0])

# Normalization preserves silence
normalized = peak_normalize(silence)
assert np.all(normalized == 0)
```

### Very Short Audio

```python
import numpy as np
from audiosig import time_stretch, pitch_shift

# Single sample
single = np.array([0.5], dtype=np.float32)

# Time stretching works with minimum length 1
result = time_stretch(single, rate=2.0)
assert len(result) >= 1

# Pitch shifting preserves length
result = pitch_shift(single, sample_rate=24000, semitones=3.0)
assert len(result) == 1
```

## Multi-Channel and Batch Processing

### Stereo Audio

```python
import numpy as np
from audiosig import time_stretch, pitch_shift

# Stereo audio: (2, samples)
stereo = np.random.randn(2, 24000).astype(np.float32)

# Processing works on last axis by default
stretched = time_stretch(stereo, rate=1.1)
assert stretched.shape[0] == 2  # Channels preserved

# Can specify axis explicitly
result = time_stretch(stereo, rate=1.1, axis=-1)
```

### Batch Dimensions

```python
import numpy as np
from audiosig import time_stretch

# Batch of audio: (batch, channels, samples)
batch = np.random.randn(8, 2, 24000).astype(np.float32)

# Processing preserves batch dimensions
result = time_stretch(batch, rate=1.2)
assert result.shape == (8, 2, 20000)  # Shorter due to rate > 1
```

## Error Handling Strategies

### Fail-Fast vs Fail-Open

```python
from audiosig import (
    time_stretch,
    AudioSignalError,
    InvalidParameterError,
    AudioShapeError,
)

def safe_time_stretch(audio, rate, fail_fast=True):
    """Time stretch with error handling."""
    try:
        return time_stretch(audio, rate=rate)
    except InvalidParameterError as e:
        if fail_fast:
            raise
        print(f"Warning: {e}")
        return audio  # Return original
    except AudioShapeError as e:
        if fail_fast:
            raise
        print(f"Warning: Invalid shape: {e}")
        return audio

# Fail-fast (default)
result = safe_time_stretch(audio, rate=-1.0, fail_fast=True)

# Fail-open (return original on error)
result = safe_time_stretch(audio, rate=-1.0, fail_fast=False)
```

### Input Validation

```python
import numpy as np
from audiosig import AudioShapeError

def validate_audio_input(audio, expected_sr=None):
    """Validate audio before processing."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(audio)}")

    if audio.dtype not in (np.float32, np.float64):
        raise ValueError(f"Expected float32 or float64, got {audio.dtype}")

    if audio.ndim == 0:
        raise ValueError("Audio must have at least 1 dimension")

    if audio.size == 0:
        raise ValueError("Audio array is empty")

    return True
```

## Performance Optimization

### Memory Layout

```python
import numpy as np

# C-contiguous arrays are faster
audio = np.ascontiguousarray(audio)

# Fortran-contiguous for column-major operations
# (not typically needed for audio)
```

### Chunked Processing

```python
import numpy as np
from audiosig import time_stretch

def process_large_file(audio, rate, chunk_seconds=10, sample_rate=24000):
    """Process large audio in chunks."""
    chunk_size = chunk_seconds * sample_rate
    results = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        processed = time_stretch(chunk, rate=rate)
        results.append(processed)

    return np.concatenate(results)
```

### Type Selection

```python
import numpy as np

# Float32: Good for most applications, lower memory
audio_32 = np.random.randn(24000).astype(np.float32)

# Float64: Higher precision, more memory
audio_64 = np.random.randn(24000).astype(np.float64)

# Processing preserves dtype
from audiosig import time_stretch
result_32 = time_stretch(audio_32, rate=1.1)  # Returns float32
result_64 = time_stretch(audio_64, rate=1.1)  # Returns float64
```

## Integration with Other Libraries

### With librosa

```python
import librosa
from audiosig import resample

# Load with librosa
audio, sr = librosa.load("file.wav", sr=None)

# Resample with AudioSig (NumPy-only)
resampled = resample(audio, source_rate=sr, target_rate=16000)
```

### With soundfile

```python
import soundfile as sf
from audiosig import trim, peak_normalize

# Load audio
audio, sr = sf.read("file.wav")

# Process with AudioSig
trimmed, _ = trim(audio, top_db=40.0)
normalized = peak_normalize(trimmed, peak=0.95)

# Save
sf.write("processed.wav", normalized, sr)
```

### With PyTorch

```python
import torch
import numpy as np
from audiosig import time_stretch

# Convert to numpy
audio_torch = torch.randn(1, 24000)
audio_np = audio_torch.numpy().squeeze()

# Process with AudioSig
processed_np = time_stretch(audio_np, rate=1.1)

# Convert back to torch
processed_torch = torch.from_numpy(processed_np).unsqueeze(0)
```

## Debugging and Profiling

### Checking Intermediate Results

```python
import numpy as np
from audiosig import frame_signal, frame_rms

def analyze_frames(audio, frame_length=2048, hop_length=512):
    """Debug frame analysis."""
    frames = frame_signal(audio, frame_length=frame_length, hop_length=hop_length)
    rms = frame_rms(audio, frame_length=frame_length, hop_length=hop_length)

    print(f"Audio length: {len(audio)}")
    print(f"Number of frames: {len(frames)}")
    print(f"Frame shape: {frames.shape}")
    print(f"RMS range: [{np.min(rms):.4f}, {np.max(rms):.4f}]")

    return frames, rms
```

### Performance Profiling

```python
import time
import numpy as np
from audiosig import time_stretch

def profile_time_stretch(audio, rate, iterations=100):
    """Profile time_stretch performance."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = time_stretch(audio, rate=rate)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average: {avg_time*1000:.2f} ms")
    print(f"Std dev: {std_time*1000:.2f} ms")
    print(f"Throughput: {len(audio) / avg_time / 1e6:.2f} Msamples/s")

    return avg_time, std_time
```
