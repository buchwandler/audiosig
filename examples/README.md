# AudioSig examples

The examples use generated NumPy arrays, so they require no input audio files and no optional dependencies.

Run them from an installed or editable AudioSig checkout:

```bash
PYTHONPATH=. python examples/trim_sine_burst.py
PYTHONPATH=. python examples/vad_noise_bursts.py
PYTHONPATH=. python examples/feature_summary.py
PYTHONPATH=. python examples/split_noise_bursts.py
```

- `trim_sine_burst.py` surrounds a 440 Hz tone with silence and reports the retained sample interval.
- `vad_noise_bursts.py` uses seeded noise as a stand-in for speech and compares normalized-energy VAD with dB-relative VAD.
- `split_noise_bursts.py` converts two seeded noise bursts into clipped non-silent sample intervals.
- `feature_summary.py` compares frame RMS, zero-crossing rate, spectral flux, and median smoothing across sine and noise regions.
