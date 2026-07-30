AudioSig
========

AudioSig is a portable, NumPy-only signal-processing layer for speech and TTS
applications. It provides gain and normalization, resampling, time and pitch
effects, silence trimming, interval extraction, frame features, and explicit
voice-activity detection policies.

Public behavior
---------------

The package accepts ``float32`` and ``float64`` NumPy arrays and keeps the
sample axis explicit through the public APIs. ``split`` returns clipped,
half-open non-silent sample intervals. ``normalized_energy_vad`` preserves its
historical non-padded default; pass ``pad_end=True`` when trailing partial
speech must be analyzed. ``find_speech_bounds`` returns ``[0, 0]`` when no
speech is found.

See the repository README and ``AUDIOSIG_TRIM_VAD_BRIEF.md`` for the detailed
replacement boundary and downstream audit status.
