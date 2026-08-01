# TD-PSOLA listening evaluation — 2026-07-31

## Status

This is the milestone-5 evaluation record and reproducible protocol for the
experimental direct TD-PSOLA speech path. No default method is changed by this
work. Listener results remain **pending** until a licensed corpus and at least
two development listeners are available; this file deliberately does not invent
ratings.

The implementation now uses approximately-two-period centered grains, local-F0
pitch-mark propagation, tracker voiced intervals for synthesis masking, and an
explicit insufficient-mark fallback. Synthetic leakage, normalization, sample-
rate, noise, and reverberation checks are automated; they do not constitute the
real-speech listening gate below.

## Corpus and reproducibility

Use licensed or consented 16/24/48 kHz mono or stereo WAV material and record,
for every source: corpus/license, attribution, sample rate, filename or stable
identifier, channel count, and whether the segment is clean, noisy, breathy,
creaky, reverberant, or TTS-generated. Do not commit private speech recordings.

Render with:

```bash
python scripts/compare_speech_effects.py INPUT.wav OUTPUT_DIR \
  --rates 0.8 1.0 1.25 --semitones -6 -4 -2 2 4 6
```

The comparison output records exact-length error, finite output, RMS/peak,
continuity jumps, runtime, and generated WAVs. Keep `metrics.json` and the
command line with the evaluation notes.

## Listening matrix

Include lower- and higher-pitched voices, breathy voice, rapid F0 motion,
plosive-rich and fricative-rich text, mixed voiced/unvoiced speech, a short TTS
fragment, silence boundaries, mild noise, creaky/fry speech, and intelligible
reverberant speech.

Compare the unmodified reference, current WSOLA plus resampling, current ESOLA
plus resampling where its rate is supported, phase-vocoder diagnostics, and
direct TD-PSOLA. Randomize A/B or ABX order and do not reveal method names.

For each trial, collect 1–5 ratings for naturalness, intelligibility, speaker
or timbre preservation, pitch correctness, buzziness, flutter, echo/reverb,
transient duplication, unvoiced tonalization, clicks, and overall preference.
Record severe-artifact flags and free-text comments.

## Promotion gates

- At least two listeners for development screening and at least ten before any
  default-method consideration.
- TD-PSOLA preferred over current WSOLA pitch shifting in at least 65% of
  moderate ±2/±4 semitone trials.
- Severe artifacts below 5% of trials, with no intelligibility regression.
- Objective synthetic and runtime gates pass alongside the listening result.

Until these fields are populated and reviewed, TD-PSOLA remains experimental and
the existing defaults remain phase vocoder for generic `time_stretch` and WSOLA
for `apply_speech_effects`.
