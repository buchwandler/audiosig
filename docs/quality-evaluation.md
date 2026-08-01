# Speech effects quality evaluation

AudioSig's automated quality tests use a deterministic source-filter-style
fixture. It combines changing-F0 harmonics, broad formant emphasis, amplitude
modulation, breath noise, plosive bursts, and a partial tail. This provides
repeatable regression coverage; it is not a substitute for listening to real
speech.

## Local comparison harness

The no-network harness reads PCM WAV with the Python standard library and
writes deterministic 16-bit WAV renders plus CSV and JSON diagnostics:

```bash
python scripts/compare_speech_effects.py input.wav comparison-output
```

It renders the basic phase-vocoder, WSOLA, and ESOLA rate paths, then supported
combined WSOLA, ESOLA, and TD-PSOLA pitch/rate cases. CSV/JSON records include
runtime, real-time factor, peak, RMS, exact-length error, and continuity
diagnostics. Use `--rates` and `--semitones` to narrow a run. The harness does not install,
invoke, or require Rubber Band, SoundTouch, WORLD, or another external backend.

The required performance and robustness matrix is reproducible with:

```bash
python scripts/benchmark_td_psola.py --output td-psola-benchmark.json
```

The TD-PSOLA benchmark covers deterministic 1 and 10 second fixtures at 8, 16,
24, and 48 kHz with mono and two-lane inputs, clean/noisy/reverberant material,
rates 0.8, 1.0, and 1.25, shifts -4 and +4, runtime, and peak memory. ESOLA's
known-tone ZFR comparison is available with:

```bash
python scripts/benchmark_esola.py --synthetic --output esola-benchmark.json
```

These objective checks cover complete-frame endpoint handling, adaptive
trend-window behavior, approximately-two-period TD-PSOLA grains, local-F0
pitch marks, fallback behavior, and exact output contracts. They do not replace
the real-speech listening protocol below.

## Listening protocol

For the TD-PSOLA release check, use the corpus categories, randomized A/B or
ABX matrix, listener fields, and promotion gates in
[`docs/td-psola-listening-evaluation-2026-07-31.md`](td-psola-listening-evaluation-2026-07-31.md).
No default change is justified by synthetic metrics alone.

Native pitch shifting changes the spectral envelope along with F0 and therefore
does not currently preserve vocal formants. Larger shifts are expected to be
more artifact-prone; this harness records metrics but does not turn them into a
claim of studio-grade perceptual quality.
