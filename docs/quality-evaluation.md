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
combined WSOLA and ESOLA pitch/rate planner cases. CSV/JSON records include
runtime, real-time factor, peak, RMS, exact-length error, and continuity
diagnostics. Use `--rates` and `--semitones` to narrow a run. The harness does not install,
invoke, or require Rubber Band, SoundTouch, WORLD, or another external backend.

The required performance matrix is reproducible with:

```bash
python scripts/benchmark_esola.py --output esola-benchmark.json
```

It covers 1, 10, and 60 seconds; 16, 24, and 48 kHz; mono and four lanes; and
rates 0.75, 1.25, 1.5, and 2.0.

## Listening protocol

For a practical release check, use at least ten utterances and six moderate
transform settings. Randomize A/B or ABX order, compare WSOLA, ESOLA, and the
phase-vocoder diagnostic, and record preference plus artifact notes. Include
two or more listeners, including an experienced speed listener for high-rate
cases when possible. The promotion gate is at least 65% overall ESOLA
preference at moderate speech rates with no severe recurring artifact class.
Keep source-license and attribution details with the evaluation notes. The
dated workspace status is recorded in
[`docs/esola-listening-evaluation-2026-07-31.md`](esola-listening-evaluation-2026-07-31.md).

Native pitch shifting changes the spectral envelope along with F0 and therefore
does not currently preserve vocal formants. Larger shifts are expected to be
more artifact-prone; this harness records metrics but does not turn them into a
claim of studio-grade perceptual quality.
