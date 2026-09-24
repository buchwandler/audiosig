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

For envelope A/B renders, add `--envelopes`. It adds rate transitions from `1.0` to each requested rate over 0.45 output seconds, pitch transitions from `0.0` to each requested semitone target over 0.30 seconds, and combined transitions. For a focused run on a permitted speech WAV:

```bash
python scripts/compare_speech_effects.py input.wav comparison-output --rates 0.9 --semitones 2 --envelopes
```

The generated `envelope_*.wav` files and metrics record the numeric control points and exact expected frame count. No speech recording is distributed with AudioSig.

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

## Time-varying envelope evaluation

The envelope regression tests live in `tests/test_automation.py`, `tests/test_speech_envelope.py`, and `tests/test_speech_envelope_quality.py`. They cover analytical integration and inverse-map round trips, exact output sizing, amplitude-coded source landmarks, local pitch trajectories, combined automation, voiced/unvoiced fallback, output-knot jumps, and high-frequency burst guards. These objective tests are diagnostics, not a substitute for listening.

Reproduce the 1, 10, and 60 second envelope benchmark with:

```bash
python scripts/benchmark_speech_envelopes.py --output /tmp/speech-envelope-benchmark.json
```

The script runs static rate, static pitch, static combined, variable-rate, variable-pitch, and combined-variable cases on a deterministic 16 kHz harmonic fixture. Constant-rate baselines are chosen to match the variable-rate output frame count. It records runtime, real-time factor, expected/output frames, length error, finite status, environment, and variable/static runtime ratio. A ratio above `1.0` means the envelope case took longer. This is a single sequential run without warm-up, so treat timings as indicative rather than a performance guarantee.

Recorded on Python 3.13.14, NumPy 2.5.3, AudioSig working-tree metadata 0.1.4, Linux 6.18.33.2 under WSL2, Intel Core Ultra 7 355, 8 logical CPUs. All 18 outputs were finite and had zero frame error. Relative elapsed-time ratios were:

| Input | Variable rate / static rate | Variable pitch / static pitch | Combined variable / static combined |
| ----: | --------------------------: | ----------------------------: | ----------------------------------: |
|   1 s |                       1.277 |                         1.054 |                               0.948 |
|  10 s |                       1.205 |                         1.133 |                               0.929 |
|  60 s |                       0.950 |                         0.866 |                               0.968 |

The fixture is synthetic and fully voiced. These timings do not establish perceptual quality or predict every speech input.

The feature is available since AudioSig 0.1.5. After redoing all three A/B comparisons on a permitted user-created PCM WAV, the user could not hear a clear difference and considered the static and envelope renders the same. This is one listener's subjective result for this source and these settings, not a general claim of equivalence. Findings are in the [listening report](envelope-listening-evaluation.md). The source and rendered files remain local and are not part of release materials.

Variable-rate-only runs use the absolute integrated rate map with WSOLA. Any variable pitch uses TD-PSOLA on voiced regions and mapped WSOLA for unvoiced fallback; the speech path is not formant-preserving and larger shifts can sound more artifact-prone. Results are deterministic within an AudioSig version but are not promised to be bit-identical across versions.

The listening report records the corrected result and can capture further listeners or samples. This listening step is complete for the current sample; overall release acceptance still depends on remaining technical validation.

## Listening protocol

For the TD-PSOLA release check, use the corpus categories, randomized A/B or
ABX matrix, listener fields, and promotion gates in
[`docs/td-psola-listening-evaluation-2026-07-31.md`](td-psola-listening-evaluation-2026-07-31.md).
No default change is justified by synthetic metrics alone.

Native pitch shifting changes the spectral envelope along with F0 and therefore
does not currently preserve vocal formants. Larger shifts are expected to be
more artifact-prone; this harness records metrics but does not turn them into a
claim of studio-grade perceptual quality.
