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

It renders the basic phase-vocoder and WSOLA rate paths, then the combined
WSOLA pitch/rate planner for the documented rate and semitone matrices. Use
`--rates` and `--semitones` to narrow a run. The harness does not install,
invoke, or require Rubber Band, SoundTouch, WORLD, or another external backend.

## Listening protocol

For a practical release check, use at least ten utterances and six moderate
transform settings. Randomize A/B order, compare the old implementation with
the new WSOLA output, and record preference plus artifact notes. Include two or
more listeners when possible. The desired result is a clear majority preference
for the new speech backend at moderate rates without a severe recurring artifact
class. Keep source-license and attribution details with the evaluation notes.

Native pitch shifting changes the spectral envelope along with F0 and therefore
does not currently preserve vocal formants. Larger shifts are expected to be
more artifact-prone; this harness records metrics but does not turn them into a
claim of studio-grade perceptual quality.
