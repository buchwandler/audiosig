# Speech quality input guidance

The automated suite uses deterministic synthetic source-filter-style speech so
it has no corpus licensing or network dependency. For listening evaluations,
place local PCM WAV files in a directory outside the package and pass each file
to `scripts/compare_speech_effects.py`.

Record the speaker/source license, attribution, sample rate, and filename in
the evaluation notes. Useful material includes lower- and higher-pitched
voices, plosive- and fricative-rich sentences, breathy speech, rapid F0 motion,
short TTS fragments, and silence/noise boundaries. Do not commit recordings
unless their redistribution license and repository size are acceptable.
