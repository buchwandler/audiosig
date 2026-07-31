# ESOLA listening evaluation — 2026-07-31

## Status

The repository contains no redistribution-safe or local real-speech WAV corpus
and no listener responses. Consequently, this report records the required
evaluation protocol and an honest `not run` result; it does not claim that
ESOLA is ready to replace WSOLA or become the default speech backend.

## Planned utterance coverage

The next evaluation should use at least one utterance in each category below,
with source license, attribution, sample rate, and filename recorded before
rendering:

|   # | Category                     | Listener result              |
| --: | ---------------------------- | ---------------------------- |
|   1 | lower-pitched voice          | not run — corpus unavailable |
|   2 | higher-pitched voice         | not run — corpus unavailable |
|   3 | breathy voice                | not run — corpus unavailable |
|   4 | rapid F0 motion              | not run — corpus unavailable |
|   5 | plosive-rich text            | not run — corpus unavailable |
|   6 | fricative-rich text          | not run — corpus unavailable |
|   7 | mixed voiced/unvoiced speech | not run — corpus unavailable |
|   8 | short TTS fragment           | not run — corpus unavailable |
|   9 | silence boundaries           | not run — corpus unavailable |
|  10 | mild background noise        | not run — corpus unavailable |

## Required matrix and protocol

Render rates `0.75, 0.8, 1.2, 1.25, 1.5, 2.0` with WSOLA, ESOLA, and
phase-vocoder diagnostic references. Include downstream pitch/rate combinations
whose computed ESOLA backend rate remains in `0.5 <= rate <= 2.0`. Randomize
A/B or ABX order, use at least two listeners, and record intelligibility,
naturalness, robotic/metallic character, pitch stability, transient duplication,
buzziness, flutter, clicks, and overall preference.

## Promotion decision

No listener ratings are available, so the 65% moderate-rate ESOLA preference
gate is `not evaluated`. Defaults remain unchanged: phase vocoder for
`time_stretch` and `pitch_shift`, and WSOLA for `apply_speech_effects`.
