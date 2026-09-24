# Speech envelope listening evaluation

## Status

**Corrected user listening result received. After redoing all comparisons, the user could not hear a clear difference and considered the static and envelope renders the same.** This is a subjective result from one listener and one source, not a general claim of equivalence. The user-provided source and renders remain local and are not included in release materials. AudioSig 0.1.5 is the minimum version containing the envelope API.

## Prepare A/B renders

Use a PCM WAV that you have permission to process. Record its license or consent basis, sample rate, channel count, and source identifier. Do not commit private speech recordings or renders without permission.

For a focused comparison at rate 0.9 and pitch +2 semitones, run:

```bash
python scripts/compare_speech_effects.py INPUT.wav OUTPUT_DIR \
  --rates 0.9 --semitones 2 --envelopes
```

The relevant static and envelope pairs are:

| Effect   | Static render                              | Envelope render                            | Control points                                      |
| -------- | ------------------------------------------ | ------------------------------------------ | --------------------------------------------------- |
| Rate     | `wsola_rate-0p9.wav`                       | `envelope_rate-0p9.wav`                    | rate `1.0` at 0 s to `0.9` at 0.45 output s         |
| Pitch    | `td_psola_pitch-2st.wav`                   | `envelope_pitch-2st.wav`                   | pitch `0` at 0 s to `+2` semitones at 0.30 output s |
| Combined | `combined_td_psola_rate-0p9_pitch-2st.wav` | `envelope_combined_rate-0p9_pitch-2st.wav` | both curves above                                   |

`metrics.json` and `metrics.csv` include output length, exact-length error, finite status, runtime, and the numeric envelope points. The envelope rate and static rate renders are configured for different instantaneous trajectories, so judge their intended timing behavior as well as audible artifacts. Randomize or hide filenames during listening if practical.

## Listening checklist

Listen to the unmodified source and each pair at a comfortable, consistent playback level. Note clicks at transitions, pitch wobble, phasiness or warbling, metallic/buzzy quality, transient smearing, changes to breath or fricatives, timbre/formant shifts, rhythm discontinuities, and intelligibility. Include the timestamps and the affected output where possible.

## User-reported result

- Evaluation date: Not provided
- Listener: User
- Source: User-created `audio.wav`, 24 kHz, mono, 16-bit PCM, 60.288 seconds
- Render command: `python scripts/compare_speech_effects.py audio.wav /tmp/audiosig-envelope-ab --rates 0.9 --semitones 2 --envelopes`
- Render directory: `/tmp/audiosig-envelope-ab`
- Technical metrics: all 10 WAV renders were finite and had zero frame-count error.

| Comparison | User findings                                                                                           | Preference |
| ---------- | ------------------------------------------------------------------------------------------------------- | ---------- |
| Rate       | On repeat listening, no clear difference; the user considered the static and envelope renders the same. | Equal      |
| Pitch      | On repeat listening, no clear difference; the user considered the static and envelope renders the same. | Equal      |
| Combined   | On repeat listening, no clear difference; the user considered the static and envelope renders the same. | Equal      |

No numerical ratings or timestamps were provided. This corrected result supersedes the earlier notes after the user repeated all three A/B comparisons. It records the user's perception of this source and these settings only.

## Reusable evaluation template

Use this template for any additional listeners or sources. Record one result for every listener and comparison.

- Evaluation date:
- Listener identifier:
- Source identifier and permission basis:
- Sample rate and channels:
- Command and AudioSig version:
- Output directory or retained artifact identifiers:

| Comparison | Naturalness (1 to 5) | Intelligibility (1 to 5) | Timing/pitch accuracy (1 to 5) | Artifact severity (1 to 5) | Preference (static/envelope/equal) | Notes and timestamps |
| ---------- | -------------------: | -----------------------: | -----------------------------: | -------------------------: | ---------------------------------- | -------------------- |
| Rate       |                      |                          |                                |                            |                                    |                      |
| Pitch      |                      |                          |                                |                            |                                    |                      |
| Combined   |                      |                          |                                |                            |                                    |                      |

- Severe artifact heard (yes/no):
- Overall findings:
- Recommendation and rationale:

The corrected listening report is recorded. The user heard no clear difference across these three pairs and judged the envelope versions equivalent for this sample. This does not authorize a release, tag, push, or publication.
