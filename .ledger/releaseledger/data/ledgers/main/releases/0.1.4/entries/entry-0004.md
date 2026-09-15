---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0004
release_version: 0.1.4
kind: added
summary: Added BS.1770-style integrated loudness and dBFS/dBTP peak measurement APIs
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0020
  - git:092c00f2355a5b15b1aaafb6cebf419960ea287f
paths:
  - audiosig/loudness.py
  - audiosig/__init__.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

AudioSig now exports LoudnessMetrics, integrated_loudness, sample_peak_dbfs, true_peak_dbtp, and measure_loudness. The NumPy-only meter supports the mandatory 24 kHz mono path plus 44.1 and 48 kHz, with explicit silence, gating, and short-input behavior. PyKokoro handoff: AUDIOSIG_MIN_VERSION=0.1.3; INTEGRATED_LOUDNESS_IMPORT=audiosig.integrated_loudness; SAMPLE_PEAK_IMPORT=audiosig.sample_peak_dbfs; TRUE_PEAK_IMPORT=audiosig.true_peak_dbtp; MEASURE_LOUDNESS_IMPORT=audiosig.measure_loudness; SUPPORTED_SAMPLE_RATES=24000,44100,48000; 24KHZ_REFERENCE_VALIDATED=yes; KNOWN_LIMITATIONS=mono-only v1.
