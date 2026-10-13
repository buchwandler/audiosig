---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.1.8
kind: quality
summary:
  Improved DSP regression fixtures and coverage for compatibility, boundaries,
  speech levels, and public APIs
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - tests/fixtures/sopro_reference_golden.json
  - tests/test_output_boundaries.py
  - tests/test_output_conditioning.py
  - tests/test_pause_crop.py
  - tests/test_public_dsp.py
  - tests/test_public_spectral.py
  - tests/test_resampling_compatibility.py
  - tests/test_spectral_compatibility.py
  - tests/test_speech_level.py
issues: []
prs: []
sources:
  - git:757d69d6925e829b791034932d4fafb49cab1374
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
