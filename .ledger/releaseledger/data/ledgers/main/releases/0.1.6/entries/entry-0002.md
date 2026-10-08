---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.1.6
kind: added
summary:
  Added NumPy-only spectral, pitch-tracking, deterministic reference-metric,
  STOI, and ESTOI APIs
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0022
paths:
  - audiosig/spectral.py
  - audiosig/pitch.py
  - audiosig/metrics.py
  - audiosig/intelligibility.py
  - audiosig/_pitch.py
  - tests/test_public_spectral.py
  - tests/test_public_pitch.py
  - tests/test_metrics.py
  - tests/test_intelligibility.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---
