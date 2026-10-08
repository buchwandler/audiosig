---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.7
kind: added
summary:
  Added NumPy-only spectral analysis, pitch tracking, reference metrics, and
  STOI/ESTOI APIs
status: accepted
audience: null
scopes: []
source_refs:
  - git:51411c8b350bfd38f33fd5e786f8b9384eff90b6
paths:
  - audiosig/_pitch.py
  - audiosig/intelligibility.py
  - audiosig/metrics.py
  - audiosig/pitch.py
  - audiosig/spectral.py
  - tests/test_intelligibility.py
  - tests/test_metrics.py
  - tests/test_public_pitch.py
  - tests/test_public_spectral.py
issues: []
prs: []
sources:
  - git:51411c8b350bfd38f33fd5e786f8b9384eff90b6
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
