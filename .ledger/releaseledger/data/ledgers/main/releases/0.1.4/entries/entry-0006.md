---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0006
release_version: 0.1.4
kind: quality
summary: Improved loudness and true-peak conformance coverage
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - tests/test_loudness.py
  - scripts/check_loudness_reference.py
issues: []
prs: []
sources:
  - tl:task-0020
contributors: []
breaking: false
internal: false
order: 3
---

The test suite covers 24 kHz, 44.1 kHz, 48 kHz, gating geometry, gain relationships, short inputs, invalid data, dtypes, axes, immutability, and inter-sample peak behavior. The reference script records 0.05 LU and 0.02 dB tolerances.
