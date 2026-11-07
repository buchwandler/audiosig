---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0001
release_version: 0.1.9
kind: changed
summary:
  Changed long-form loudness processing to use bounded chunks and added generic
  progress callbacks
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0024
  - git:6bc3abe3f04eaf747213a20b80a7f5e50f8e5fb0
paths:
  - audiosig/loudness.py
  - audiosig/__init__.py
  - tests/test_loudness.py
  - benchmarks/bench_loudness.py
  - README.md
  - docs/api-reference.md
  - docs/advanced.md
  - docs/getting-started.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Integrated loudness and standard 4x true peak processing use bounded chunks, and selective analysis avoids full-waveform float64 copies for unrequested metrics. Optional immutable LoudnessProgress callbacks report synchronous generic phase and frame progress without estimating an ETA.
