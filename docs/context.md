---
documentledger_schema: documentledger.context.v4
scan_version: 1
state_version: 2
mode: bootstrap
---

# Documentation update context

## Document inventory

- README.md
- docs/index.rst

## Unlinked source inventory

- audiosig/**init**.py
- audiosig/\_framing.py
- audiosig/\_resampling.py
- audiosig/\_spectral.py
- audiosig/\_validation.py
- audiosig/\_version.py
- audiosig/amplitude.py
- audiosig/effects.py
- audiosig/exceptions.py
- audiosig/silence.py
- tests/**init**.py
- tests/\_quality_helpers.py
- tests/conftest.py
- tests/test_amplitude.py
- tests/test_effects.py
- tests/test_effects_quality.py
- tests/test_framing.py
- tests/test_librosa_compat.py
- tests/test_packaging.py
- tests/test_resampling.py
- tests/test_resampling_quality.py
- tests/test_silence.py
- tests/test_spectral.py
- tests/test_spectral_quality.py
- tests/test_trim_split_quality.py
- tests/test_trim_vad_compat.py
- tests/test_vad_quality.py
- tests/test_validation.py

Create or update relevant docs, then add links with `docledger links add` or `docledger links add-section`.

## Unlinked changed sources

- None

## Validation commands

- None configured

## Agent rules

- Inspect affected or selected source units before editing docs.
- Rewrite only the selected sections unless broader consistency requires more.
- Do not invent behavior.
- Run the configured validation commands when they exist.
- Run `docledger mark-fresh --doc DOC --section SECTION --reason "Docs updated after scan version VERSION."` only after docs are updated and validated.
