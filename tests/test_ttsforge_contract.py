from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.downstream


def test_ttsforge_import_and_basic_usage_contract() -> None:
    from audiosig import downmix_to_mono, generate_silence

    stereo = np.ones((240, 2), dtype=np.float32)
    mono = downmix_to_mono(stereo, channel_axis=1)
    pause = generate_silence(0.1, 24_000)

    assert mono.shape == (240,)
    assert pause.shape == (2_400,)
    assert mono.dtype == pause.dtype == np.float32


def test_ttsforge_import_resolves_from_source_package_without_optional_dependencies() -> None:
    root = str(Path(__file__).resolve().parents[1])
    code = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from audiosig import downmix_to_mono, generate_silence
import numpy as np
stereo = np.ones((10, 2), dtype=np.float32)
assert downmix_to_mono(stereo).shape == (10,)
assert generate_silence(0.01, 24_000).shape == (240,)
forbidden = {'librosa', 'scipy', 'sklearn', 'audiomentations', 'torch', 'numba'}
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, root],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
