from __future__ import annotations

import csv
import importlib.metadata
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"


def _single_wheel() -> Path:
    wheels = sorted(DIST_DIR.glob("audiosig-*.whl"))
    if len(wheels) != 1:
        pytest.skip(f"expected one wheel in {DIST_DIR}, found {wheels}")
    return wheels[0]


def _single_sdist() -> Path:
    sdists = sorted(DIST_DIR.glob("audiosig-*.tar.gz"))
    if len(sdists) != 1:
        pytest.skip(f"expected one sdist in {DIST_DIR}, found {sdists}")
    return sdists[0]


def test_dynamic_versioning_is_configured() -> None:
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dynamic"] == ["version"]
    assert data["project"]["license"] == "Apache-2.0"
    scm = data["tool"]["setuptools_scm"]
    assert scm["version_file"] == "audiosig/_version.py"
    assert scm["write_to_source"] is True
    assert scm["fallback_version"] != "0.0.0"


def test_core_dependencies_are_portable() -> None:
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    assert any(dep.startswith("numpy") for dep in dependencies)
    forbidden = ("librosa", "scipy", "scikit-learn", "sklearn", "audiomentations", "numba", "torch")
    assert not any(dep.lower().startswith(prefix) for dep in dependencies for prefix in forbidden)


def test_built_wheel_is_universal_and_versioned() -> None:
    wheel = _single_wheel()

    assert wheel.name.endswith("-py3-none-any.whl")
    assert "0.0.0" not in wheel.name


def test_wheel_contains_typing_marker_and_required_modules() -> None:
    wheel = _single_wheel()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "audiosig/__init__.py",
        "audiosig/basic.py",
        "audiosig/py.typed",
        "audiosig/channels.py",
        "audiosig/generation.py",
        "audiosig/silence.py",
        "audiosig/effects.py",
        "audiosig/speech.py",
        "audiosig/_esola.py",
        "audiosig/_resampling.py",
        "audiosig/_spectral.py",
    }
    assert required <= names
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)


def test_source_distribution_contains_basic_module_and_typing_marker() -> None:
    sdist = _single_sdist()

    with tarfile.open(sdist, "r:gz") as archive:
        names = {member.name.split("/", 1)[-1] for member in archive.getmembers()}

    assert "audiosig/basic.py" in names
    assert "audiosig/py.typed" in names


def test_wheel_record_has_no_absolute_paths() -> None:
    wheel = _single_wheel()

    with zipfile.ZipFile(wheel) as archive:
        record_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        rows = csv.reader(archive.read(record_name).decode("utf-8").splitlines())
        paths = [row[0] for row in rows]

    assert paths
    assert all(not Path(path).is_absolute() for path in paths)


def test_installed_wheel_imports_without_source_tree(tmp_path: Path) -> None:
    wheel = _single_wheel()
    target = tmp_path / "site"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
    )

    code = """
import pathlib
import sys
site = str(pathlib.Path(sys.argv[1]))
sys.path.insert(0, site)
import audiosig
assert audiosig.__version__
assert callable(audiosig.trim)
assert callable(audiosig.resample)
assert callable(audiosig.resample_to_length)
assert callable(audiosig.resample_speed)
assert callable(audiosig.minmax_normalize)
assert callable(audiosig.apply_speech_effects)
assert callable(audiosig.downmix_to_mono)
assert callable(audiosig.generate_silence)
from audiosig.basic import downmix_to_mono, generate_silence
assert callable(downmix_to_mono)
assert callable(generate_silence)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(target)],
        cwd=tmp_path,
        check=True,
    )


def test_distribution_metadata_has_no_librosa_dependency(tmp_path: Path) -> None:
    wheel = _single_wheel()
    target = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
    )

    distributions = list(importlib.metadata.distributions(path=[str(target)]))
    distribution = next(item for item in distributions if item.metadata["Name"] == "audiosig")
    # Filter out optional extras - only core requirements are checked.
    requirements = [r for r in (distribution.requires or []) if "extra ==" not in r]

    assert all("librosa" not in r.lower() for r in requirements)
    assert all("scipy" not in r.lower() for r in requirements)
    assert all("numba" not in r.lower() for r in requirements)


def test_documentation_configuration_has_a_buildable_index() -> None:
    assert Path("docs/conf.py").is_file()
    assert Path("docs/index.rst").is_file()
    assert Path("docs/requirements.txt").is_file()


def test_release_files_exist() -> None:
    assert Path("audiosig/py.typed").is_file()
    assert Path("docs/index.rst").is_file()


def test_import_boundary() -> None:
    root = str(Path.cwd().resolve())
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
import audiosig
forbidden = {'librosa', 'scipy', 'sklearn', 'audiomentations', 'torch', 'numba'}
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, root], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == []
