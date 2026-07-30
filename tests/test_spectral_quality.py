"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from audiosig._spectral import istft, phase_vocoder, stft
from audiosig.exceptions import AudioShapeError, InvalidParameterError

pytestmark = pytest.mark.quality


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("n_fft", [256, 257, 512, 513])
@pytest.mark.parametrize("hop_length", [64, 128])
def test_stft_shape_and_finite_values(
    dtype: type[np.floating],
    center: bool,
    n_fft: int,
    hop_length: int,
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal(4096).astype(dtype)

    spectrum = stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
    )

    assert spectrum.ndim == 2
    assert spectrum.shape[-2] == n_fft // 2 + 1
    assert spectrum.shape[-1] >= 1
    assert np.all(np.isfinite(spectrum.real))
    assert np.all(np.isfinite(spectrum.imag))


@pytest.mark.parametrize("n_fft", [256, 257])
@pytest.mark.parametrize("center", [False, True])
def test_stft_matches_direct_numpy_calculation(
    n_fft: int,
    center: bool,
    rng: np.random.Generator,
) -> None:
    hop_length = 64
    source = rng.standard_normal(1500)

    actual = stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
    )

    reference = source
    if center:
        pad = n_fft // 2
        mode = "reflect" if source.size > 1 else "constant"
        reference = np.pad(reference, (pad, pad), mode=mode)
    if reference.size < n_fft:
        reference = np.pad(reference, (0, n_fft - reference.size))
    remainder = (reference.size - n_fft) % hop_length
    if remainder:
        reference = np.pad(reference, (0, hop_length - remainder))

    frame_count = 1 + (reference.size - n_fft) // hop_length
    expected = np.empty((n_fft // 2 + 1, frame_count), dtype=np.complex128)
    window = np.hanning(n_fft)
    for index in range(frame_count):
        start = index * hop_length
        expected[:, index] = np.fft.rfft(reference[start : start + n_fft] * window)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(np.float32, 2e-5), (np.float64, 1e-10)],
)
@pytest.mark.parametrize("n_fft", [256, 257, 512, 513])
@pytest.mark.parametrize("hop_length", [64, 128])
def test_stft_istft_reconstruction(
    dtype: type[np.floating],
    atol: float,
    n_fft: int,
    hop_length: int,
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal(5000).astype(dtype)

    spectrum = stft(source, n_fft=n_fft, hop_length=hop_length, center=True)
    reconstructed = istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        length=source.size,
        center=True,
        dtype=np.dtype(dtype),
    )

    assert reconstructed.dtype == np.dtype(dtype)
    assert reconstructed.shape == source.shape
    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(reconstructed, source, atol=atol, rtol=atol)


@pytest.mark.parametrize("n_fft", [128, 129])
def test_stft_istft_batch_independence(
    n_fft: int,
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal((2, 1024))
    source[1] *= 0.25
    hop_length = 32

    combined_spectrum = stft(source, n_fft=n_fft, hop_length=hop_length)
    combined = istft(
        combined_spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        length=source.shape[-1],
    )

    for channel in range(source.shape[0]):
        single_spectrum = stft(
            source[channel],
            n_fft=n_fft,
            hop_length=hop_length,
        )
        single = istft(
            single_spectrum,
            n_fft=n_fft,
            hop_length=hop_length,
            length=source.shape[-1],
        )
        np.testing.assert_allclose(combined[channel], single, atol=1e-10)

    assert not np.allclose(combined[0], combined[1])


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 3.0])
def test_phase_vocoder_output_shape_and_finite_values(
    rate: float,
    rng: np.random.Generator,
) -> None:
    n_fft = 512
    hop_length = 128
    spectrum = rng.standard_normal((n_fft // 2 + 1, 12)) + 1j * rng.standard_normal(
        (n_fft // 2 + 1, 12)
    )

    stretched = phase_vocoder(
        spectrum,
        rate=rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    assert stretched.shape == (
        spectrum.shape[0],
        int(np.ceil(spectrum.shape[1] / rate)),
    )
    assert np.all(np.isfinite(stretched.real))
    assert np.all(np.isfinite(stretched.imag))


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0])
def test_phase_vocoder_magnitudes_at_exact_source_frames(
    rate: float,
    rng: np.random.Generator,
) -> None:
    n_fft = 256
    hop_length = 64
    spectrum = rng.standard_normal((n_fft // 2 + 1, 10)) + 1j * rng.standard_normal(
        (n_fft // 2 + 1, 10)
    )

    stretched = phase_vocoder(
        spectrum,
        rate=rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    output_times = np.arange(stretched.shape[-1], dtype=np.float64) * rate
    exact = np.isclose(output_times, np.round(output_times))
    source_indices = np.round(output_times[exact]).astype(int)
    valid = source_indices < spectrum.shape[-1]

    np.testing.assert_allclose(
        np.abs(stretched[..., np.flatnonzero(exact)[valid]]),
        np.abs(spectrum[..., source_indices[valid]]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_phase_vocoder_batch_independence(rng: np.random.Generator) -> None:
    n_fft = 256
    hop_length = 64
    spectrum = rng.standard_normal((2, n_fft // 2 + 1, 8)) + 1j * rng.standard_normal(
        (2, n_fft // 2 + 1, 8)
    )

    combined = phase_vocoder(
        spectrum,
        rate=1.25,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    for channel in range(2):
        single = phase_vocoder(
            spectrum[channel],
            rate=1.25,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        np.testing.assert_allclose(combined[channel], single, atol=1e-12)


def test_spectral_primitive_validation() -> None:
    source = np.ones(128, dtype=np.float32)

    with pytest.raises(InvalidParameterError, match="n_fft"):
        stft(source, n_fft=1, hop_length=1)

    with pytest.raises(InvalidParameterError, match="hop_length"):
        stft(source, n_fft=64, hop_length=65)

    with pytest.raises(AudioShapeError, match="spectrum"):
        istft(
            np.ones((10, 3), dtype=np.complex64),
            n_fft=64,
            hop_length=16,
        )

    with pytest.raises(InvalidParameterError, match="rate"):
        phase_vocoder(
            np.ones((33, 3), dtype=np.complex64),
            rate=0.0,
            hop_length=16,
            n_fft=64,
        )
