#!/usr/bin/env python3
"""Render and measure local speech-effect comparisons without external tools."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np

from audiosig import apply_speech_effects, time_stretch

DEFAULT_RATES = (0.75, 0.85, 1.15, 1.30, 1.50)
DEFAULT_SEMITONES = (-5.0, -3.0, -2.0, 2.0, 3.0, 5.0)


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError("only uncompressed PCM WAV input is supported")
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 1:
        values = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        packed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        packed[packed & 0x800000 != 0] -= 1 << 24
        values = packed.astype(np.float64) / 8388608.0
    elif width == 4:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")
    if channels < 1 or values.size % channels:
        raise ValueError("invalid WAV channel/frame layout")
    reshaped = values.reshape(-1, channels).T
    return (reshaped[0] if channels == 1 else reshaped), sample_rate


def _write_pcm(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(audio, dtype=np.float64)
    channels = 1 if values.ndim == 1 else values.shape[0]
    samples = values.reshape(1, -1) if values.ndim == 1 else values
    interleaved = np.clip(samples.T.reshape(-1), -1.0, 1.0)
    encoded = np.round(interleaved * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.setcomptype("NONE", "not compressed")
        handle.writeframes(encoded)


def _metric(
    audio: np.ndarray,
    sample_rate: int,
    *,
    method: str,
    rate: float,
    semitones: float,
) -> dict[str, object]:
    values = np.asarray(audio, dtype=np.float64)
    samples = values.shape[-1]
    return {
        "method": method,
        "rate": rate,
        "semitones": semitones,
        "sample_rate": sample_rate,
        "samples": samples,
        "duration_seconds": samples / sample_rate,
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "finite": bool(np.isfinite(values).all()),
    }


def _number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def compare(
    input_path: Path,
    output_dir: Path,
    rates: tuple[float, ...],
    semitones: tuple[float, ...],
) -> list[dict[str, object]]:
    audio, sample_rate = _read_pcm(input_path)
    records: list[dict[str, object]] = []
    for rate in rates:
        pv = time_stretch(audio, rate, method="phase_vocoder")
        wsola = time_stretch(audio, rate, sample_rate=sample_rate, method="wsola")
        for method, rendered in (("phase_vocoder", pv), ("wsola", wsola)):
            filename = f"{method}_rate-{_number(rate)}.wav"
            _write_pcm(output_dir / filename, rendered, sample_rate)
            records.append(_metric(rendered, sample_rate, method=method, rate=rate, semitones=0.0))
        for pitch in semitones:
            rendered = apply_speech_effects(
                audio,
                sample_rate=sample_rate,
                rate=rate,
                semitones=pitch,
                method="wsola",
            )
            filename = f"combined_wsola_rate-{_number(rate)}_pitch-{_number(pitch)}st.wav"
            _write_pcm(output_dir / filename, rendered, sample_rate)
            records.append(
                _metric(
                    rendered,
                    sample_rate,
                    method="combined_wsola",
                    rate=rate,
                    semitones=pitch,
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source PCM WAV file")
    parser.add_argument("output_dir", type=Path, help="directory for rendered WAVs and metrics")
    parser.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES)
    parser.add_argument("--semitones", nargs="+", type=float, default=DEFAULT_SEMITONES)
    args = parser.parse_args()
    records = compare(args.input, args.output_dir, tuple(args.rates), tuple(args.semitones))
    json_path = args.output_dir / "metrics.json"
    csv_path = args.output_dir / "metrics.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} renders to {args.output_dir}")
    print(f"metrics: {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
