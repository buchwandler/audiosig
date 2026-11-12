"""Evaluate conservative terminal trimming against reviewed WAV boundary labels."""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiosig import trim_terminal_silence


@dataclass(frozen=True)
class BoundaryLabel:
    audio_path: str
    protected_end_sample: int


def _load_labels(path: Path) -> list[BoundaryLabel]:
    document = json.loads(path.read_text(encoding="utf-8"))
    items = document["items"]
    labels: list[BoundaryLabel] = []
    seen: set[str] = set()
    for item in items:
        audio_path = item["audio_path"]
        protected_end_sample = item["protected_end_sample"]
        if item.get("reviewed") is not True:
            raise ValueError(f"annotation for {audio_path} is not marked reviewed")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("audio_path must be a non-empty relative path")
        relative_path = Path(audio_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"audio_path must stay inside the audio directory: {audio_path}")
        if isinstance(protected_end_sample, bool) or not isinstance(protected_end_sample, int):
            raise ValueError(f"protected_end_sample must be an integer for {audio_path}")
        if audio_path in seen:
            raise ValueError(f"duplicate annotation for {audio_path}")
        seen.add(audio_path)
        labels.append(BoundaryLabel(audio_path, protected_end_sample))
    if not labels:
        raise ValueError("annotations must contain at least one reviewed item")
    return labels


def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError(f"expected mono audio: {path}")
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError(f"expected uncompressed 16-bit PCM audio: {path}")
        sample_rate = source.getframerate()
        sample_count = source.getnframes()
        pcm = source.readframes(sample_count)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size != sample_count:
        raise ValueError(f"incomplete PCM data: {path}")
    return samples, sample_rate


def evaluate(audio_dir: Path, labels: list[BoundaryLabel]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    negative_counts = {"0ms": 0, "10ms": 0, "25ms": 0, "50ms": 0}
    retained_tail_ms: list[float] = []

    for label in labels:
        samples, sample_rate = _read_pcm16_mono(audio_dir / label.audio_path)
        if not 0 <= label.protected_end_sample <= samples.size:
            raise ValueError(f"protected_end_sample is outside {label.audio_path}")
        _, analysis = trim_terminal_silence(samples, sample_rate=sample_rate)
        error_samples = analysis.trim_end_sample - label.protected_end_sample
        error_ms = error_samples * 1_000.0 / sample_rate
        for threshold_ms in (0, 10, 25, 50):
            if error_ms < -threshold_ms:
                negative_counts[f"{threshold_ms}ms"] += 1
        tail_ms = max(0.0, error_ms)
        retained_tail_ms.append(tail_ms)
        records.append(
            {
                "audio_path": label.audio_path,
                "sample_rate": sample_rate,
                "sample_count": samples.size,
                "protected_end_sample": label.protected_end_sample,
                "trim_end_sample": analysis.trim_end_sample,
                "end_error_samples": error_samples,
                "end_error_ms": error_ms,
                "retained_tail_ms": tail_ms,
                "confirmed": analysis.confirmed,
            }
        )

    tails = np.asarray(retained_tail_ms, dtype=np.float64)
    summary = {
        "item_count": len(records),
        "negative_end_error_counts": negative_counts,
        "mean_retained_tail_ms": float(np.mean(tails)),
        "p95_retained_tail_ms": float(np.percentile(tails, 95)),
    }
    return {"summary": summary, "items": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = _load_labels(args.annotations)
    report = evaluate(args.audio_dir, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
