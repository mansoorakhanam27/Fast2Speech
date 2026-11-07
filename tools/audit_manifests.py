"""Audit speech manifests and create deterministic, duration-targeted splits.

This is dataset bookkeeping only. It deliberately does not normalize transcripts,
resample audio, or claim that forced alignment has succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio inspection requires soundfile. Install project requirements first."
        ) from exc
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid audio dimensions: {path}")
    return info.frames / info.samplerate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            records.append(row)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _stable_key(row: dict[str, Any], seed: int) -> str:
    material = f"{seed}\0{row['id']}\0{row['audio_filepath']}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_records(manifest: Path, root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest)
    seen: set[str] = set()
    seen_audio: dict[str, str] = {}
    for idx, row in enumerate(rows, start=1):
        if "id" not in row or not str(row["id"]).strip():
            raise ValueError(f"Row {idx} has no non-empty id")
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate utterance id: {row['id']}")
        seen.add(row["id"])
        if "audio_filepath" not in row or not str(row["audio_filepath"]).strip():
            raise ValueError(f"Row {idx} ({row['id']}) has no audio_filepath")
        if "text" not in row or not str(row["text"]).strip():
            raise ValueError(f"Row {idx} ({row['id']}) has no transcript")
        audio_path = Path(str(row["audio_filepath"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        audio_hash = _file_sha256(audio_path)
        if audio_hash in seen_audio:
            raise ValueError(
                f"Duplicate audio content in rows {seen_audio[audio_hash]!r} and {row['id']!r}"
            )
        seen_audio[audio_hash] = row["id"]
        duration = _load_audio_duration(audio_path)
        declared_duration = row.get("duration")
        if declared_duration is not None:
            declared_duration = float(declared_duration)
            if not math.isfinite(declared_duration) or declared_duration <= 0:
                raise ValueError(f"Invalid declared duration for {row['id']}")
            if abs(declared_duration - duration) > max(0.25, duration * 0.02):
                raise ValueError(
                    f"Duration mismatch for {row['id']}: manifest={declared_duration:.3f}s, "
                    f"audio={duration:.3f}s"
                )
        row["duration"] = duration
        row["audio_filepath"] = str(audio_path)
        row["speaker"] = str(row.get("speaker", "unknown"))
    return rows


def _choose_duration_target(
    rows: list[dict[str, Any]], target_seconds: float, seed: int
) -> list[dict[str, Any]]:
    """Choose a stable pseudo-random subset, stopping at the nearest duration."""
    ordered = sorted(rows, key=lambda row: _stable_key(row, seed))
    chosen: list[dict[str, Any]] = []
    total = 0.0
    for row in ordered:
        duration = float(row["duration"])
        if total >= target_seconds:
            break
        # Prefer the utterance that leaves the smaller absolute target error.
        if abs(total + duration - target_seconds) < abs(total - target_seconds):
            chosen.append(row)
            total += duration
    return chosen


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_and_split(
    manifest: Path,
    audio_root: Path,
    output_dir: Path,
    language: str,
    train_hours: float,
    validation_hours: float,
    tolerance_minutes: float,
    seed: int,
) -> dict[str, Any]:
    rows = _prepare_records(manifest, audio_root)
    train_seconds = train_hours * 3600
    val_seconds = validation_hours * 3600

    # Respect caller-provided splits when present. Otherwise perform a deterministic
    # split from the full corpus. Existing split names are never silently ignored.
    declared = {str(row.get("split", "")).lower() for row in rows if row.get("split")}
    if declared:
        if not declared <= {"train", "validation", "valid", "val", "test"}:
            raise ValueError(f"Unrecognized split labels in manifest: {sorted(declared)}")
        has_train = "train" in declared
        has_validation = bool(declared.intersection({"validation", "valid", "val"}))
        if has_train and has_validation:
            train_pool = [
                row for row in rows if str(row.get("split", "")).lower() == "train"
            ]
            val_pool = [
                row
                for row in rows
                if str(row.get("split", "")).lower() in {"validation", "valid", "val"}
            ]
            # A separately labeled source test set is never used for model training.
        elif declared == {"test"}:
            # Some hubs publish the entire benchmark as one "test" split. The
            # assignment requires a new train/validation partition from that pool.
            train_pool = rows
            val_pool = rows
        else:
            raise ValueError(
                "Manifest has an incomplete preassigned split set {}. Provide both "
                "train and validation labels, or remove split labels to create a "
                "seeded assignment split.".format(sorted(declared))
            )
    else:
        ordered = sorted(rows, key=lambda row: _stable_key(row, seed))
        val_pool = ordered
        train_pool = ordered

    train = _choose_duration_target(train_pool, train_seconds, seed)
    used_train_ids = {row["id"] for row in train}
    val_candidates = [row for row in val_pool if row["id"] not in used_train_ids]
    val = _choose_duration_target(val_candidates, val_seconds, seed + 1)

    train_duration = sum(row["duration"] for row in train)
    val_duration = sum(row["duration"] for row in val)
    overlap = used_train_ids.intersection(row["id"] for row in val)
    if overlap:
        raise AssertionError(f"Train/validation overlap: {sorted(overlap)[:5]}")

    tolerance = tolerance_minutes * 60
    if abs(train_duration - train_seconds) > tolerance:
        raise ValueError(
            f"Could not meet training target: got {train_duration / 3600:.3f} h, "
            f"target {train_hours:.3f} h, tolerance {tolerance_minutes:g} min. "
            "Check dataset duration and manifest splits."
        )
    if abs(val_duration - val_seconds) > tolerance:
        raise ValueError(
            f"Could not meet validation target: got {val_duration / 3600:.3f} h, "
            f"target {validation_hours:.3f} h, tolerance {tolerance_minutes:g} min. "
            "Check dataset duration and manifest splits."
        )

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", val)
    report = {
        "language": language,
        "source_manifest": str(manifest.resolve()),
        "seed": seed,
        "total_rows_audited": len(rows),
        "total_audio_hours_audited": sum(row["duration"] for row in rows) / 3600,
        "train_rows": len(train),
        "train_audio_hours": train_duration / 3600,
        "validation_rows": len(val),
        "validation_audio_hours": val_duration / 3600,
        "train_validation_id_overlap": len(overlap),
        "speakers_in_train": sorted({row["speaker"] for row in train}),
        "speakers_in_validation": sorted({row["speaker"] for row in val}),
        "transcripts_modified": False,
        "note": "Durations were measured from audio files; alignment and transcript correctness are not verified by this script.",
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--train-hours", type=float, default=5.0)
    parser.add_argument("--validation-hours", type=float, default=1.0)
    parser.add_argument("--tolerance-minutes", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = audit_and_split(
        args.manifest.resolve(), args.audio_root.resolve(), args.output_dir.resolve(),
        args.language, args.train_hours, args.validation_hours,
        args.tolerance_minutes, args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
