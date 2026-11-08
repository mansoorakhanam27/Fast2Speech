"""Prepare an audited corpus, align it, and extract FastSpeech 2 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FastSpeech2"))
sys.path.insert(0, str(ROOT))

from preprocessor.preprocessor import Preprocessor  # noqa: E402
from tools.audit_manifests import audit_and_split  # noqa: E402
from tools.run_reference_training import build_reference_configs  # noqa: E402


def _speaker_folder(speaker: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker).strip("._-") or "speaker"
    suffix = hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:48]}_{suffix}"


def _safe_id(utterance_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", utterance_id):
        raise ValueError(
            f"Unsafe utterance ID {utterance_id!r}; use 1-120 ASCII letters, digits, dot, underscore, or hyphen"
        )
    return utterance_id


def _normalize_transcript(text: str, cleaner_name: str) -> str:
    if cleaner_name == "english_cleaners":
        from text.cleaners import english_cleaners

        # Dataset exports sometimes use underscores as transcript separators.
        normalized = english_cleaners(text.replace("_", " "))
    elif cleaner_name == "telugu_cleaners":
        normalized = unicodedata.normalize("NFC", text)
        normalized = " ".join(normalized.split())
    else:
        raise ValueError(f"Unsupported cleaner configured: {cleaner_name}")
    if not normalized:
        raise ValueError("Transcript becomes empty after configured normalization")
    return normalized


def _resample_to_pcm16(source: Path, destination: Path, target_rate: int) -> None:
    audio, source_rate = sf.read(str(source), dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {source}")
    mono = audio.mean(axis=1)
    source_peak = float(np.max(np.abs(mono)))
    if source_peak > 1.0001:
        raise ValueError(f"Source audio samples exceed normalized PCM range: {source} (peak={source_peak})")
    if source_rate != target_rate:
        from math import gcd

        divisor = gcd(int(source_rate), int(target_rate))
        mono = resample_poly(
            mono, target_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    # Band-limited resampling can ring slightly beyond full scale even when
    # the source PCM is valid. PCM16 output is bounded, so clamp those peaks.
    mono = np.clip(mono, -1.0, 1.0)
    pcm16 = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(destination), target_rate, pcm16)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows


def _prepare(
    config_path: Path, config: dict[str, Any], reuse_audited_splits: bool = False
) -> tuple[Path, Path, Path]:
    base = config_path.parent
    dataset = config["dataset"]
    paths = config["paths"]
    train_manifest = (base / dataset["train_manifest"]).resolve()
    validation_manifest = (base / dataset["validation_manifest"]).resolve()
    if train_manifest.parent != validation_manifest.parent:
        raise ValueError("Train and validation manifests must share one splits directory")

    if reuse_audited_splits:
        audit_path = train_manifest.parent / "audit.json"
        if not audit_path.is_file() or not train_manifest.is_file() or not validation_manifest.is_file():
            raise FileNotFoundError("Cannot reuse splits: audit.json or split manifests are missing")
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        source_manifest = (base / dataset["manifest"]).resolve()
        if report.get("language") != config["language"] or report.get("seed") != int(dataset["split_seed"]):
            raise ValueError("Existing audit language/seed does not match the configuration")
        if Path(report.get("source_manifest", "")).resolve() != source_manifest:
            raise ValueError("Existing audit refers to a different source manifest")
        if source_manifest.stat().st_mtime > audit_path.stat().st_mtime:
            raise ValueError("Source manifest changed after its audit; rerun without --reuse-audited-splits")
        train_rows = _read_jsonl(train_manifest)
        validation_rows = _read_jsonl(validation_manifest)
        train_ids = {str(row["id"]) for row in train_rows}
        validation_ids = {str(row["id"]) for row in validation_rows}
        tolerance = float(dataset["duration_tolerance_minutes"]) / 60
        train_hours = sum(float(row["duration"]) for row in train_rows) / 3600
        validation_hours = sum(float(row["duration"]) for row in validation_rows) / 3600
        if train_ids & validation_ids:
            raise ValueError("Existing train and validation manifests overlap")
        if len(train_rows) != report.get("train_rows") or len(validation_rows) != report.get("validation_rows"):
            raise ValueError("Existing split row counts disagree with audit.json")
        if abs(train_hours - float(dataset["target_train_hours"])) > tolerance:
            raise ValueError("Existing training duration does not meet the configured target")
        if abs(validation_hours - float(dataset["target_validation_hours"])) > tolerance:
            raise ValueError("Existing validation duration does not meet the configured target")
        print(f"Reusing audited split: {len(train_rows)} train and {len(validation_rows)} validation rows")
    else:
        audit_and_split(
            (base / dataset["manifest"]).resolve(),
            (base / dataset["audio_root"]).resolve(),
            train_manifest.parent,
            config["language"],
            float(dataset["target_train_hours"]),
            float(dataset["target_validation_hours"]),
            float(dataset["duration_tolerance_minutes"]),
            int(dataset["split_seed"]),
        )
    split_rows = {
        "train": _read_jsonl(train_manifest),
        "validation": _read_jsonl(validation_manifest),
    }
    preprocessed = (base / paths["preprocessed"]).resolve()
    raw_root = preprocessed / "raw"
    train_corpus = preprocessed / "mfa_corpus_train"
    validation_corpus = preprocessed / "mfa_corpus_validation"
    target_rate = int(config["audio"]["sample_rate"])
    cleaner = config["text"]["cleaner"]
    all_manifest_rows: dict[str, Any] = {}
    speaker_labels: dict[str, str] = {}
    speaker_labels_path = preprocessed / "speaker_labels.json"
    previous_speaker_labels = (
        json.loads(speaker_labels_path.read_text(encoding="utf-8"))
        if speaker_labels_path.is_file()
        else {}
    )
    # A prior run may already have rewritten manifest labels to sanitized folder
    # names. Map one additional hashing layer back to its canonical source label.
    for folder, label in list(previous_speaker_labels.items()):
        previous_speaker_labels.setdefault(_speaker_folder(folder), label)
    used_ids: set[str] = set()

    for split, rows in split_rows.items():
        corpus_root = train_corpus if split == "train" else validation_corpus
        for row in rows:
            utterance_id = _safe_id(str(row["id"]))
            if utterance_id in used_ids:
                raise ValueError(f"Utterance ID occurs in multiple splits: {utterance_id}")
            used_ids.add(utterance_id)
            manifest_speaker = str(row["speaker"])
            speaker = previous_speaker_labels.get(manifest_speaker, manifest_speaker)
            folder = _speaker_folder(speaker)
            speaker_labels[folder] = speaker
            raw_dir = raw_root / folder
            wav_path = raw_dir / f"{utterance_id}.wav"
            lab_path = raw_dir / f"{utterance_id}.lab"
            source_path = Path(row["audio_filepath"])
            if not wav_path.exists():
                _resample_to_pcm16(source_path, wav_path, target_rate)
            elif sf.info(str(wav_path)).samplerate != target_rate:
                raise ValueError(f"Unexpected sample rate in prepared audio: {wav_path}")
            normalized_text = _normalize_transcript(str(row["text"]), cleaner)
            lab_path.write_text(normalized_text, encoding="utf-8")
            _link_or_copy(wav_path, corpus_root / folder / f"{utterance_id}.wav")
            _link_or_copy(lab_path, corpus_root / folder / f"{utterance_id}.lab")
            all_manifest_rows[utterance_id] = {
                **row,
                "speaker": speaker,
                "source_text": str(row["text"]),
                "normalized_text": normalized_text,
                "speaker_folder": folder,
                "split": split,
            }

    (preprocessed / "source_manifest_records.jsonl").write_text(
        "".join(
            json.dumps(all_manifest_rows[utterance_id], ensure_ascii=False) + "\n"
            for utterance_id in sorted(all_manifest_rows)
        ),
        encoding="utf-8",
    )
    speaker_labels_path.write_text(
        json.dumps(speaker_labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Replace speaker labels with safe folder names for the reference feature loader.
    for split, manifest in (("train", train_manifest), ("validation", validation_manifest)):
        rows = split_rows[split]
        manifest.write_text(
            "".join(
                json.dumps(
                    {
                        **row,
                        "speaker": _speaker_folder(
                            previous_speaker_labels.get(
                                str(row["speaker"]), str(row["speaker"])
                            )
                        ),
                        "split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    lexicon_command = [
        sys.executable,
        str(ROOT / "tools" / "generate_lexicon.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(lexicon_command, cwd=ROOT, check=True)
    return preprocessed, train_corpus, validation_corpus


def _run_mfa(
    config: dict[str, Any], preprocessed: Path, train_corpus: Path,
    validation_corpus: Path, run_validation: bool = True,
) -> None:
    mfa_python = os.environ.get("MFA_PYTHON")
    mfa_script = None
    if mfa_python:
        mfa_script = Path(mfa_python).resolve().parent / "Scripts" / "mfa-script.py"
        if not mfa_script.is_file():
            raise FileNotFoundError(f"MFA_PYTHON is set but its CLI script is missing: {mfa_script}")
    elif not shutil.which("mfa"):
        raise RuntimeError(
            "Montreal Forced Aligner (mfa) is not installed/on PATH. Install it in a dedicated "
            "conda environment, or set MFA_PYTHON to that environment's Python executable, then "
            "rerun this command. Prepared audio and lexicon are retained."
        )

    def mfa_command(*args: str) -> list[str]:
        if mfa_python and mfa_script:
            return [mfa_python, str(mfa_script), *args]
        return ["mfa", *args]
    dictionary = Path(config["text"]["dictionary"])
    if not dictionary.is_absolute():
        # The config file path is relative to the language folder; recover that
        # path from its stable path field in the original configuration below.
        raise AssertionError("Expected absolute dictionary path from resolved config")

    align_train = preprocessed / "mfa_aligned_train"
    align_validation = preprocessed / "mfa_aligned_validation"
    textgrid_root = preprocessed / "TextGrid"
    for path in (align_train, align_validation, textgrid_root):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Alignment output already exists: {path}. Move it aside before rerunning alignment."
            )
        path.mkdir(parents=True, exist_ok=True)

    if config["language"] == "en":
        acoustic_model = str(config["text"]["acoustic_model"])
        mfa_root = Path(
            os.environ.get("MFA_ROOT_DIR", str(Path.home() / "Documents" / "MFA"))
        ).expanduser()
        acoustic_archive = mfa_root / "pretrained_models" / "acoustic" / f"{acoustic_model}.zip"
        if acoustic_archive.is_file():
            print(f"Using cached MFA acoustic model: {acoustic_archive}")
        else:
            subprocess.run(
                mfa_command("model", "download", "acoustic", acoustic_model), check=True
            )
        if run_validation:
            subprocess.run(
                mfa_command("validate", str(train_corpus), str(dictionary)),
                check=True,
            )
        subprocess.run(
            mfa_command(
                "align", str(train_corpus), str(dictionary), acoustic_model,
                str(align_train), "--clean",
            ),
            check=True,
        )
        subprocess.run(
            mfa_command(
                "align", str(validation_corpus), str(dictionary), acoustic_model,
                str(align_validation), "--clean",
            ),
            check=True,
        )
    else:
        acoustic_model = Path(config["text"]["acoustic_model"])
        if not acoustic_model.is_absolute():
            raise AssertionError("Expected absolute Telugu acoustic-model path")
        acoustic_model.parent.mkdir(parents=True, exist_ok=True)
        if acoustic_model.exists():
            raise FileExistsError(
                f"Telugu MFA acoustic model already exists: {acoustic_model}. "
                "Refusing to silently reuse a model with unknown provenance."
            )
        if run_validation:
            subprocess.run(
                mfa_command("validate", str(train_corpus), str(dictionary)),
                check=True,
            )
        subprocess.run(
            mfa_command(
                "train", str(train_corpus), str(dictionary), str(acoustic_model),
                "--output_directory", str(align_train),
            ),
            check=True,
        )
        subprocess.run(
            mfa_command(
                "align", str(validation_corpus), str(dictionary),
                str(acoustic_model), str(align_validation),
            ),
            check=True,
        )

    # Keep a unified layout for the reference Preprocessor, without changing
    # the independent train/validation audio corpora used by MFA.
    for aligned_root in (align_train, align_validation):
        for textgrid in aligned_root.rglob("*.TextGrid"):
            relative = textgrid.relative_to(aligned_root)
            destination = textgrid_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(textgrid, destination)
    if not list(textgrid_root.rglob("*.TextGrid")):
        raise RuntimeError("MFA completed without producing any .TextGrid alignments")


def _build_features(config_path: Path, preprocessed: Path) -> None:
    preprocess_config, _, _ = build_reference_configs(config_path)
    # Resolve the manifests from the YAML instead of relying on the current directory.
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_manifest = (config_path.parent / raw_config["dataset"]["train_manifest"]).resolve()
    validation_manifest = (
        config_path.parent / raw_config["dataset"]["validation_manifest"]
    ).resolve()
    preprocessor = Preprocessor(preprocess_config)
    preprocessor.build_from_manifests(str(train_manifest), str(validation_manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "align", "features", "all"), default="all"
    )
    parser.add_argument(
        "--reuse-audited-splits", action="store_true",
        help="Reuse existing split manifests after checking their audit metadata and durations",
    )
    parser.add_argument(
        "--skip-mfa-validation", action="store_true",
        help="Skip MFA's corpus validation pass (which trains a temporary aligner); align directly",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.stage in {"prepare", "all"}:
        preprocessed, train_corpus, validation_corpus = _prepare(
            config_path, config, reuse_audited_splits=args.reuse_audited_splits
        )
    else:
        preprocessed = (config_path.parent / config["paths"]["preprocessed"]).resolve()
        train_corpus = preprocessed / "mfa_corpus_train"
        validation_corpus = preprocessed / "mfa_corpus_validation"
        for path in (train_corpus, validation_corpus):
            if not path.is_dir():
                raise FileNotFoundError(f"Prepared MFA corpus is missing: {path}")

    if args.stage in {"align", "all"}:
        # Resolve path-valued configuration entries against the language config.
        config["text"]["dictionary"] = str(
            (config_path.parent / config["text"]["dictionary"]).resolve()
        )
        config["text"]["acoustic_model"] = (
            str((config_path.parent / config["text"]["acoustic_model"]).resolve())
            if config["language"] == "te"
            else config["text"]["acoustic_model"]
        )
        _run_mfa(
            config, preprocessed, train_corpus, validation_corpus,
            run_validation=not args.skip_mfa_validation,
        )

    if args.stage in {"features", "all"}:
        _build_features(config_path, preprocessed)


if __name__ == "__main__":
    main()
