"""Stream a selected Hugging Face speech dataset into local WAV + JSONL files.

Run only after personally reviewing and accepting the dataset access conditions.
The script uses the user's existing Hugging Face login and never uploads files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


DATASETS = {
    "svarah": ("ai4bharat/Svarah", "test", "audio_filepath", "English"),
    "indictts_telugu": (
        "SPRINGLab/IndicTTS_Telugu",
        "train",
        "audio",
        "Telugu",
    ),
}


def _audio_array(value: Any) -> tuple[np.ndarray, int, str]:
    if isinstance(value, dict):
        source_path = str(value.get("path") or "")
        rate = int(value.get("sampling_rate") or value.get("sample_rate") or 0)
        if value.get("array") is not None:
            samples = np.asarray(value["array"])
            if rate <= 0:
                raise ValueError("Decoded Hugging Face audio has no sampling_rate")
            return samples, rate, source_path
        if value.get("bytes"):
            samples, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            return np.asarray(samples), int(rate), source_path
        if source_path:
            samples, rate = sf.read(source_path, dtype="float32", always_2d=False)
            return np.asarray(samples), int(rate), source_path
    raise ValueError("Unsupported Hugging Face audio value; expected decoded array or bytes")


def _speaker_value(row: dict[str, Any], features: Any) -> tuple[str, bool]:
    for key in ("speaker_id", "speaker", "speaker_name"):
        if row.get(key) not in (None, ""):
            return str(row[key]), True
    if "gender" in row and row["gender"] not in (None, ""):
        value = row["gender"]
        feature = features.get("gender") if hasattr(features, "get") else None
        names = getattr(feature, "names", None)
        if isinstance(value, (int, np.integer)) and names and 0 <= int(value) < len(names):
            value = names[int(value)]
        return str(value), True
    return "unknown", False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=None, help="Pin a dataset commit SHA for reproducibility")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional diagnostic limit")
    args = parser.parse_args()

    try:
        from datasets import Audio, load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before exporting Hugging Face datasets") from exc

    repo_id, source_split, audio_column, language = DATASETS[args.dataset]
    api = HfApi()
    info = api.dataset_info(repo_id, revision=args.revision, token=True)
    revision = info.sha
    dataset = load_dataset(
        repo_id,
        split=source_split,
        revision=revision,
        streaming=True,
        token=True,
    )
    # Avoid an optional torchcodec dependency: read the encoded audio bytes and
    # decode them with SoundFile in _audio_array instead.
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    features = dataset.features
    if audio_column not in features or "text" not in features:
        raise ValueError(
            f"Unexpected dataset columns: {list(features)}; expected audio={audio_column!r} and text='text'"
        )

    root = args.output_dir.resolve()
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    durations = []
    speaker_labels = set()
    speaker_metadata_found = True
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(dataset, start=1):
            if args.max_rows is not None and index > args.max_rows:
                break
            transcript = row.get("text")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"Missing transcript in source row {index}")
            samples, sample_rate, source_path = _audio_array(row[audio_column])
            if sample_rate <= 0 or samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError(f"Invalid audio in source row {index}")
            duration = samples.shape[0] / sample_rate
            speaker, has_speaker = _speaker_value(row, features)
            speaker_metadata_found = speaker_metadata_found and has_speaker
            speaker_labels.add(speaker)

            identity = source_path or f"{repo_id}:{source_split}:{index}"
            utterance_id = "utt_{:06d}_{}".format(
                index, hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
            )
            destination = audio_dir / (utterance_id + ".wav")
            sf.write(str(destination), samples, sample_rate, subtype="PCM_16")
            row_out = {
                "id": utterance_id,
                "audio_filepath": str(destination.relative_to(root)),
                "duration": duration,
                "text": transcript,
                "speaker": speaker,
                "source_dataset": repo_id,
                "source_revision": revision,
                "source_split": source_split,
                "source_row": index - 1,
                "source_audio_path": source_path,
            }
            stream.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            durations.append(duration)
            if index % 250 == 0:
                print(f"Exported {index} utterances ({sum(durations) / 3600:.2f} h)")

    summary = {
        "dataset": args.dataset,
        "repository": repo_id,
        "revision": revision,
        "source_split": source_split,
        "language": language,
        "rows": len(durations),
        "audio_hours_measured_from_decoded_arrays": sum(durations) / 3600,
        "speaker_metadata_found_for_all_rows": speaker_metadata_found,
        "speaker_labels": sorted(speaker_labels),
        "manifest": str(manifest),
        "access_terms_accepted_by_user": "not recorded by this script",
        "audio_license": "must be documented from the source dataset card/license",
    }
    (root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
