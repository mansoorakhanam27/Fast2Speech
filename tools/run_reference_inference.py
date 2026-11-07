"""Run the reference synthesizer and save audio plus actual variance predictions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from tools.run_reference_training import build_reference_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restore-step", type=int, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default=None, help="Speaker label (needed for a multi-speaker model)")
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    reference_root = project_root / "FastSpeech2"
    preprocess, model, train = build_reference_configs(config_path)
    preprocessed = Path(preprocess["path"]["preprocessed_path"])
    required = [
        preprocessed / "stats.json",
        preprocessed / "phone_symbols.json",
        Path(train["path"]["ckpt_path"]) / f"{args.restore_step}.pth.tar",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained inference artifacts:\n  " + "\n  ".join(missing))

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    speaker_id = 0
    if model["multi_speaker"]:
        labels_path = preprocessed / "speaker_labels.json"
        speakers_path = preprocessed / "speakers.json"
        if not labels_path.is_file() or not speakers_path.is_file():
            raise FileNotFoundError("Multi-speaker inference requires speaker_labels.json and speakers.json")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        speaker_ids = json.loads(speakers_path.read_text(encoding="utf-8"))
        if args.speaker is None:
            speaker_id = min(speaker_ids.values())
            print("No speaker selected; using first speaker ID {}.".format(speaker_id))
        else:
            matches = [
                folder
                for folder, label in labels.items()
                if label.casefold() == args.speaker.casefold()
                or folder.casefold() == args.speaker.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Speaker {!r} did not uniquely match available labels: {}".format(
                        args.speaker, sorted(labels.values())
                    )
                )
            speaker_id = int(speaker_ids[matches[0]])
    results = (config_path.parent / raw_config["paths"]["results"]).resolve()
    audio_dir = results / "test_generated_audios"
    analysis_dir = results / "variance_adaptor_plots"
    audio_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    train["path"]["result_path"] = str(audio_dir)
    runtime_dir = preprocessed / "runtime_config"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_files = {
        "preprocess.yaml": preprocess,
        "model.yaml": model,
        "train.yaml": train,
    }
    for name, value in runtime_files.items():
        (runtime_dir / name).write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    command = [
        sys.executable,
        str(reference_root / "synthesize.py"),
        "--restore_step",
        str(args.restore_step),
        "--mode",
        "single",
        "--text",
        args.text,
        "--speaker_id",
        str(speaker_id),
        "--case_id",
        args.case_id,
        "--analysis_output",
        str(analysis_dir),
        "--preprocess_config",
        str(runtime_dir / "preprocess.yaml"),
        "--model_config",
        str(runtime_dir / "model.yaml"),
        "--train_config",
        str(runtime_dir / "train.yaml"),
        "--pitch_control",
        str(args.pitch_control),
        "--energy_control",
        str(args.energy_control),
        "--duration_control",
        str(args.duration_control),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["FASTSPEECH2_EXTRA_PHONES_FILE"] = str(preprocessed / "phone_symbols.json")
    subprocess.run(command, cwd=reference_root, env=env, check=True)


if __name__ == "__main__":
    main()
