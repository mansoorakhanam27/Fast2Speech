"""Translate an assignment config and invoke the vendored reference trainer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml


def _resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def build_reference_configs(config_path: Path) -> tuple[dict[str, Any], ...]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    dataset = config["dataset"]
    text = config["text"]
    audio = config["audio"]
    variance = config["variance"]
    model = config["model"]
    training = config["training"]
    paths = config["paths"]

    preprocessed = _resolved(base, paths["preprocessed"])
    checkpoints = _resolved(base, paths["checkpoints"])
    results = _resolved(base, paths["results"])
    (preprocessed / "runtime_config").mkdir(parents=True, exist_ok=True)

    preprocess = {
        "dataset": dataset["name"],
        "path": {
            "corpus_path": str(_resolved(base, dataset["audio_root"])),
            "lexicon_path": str(_resolved(base, text["dictionary"])),
            "raw_path": str(preprocessed / "raw"),
            "preprocessed_path": str(preprocessed),
        },
        "preprocessing": {
            "val_size": 1,
            "text": {
                "text_cleaners": [text["cleaner"]],
                "language": config["language"],
            },
            "audio": {
                "sampling_rate": audio["sample_rate"],
                "max_wav_value": audio["max_wav_value"],
            },
            "stft": {
                "filter_length": audio["n_fft"],
                "hop_length": audio["hop_length"],
                "win_length": audio["win_length"],
            },
            "mel": {
                "n_mel_channels": audio["n_mels"],
                "mel_fmin": audio["mel_fmin"],
                "mel_fmax": audio["mel_fmax"],
            },
            "pitch": {
                "feature": variance["feature_level"],
                "normalization": variance["normalize_pitch"],
            },
            "energy": {
                "feature": variance["feature_level"],
                "normalization": variance["normalize_energy"],
            },
        },
    }
    model_config = {
        "transformer": {
            "encoder_layer": model["encoder_layers"],
            "encoder_head": model["encoder_heads"],
            "encoder_hidden": model["encoder_hidden"],
            "decoder_layer": model["decoder_layers"],
            "decoder_head": model["decoder_heads"],
            "decoder_hidden": model["decoder_hidden"],
            "conv_filter_size": model.get("conv_filter_size", 1024),
            "conv_kernel_size": model.get("conv_kernel_size", [9, 1]),
            "encoder_dropout": model["dropout"],
            "decoder_dropout": model["dropout"],
        },
        "variance_predictor": {
            "filter_size": model["variance_predictor_hidden"],
            "kernel_size": 3,
            "dropout": model.get("variance_predictor_dropout", 0.5),
        },
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": 256,
        },
        "multi_speaker": model["multi_speaker"],
        "max_seq_len": model.get("max_seq_len", 1000),
        "vocoder": {
            "model": model["vocoder"],
            "speaker": model.get("vocoder_checkpoint", "universal"),
        },
    }
    train = {
        "path": {
            "ckpt_path": str(checkpoints),
            "log_path": str(results / "logs"),
            "result_path": str(results / "training_previews"),
        },
        "optimizer": {
            "batch_size": training["batch_size"],
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0,
            "grad_clip_thresh": 1.0,
            "grad_acc_step": training["gradient_accumulation_steps"],
            "warm_up_step": training.get("warm_up_steps", 4000),
            "anneal_steps": [],
            "anneal_rate": 1.0,
        },
        "step": {
            "total_step": training["max_steps"],
            "log_step": training.get("log_interval", 100),
            "synth_step": training.get("preview_interval", 1000),
            "val_step": training["validation_interval"],
            "save_step": training["checkpoint_interval"],
        },
    }
    return preprocess, model_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restore-step", type=int, default=0)
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    reference_root = project_root / "FastSpeech2"
    if not (reference_root / "train.py").is_file():
        raise FileNotFoundError(f"Vendored FastSpeech 2 source not found: {reference_root}")
    preprocess, model, train = build_reference_configs(config_path)
    runtime_dir = Path(preprocess["path"]["preprocessed_path"]) / "runtime_config"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    required_preprocessed = [
        Path(preprocess["path"]["preprocessed_path"]) / name
        for name in (
            "train.txt", "val.txt", "speakers.json", "stats.json", "phone_symbols.json"
        )
    ]
    missing = [str(path) for path in required_preprocessed if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Preprocessing is not complete. Missing reference-training files:\n  "
            + "\n  ".join(missing)
        )

    vocoder_name = model["vocoder"]["speaker"]
    vocoder_file = reference_root / "hifigan" / f"generator_{vocoder_name}.pth.tar"
    vocoder_zip = Path(str(vocoder_file) + ".zip")
    if not vocoder_file.is_file():
        if not vocoder_zip.is_file():
            raise FileNotFoundError(
                f"Reference HiFi-GAN checkpoint not found: {vocoder_file} or {vocoder_zip}"
            )
        with zipfile.ZipFile(vocoder_zip) as archive:
            expected = vocoder_file.name
            if expected not in archive.namelist():
                raise ValueError(f"Unexpected checkpoint member in {vocoder_zip}")
            with archive.open(expected) as source, vocoder_file.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
    files = {
        "preprocess.yaml": preprocess,
        "model.yaml": model,
        "train.yaml": train,
    }
    for name, value in files.items():
        (runtime_dir / name).write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    (Path(train["path"]["ckpt_path"]).parent / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": str(config_path),
                "restore_step": args.restore_step,
                "python": sys.version,
                "platform": sys.platform,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(reference_root / "train.py"),
        "--restore_step",
        str(args.restore_step),
        "--preprocess_config",
        str(runtime_dir / "preprocess.yaml"),
        "--model_config",
        str(runtime_dir / "model.yaml"),
        "--train_config",
        str(runtime_dir / "train.yaml"),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["FASTSPEECH2_EXTRA_PHONES_FILE"] = str(
        Path(preprocess["path"]["preprocessed_path"]) / "phone_symbols.json"
    )
    completed = subprocess.run(command, cwd=reference_root, env=env, check=False)
    log_command = [
        sys.executable,
        str(project_root / "tools" / "plot_losses.py"),
        "--training-log",
        str(Path(train["path"]["log_path"]) / "train" / "log.txt"),
        "--validation-log",
        str(Path(train["path"]["log_path"]) / "val" / "log.txt"),
        "--output-dir",
        str(Path(train["path"]["log_path"]).parent),
        "--language",
        "English" if preprocess["preprocessing"]["text"]["language"] == "en" else "Telugu",
    ]
    plot_result = subprocess.run(log_command, cwd=project_root, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if plot_result.returncode != 0:
        raise SystemExit(plot_result.returncode)


if __name__ == "__main__":
    main()
