import re
import argparse
import json
import os
import unicodedata
from string import punctuation

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, sequence_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_telugu(text, preprocess_config):
    """Phonemize Telugu Unicode text with eSpeak NG and the frozen phone IDs."""
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from text.symbols import symbols

    backend = EspeakBackend("te")
    if not backend.is_supported_language("te"):
        raise RuntimeError("eSpeak NG does not provide the configured Telugu voice 'te'")
    phonetic = backend.phonemize(
        [text], separator=Separator(phone=" ", word=None, syllable=None), strip=True
    )[0]
    phones = phonetic.split()
    if not phones:
        raise ValueError("Telugu phonemizer returned no phones")
    symbol_set = set(symbols)
    missing = [phone for phone in phones if "@" + phone not in symbol_set]
    if missing:
        raise ValueError(
            "Telugu inference produced phones absent from the checkpoint vocabulary: "
            + repr(missing)
            + ". Rebuild preprocessing, phone_symbols.json, and the model together."
        )
    sequence = text_to_sequence("{" + " ".join(phones) + "}", [])
    if not sequence:
        raise ValueError("Telugu phone sequence mapped to an empty ID sequence")
    print("Raw Text Sequence (preserved): {}".format(text))
    print("Phone Sequence: {}".format(" ".join(phones)))
    return np.asarray(sequence)


def _save_variance_analysis(batch, predictions, preprocess_config, analysis_output, case_id):
    os.makedirs(analysis_output, exist_ok=True)
    for index in range(len(predictions[0])):
        source_id = str(batch[0][index])
        label = str(case_id or source_id)
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
        if not label:
            raise ValueError("Test case ID contains no filename-safe characters")
        src_len = int(predictions[8][index].item())
        phone_string = sequence_to_text(batch[3][index, :src_len].detach().cpu().tolist())
        phones = re.findall(r"\{([^}]+)\}", phone_string)
        if len(phones) != src_len:
            phones = phone_string.replace("{", " ").replace("}", " ").split()
        if len(phones) != src_len:
            phones = ["token_{}".format(i) for i in range(src_len)]

        duration_frames = predictions[5][index, :src_len].detach().cpu().numpy().astype(int)
        hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
        sample_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        duration_seconds = duration_frames * hop / sample_rate
        pitch_normalized = predictions[2][index, :src_len].detach().cpu().numpy()
        energy_normalized = predictions[3][index, :src_len].detach().cpu().numpy()

        stats_path = os.path.join(
            preprocess_config["path"]["preprocessed_path"], "stats.json"
        )
        with open(stats_path, "r", encoding="utf-8") as stream:
            stats = json.load(stream)
        pitch_mean, pitch_std = stats["pitch"][2:4]
        energy_mean, energy_std = stats["energy"][2:4]
        pitch_values = (
            pitch_normalized * pitch_std + pitch_mean
            if preprocess_config["preprocessing"]["pitch"]["normalization"]
            else pitch_normalized
        )
        energy_values = (
            energy_normalized * energy_std + energy_mean
            if preprocess_config["preprocessing"]["energy"]["normalization"]
            else energy_normalized
        )

        category_data = {
            "duration": (duration_seconds, "Predicted duration (seconds)", "#007C91"),
            "pitch": (pitch_values, "Predicted F0 (Hz)", "#A23B72"),
            "energy": (energy_values, "Predicted energy (linear scale)", "#DB7C26"),
        }
        for category, (values, ylabel, color) in category_data.items():
            output_dir = os.path.join(analysis_output, category)
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(
                figsize=(max(9, min(18, len(phones) * 0.24)), 4.8),
                constrained_layout=True,
            )
            ax.plot(
                np.arange(len(values)), values, marker="o", markersize=2.8,
                linewidth=1.35, color=color,
            )
            ax.set_xticks(np.arange(len(phones)))
            ax.set_xticklabels(phones, rotation=65, ha="right", fontsize=7)
            ax.set_xlabel("Aligned phone token")
            ax.set_ylabel(ylabel)
            ax.set_title("{}: {} prediction".format(label, category.title()))
            ax.grid(True, axis="y", alpha=0.25)
            fig.savefig(
                os.path.join(output_dir, label + ".png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

        record = {
            "case_id": label,
            "raw_text": str(batch[1][index]),
            "speaker_id": int(batch[2][index].item()),
            "phone_tokens": phones,
            "duration_frames": duration_frames.tolist(),
            "duration_seconds": duration_seconds.tolist(),
            "pitch_normalized": pitch_normalized.tolist(),
            "pitch_hz": pitch_values.tolist(),
            "energy_normalized": energy_normalized.tolist(),
            "energy": energy_values.tolist(),
            "sample_rate_hz": sample_rate,
            "hop_length_samples": hop,
            "f0_and_energy_are_model_predictions": True,
        }
        json_dir = os.path.join(analysis_output, "analysis_json")
        os.makedirs(json_dir, exist_ok=True)
        with open(os.path.join(json_dir, label + ".json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)


def synthesize(model, step, configs, vocoder, batchs, control_values, analysis_output=None, case_id=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            if analysis_output is not None:
                _save_variance_analysis(
                    batch, output, preprocess_config, analysis_output, case_id
                )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                save_mel_figures=analysis_output is None,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--analysis_output", type=str, default=None)
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "te":
            texts = np.array([preprocess_telugu(args.text, preprocess_config)])
        else:
            raise ValueError(
                "Unsupported inference language: {}".format(
                    preprocess_config["preprocessing"]["text"]["language"]
                )
            )
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(
        model,
        args.restore_step,
        configs,
        vocoder,
        batchs,
        control_values,
        analysis_output=args.analysis_output,
        case_id=args.case_id,
    )
