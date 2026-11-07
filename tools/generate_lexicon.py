"""Generate a corpus-specific MFA dictionary and frozen FastSpeech 2 phone IDs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FastSpeech2"))


def _read_words(manifests: list[Path], language: str) -> list[str]:
    words: set[str] = set()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row["text"])
                if language == "en":
                    from text.cleaners import english_cleaners

                    text = english_cleaners(text)
                    words.update(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))
                elif language == "te":
                    text = unicodedata.normalize("NFC", text)
                    # IndicTTS transcripts are whitespace-delimited. Strip only
                    # punctuation/symbol edges; preserve internal Telugu codepoints.
                    for word in text.split():
                        word = word.strip("\t\r\n,.;:!?()[]{}\"'“”‘’")
                        if word:
                            words.add(word)
                else:
                    raise ValueError(f"Unsupported language code: {language}")
    if not words:
        raise ValueError("No words found in the split manifests")
    return sorted(words)


def _english_lexicon(words: list[str]) -> dict[str, list[str]]:
    from g2p_en import G2p

    g2p = G2p()
    result: dict[str, list[str]] = {}
    for word in words:
        phones = [phone for phone in g2p(word) if phone and phone.strip()]
        if not phones:
            raise ValueError(f"G2P produced no phones for English word {word!r}")
        result[word] = phones
    return result


def _telugu_lexicon(words: list[str]) -> dict[str, list[str]]:
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    backend = EspeakBackend("te")
    if not backend.is_supported_language("te"):
        raise RuntimeError("The installed eSpeak NG backend does not expose Telugu ('te')")
    separator = Separator(phone=" ", word=None, syllable=None)
    result: dict[str, list[str]] = {}
    for word in words:
        phonetic = backend.phonemize([word], separator=separator, strip=True)[0]
        phones = phonetic.split()
        if not phones:
            raise ValueError(f"eSpeak NG produced no phones for Telugu word {word!r}")
        result[word] = phones
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    language = config["language"]
    dataset = config["dataset"]
    manifests = [
        (base / dataset["train_manifest"]).resolve(),
        (base / dataset["validation_manifest"]).resolve(),
    ]
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}")

    words = _read_words(manifests, language)
    if language == "en":
        lexicon = _english_lexicon(words)
    elif language == "te":
        lexicon = _telugu_lexicon(words)
    else:
        raise ValueError(f"Unsupported language code in config: {language}")

    dictionary = (base / config["text"]["dictionary"]).resolve()
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    with dictionary.open("w", encoding="utf-8", newline="\n") as stream:
        for word in sorted(lexicon):
            stream.write(word + "\t" + " ".join(lexicon[word]) + "\n")

    phone_inventory = sorted(
        {phone for phones in lexicon.values() for phone in phones}
        | {"sil", "sp", "spn"}
    )
    preprocessed = (base / config["paths"]["preprocessed"]).resolve()
    preprocessed.mkdir(parents=True, exist_ok=True)
    (preprocessed / "phone_symbols.json").write_text(
        json.dumps(phone_inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "language": language,
        "word_count": len(words),
        "phone_count": len(phone_inventory),
        "dictionary": str(dictionary),
        "phone_symbols": str(preprocessed / "phone_symbols.json"),
        "pronunciation_source": "g2p-en" if language == "en" else "eSpeak NG via phonemizer",
        "human_pronunciation_reviewed": False,
    }
    (preprocessed / "lexicon_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
