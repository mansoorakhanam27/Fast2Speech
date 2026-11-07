""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Optional phones learned from the language-specific pronunciation dictionary.
# The reference vocabulary is fixed for English and Mandarin. FastSpeech 2 needs
# an embedding row for every aligned phone, so this opt-in extension lets each
# language experiment freeze and reuse its own inventory without changing the
# reference model architecture.
import json
import os

_extra_phones_path = os.environ.get("FASTSPEECH2_EXTRA_PHONES_FILE")
_extra_phones = []
if _extra_phones_path:
    with open(_extra_phones_path, "r", encoding="utf-8") as _stream:
        _extra_phones = json.load(_stream)
    if not isinstance(_extra_phones, list) or not all(
        isinstance(phone, str) and phone for phone in _extra_phones
    ):
        raise ValueError("FASTSPEECH2_EXTRA_PHONES_FILE must contain a JSON string list")

# Export all symbols. Keep ordering stable because checkpoint embedding rows
# depend on the exact symbol-to-ID mapping.
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
    + ["@" + phone for phone in _extra_phones]
)
symbols = list(dict.fromkeys(symbols))
