# English and Telugu FastSpeech 2

This repository is the working implementation for the assignment. It is based
on the MIT-licensed PyTorch reference implementation
[`ming024/FastSpeech2`](https://github.com/ming024/FastSpeech2), fetched at
commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`. The upstream source is
preserved in `FastSpeech2/`; language-specific adaptation files live under
`English/` and `Telugu/`.

Public repository: https://github.com/mansoorakhanam27/Fast2Speech

## Current state

- Reference source is vendored.
- Language configuration drafts, export tooling, and a deterministic
  manifest/duration audit utility are present.
- Svarah access was granted to the signed-in Hugging Face account. The export
  contains 6,656 WAV utterances (9.6144 h), all with speaker labels, from
  revision `ebbf7777fe771490696a3f7b007097606fa8c924` of
  `ai4bharat/Svarah`. Its dataset card identifies the license as CC BY 4.0.
- The reproducible seed-42 English split has 3,429 training utterances
  (5.0007 h) and 686 validation utterances (1.0001 h), with zero ID overlap.
  Both Female and Male labels occur in both splits. See
  `data/Svarah/export_summary.json` and `data/Svarah/splits/audit.json`.
- The Telugu corpus has not been downloaded. The Hugging Face page named
  IndicTTS Telugu has conflicting language information (including a note that
  its mirror contains Hindi); do not train on it until the language is verified.
  The original IIT Madras IndicTTS license and request process also remain to be
  reviewed and completed.
- A technical progress report is available at `Report/Technical_Report.pdf`.
  Forced alignment, model training, inference, loss/variance plots, and the final
  technical report have not been completed. No model checkpoints or experimental
  findings are claimed.

## Dataset preparation

Keep audio and any license-restricted files outside version control, under
`data/Svarah/` and `data/IndicTTS_Telugu/`. Create a UTF-8 JSON Lines manifest
for each corpus, one JSON object per utterance:

```json
{"id":"utt_0001","audio_filepath":"audio/utt_0001.wav","text":"Transcript exactly as supplied","speaker":"speaker_1"}
```

Paths may be absolute or relative to the `--audio-root`. The audit measures WAV
duration from the audio, checks declared durations when present, rejects
duplicate IDs and missing audio/transcripts, and selects deterministic
duration-targeted train and validation sets. It never normalizes or edits text.

From the repository root in PowerShell, after installing the requirements and
preparing the manifests:

```powershell
python tools\audit_manifests.py `
  --manifest data\Svarah\manifest.jsonl `
  --audio-root data\Svarah `
  --output-dir data\Svarah\splits `
  --language English

python tools\audit_manifests.py `
  --manifest data\IndicTTS_Telugu\manifest.jsonl `
  --audio-root data\IndicTTS_Telugu `
  --output-dir data\IndicTTS_Telugu\splits `
  --language Telugu
```

Both commands require about five hours for training and one hour for validation,
each within a five-minute tolerance. If the measured usable corpus cannot meet
those targets, the audit stops instead of silently changing the requirement.
Inspect `audit.json` and the emitted split manifests before forced alignment.

### Dataset notes

Svarah is a multi-speaker Indian-accented English corpus with read and
spontaneous speech. The exported set's measured duration and speaker labels
are recorded above. Transcript correctness and forced alignment have not been
verified. Access to the Hugging Face files requires accepting the dataset
conditions; the dataset card identifies CC BY 4.0.

IndicTTS is distributed under the IIT Madras IndicTTS license terms. The
Hugging Face page titled IndicTTS Telugu reports approximately 8.74 hours but
also contains a note claiming that the mirror is Hindi. Verify Telugu language
labels, transcripts, speakers, and audio duration against the downloaded files
and preferably the original source before training. Do not redistribute raw
corpus audio in this repository.

## Model and language adaptation

The upstream model implements a Transformer encoder, duration/pitch/energy
variance adaptor, length regulator, mel decoder, postnet, and a separately
loaded HiFi-GAN vocoder. Its default text frontend includes English-oriented
symbols and Mandarin pinyin; it does not provide Telugu support. The two
configurations therefore specify independent phoneme frontends and alignment
resources. English can start from the reference English/G2P/MFA path. Telugu
requires a Telugu phone inventory, Unicode-safe normalization, a validated
Telugu pronunciation dictionary, and an acoustic alignment model. The
`Telugu/config.yaml` currently marks those resources as prerequisites; the
reference English lexicon must not be reused for Telugu.

The assignment's original test strings will be stored verbatim. Apparent
spacing/OCR problems in those strings must be clarified before correcting or
normalizing them. Training-text normalization and inference-time linguistic
normalization will be documented separately.

## Hardware and software

Use a dedicated Python 3.10 environment for compatibility with the reference
code and audio/alignment tools. Install PyTorch and torchaudio as a matching
pair using the official PyTorch selector. With the reported 8 GB of RAM and no
CUDA device detected in the current environment, use small batches for
preprocessing and expect CPU training to be very slow. A university or cloud
CUDA GPU is the practical route for completing both training runs. Record the
exact hardware, package versions, seeds, and configuration with the report.

## Required structure

```text
English/{train.py,preprocess.py,inference.py,config.yaml,checkpoints/}
Telugu/{train.py,preprocess.py,inference.py,config.yaml,checkpoints/}
Results/{English,Telugu}/
Report/Technical_Report.pdf
FastSpeech2/                 # upstream reference source
tools/audit_manifests.py     # shared, deterministic data audit/split utility
requirements.txt
README.md
```

Loss plots, generated test audio, variance-adaptor plots, and the final report
must be added only after actual training and inference. Large checkpoints should
use Git LFS or a stable external download link; do not commit data licensed for
restricted use.
