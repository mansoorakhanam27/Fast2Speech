import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def build_from_manifests(self, train_manifest, validation_manifest):
        """Build reference features from explicit, duration-audited split manifests.

        Unlike ``build_from_path``, this method never shuffles or repartitions the
        corpus. Pitch and energy normalization statistics are fitted on training
        utterances only and then applied to both splits.
        """
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        def read_manifest(path):
            with open(path, "r", encoding="utf-8-sig") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if not rows:
                raise ValueError("Empty split manifest: {}".format(path))
            return rows

        train_rows = read_manifest(train_manifest)
        validation_rows = read_manifest(validation_manifest)
        train_ids = {str(row["id"]) for row in train_rows}
        validation_ids = {str(row["id"]) for row in validation_rows}
        overlap = train_ids.intersection(validation_ids)
        if overlap:
            raise ValueError("Train/validation ID overlap: {}".format(sorted(overlap)[:5]))

        all_rows = [("train", row) for row in train_rows] + [
            ("validation", row) for row in validation_rows
        ]
        speakers = sorted({str(row["speaker"]) for _, row in all_rows})
        speaker_map = {speaker: i for i, speaker in enumerate(speakers)}
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        metadata = {"train": [], "validation": []}
        train_feature_files = {"pitch": [], "energy": []}
        validation_feature_files = {"pitch": [], "energy": []}
        total_frames = 0

        for split, row in tqdm(all_rows, desc="Extracting FastSpeech 2 features"):
            speaker = str(row["speaker"])
            basename = str(row["id"])
            expected = [
                os.path.join(self.in_dir, speaker, basename + ".wav"),
                os.path.join(self.in_dir, speaker, basename + ".lab"),
                os.path.join(self.out_dir, "TextGrid", speaker, basename + ".TextGrid"),
            ]
            missing = [path for path in expected if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    "Missing prepared/aligned input for {}: {}".format(basename, missing)
                )

            result = self.process_utterance(speaker, basename)
            if result is None:
                raise ValueError(
                    "Utterance {} was rejected during feature extraction; inspect its "
                    "audio and phone alignment instead of silently dropping it.".format(basename)
                )
            info, pitch_values, energy_values, n_frames = result
            metadata[split].append(info)
            total_frames += n_frames

            pitch_path = os.path.join(
                self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename)
            )
            energy_path = os.path.join(
                self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename)
            )
            target_files = (
                train_feature_files if split == "train" else validation_feature_files
            )
            target_files["pitch"].append(pitch_path)
            target_files["energy"].append(energy_path)

            if split == "train":
                if len(pitch_values):
                    pitch_scaler.partial_fit(np.asarray(pitch_values).reshape((-1, 1)))
                if len(energy_values):
                    energy_scaler.partial_fit(np.asarray(energy_values).reshape((-1, 1)))

        if not train_feature_files["pitch"] or not train_feature_files["energy"]:
            raise ValueError("No training pitch/energy features were extracted")
        if not hasattr(pitch_scaler, "mean_") or not hasattr(energy_scaler, "mean_"):
            raise ValueError("Training split contains no usable pitch/energy values")

        pitch_mean = float(pitch_scaler.mean_[0]) if self.pitch_normalization else 0.0
        pitch_std = float(pitch_scaler.scale_[0]) if self.pitch_normalization else 1.0
        energy_mean = float(energy_scaler.mean_[0]) if self.energy_normalization else 0.0
        energy_std = float(energy_scaler.scale_[0]) if self.energy_normalization else 1.0
        if pitch_std == 0 or energy_std == 0:
            raise ValueError("Zero variance in training pitch or energy features")

        def normalize_files(paths, mean, std):
            values_for_stats = []
            for path in paths:
                values = (np.load(path) - mean) / std
                np.save(path, values)
                values_for_stats.append(values)
            return values_for_stats

        train_pitch = normalize_files(train_feature_files["pitch"], pitch_mean, pitch_std)
        validation_pitch = normalize_files(
            validation_feature_files["pitch"], pitch_mean, pitch_std
        )
        train_energy = normalize_files(
            train_feature_files["energy"], energy_mean, energy_std
        )
        validation_energy = normalize_files(
            validation_feature_files["energy"], energy_mean, energy_std
        )

        def extrema(arrays):
            flattened = [np.asarray(values).reshape(-1) for values in arrays if len(values)]
            if not flattened:
                raise ValueError("Cannot compute feature range from empty arrays")
            joined = np.concatenate(flattened)
            return float(np.min(joined)), float(np.max(joined))

        pitch_min, pitch_max = extrema(train_pitch)
        energy_min, energy_max = extrema(train_energy)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "speakers.json"), "w", encoding="utf-8") as f:
            json.dump(speaker_map, f, ensure_ascii=False)
        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pitch": [pitch_min, pitch_max, pitch_mean, pitch_std],
                    "energy": [energy_min, energy_max, energy_mean, energy_std],
                },
                f,
            )
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata["train"]) + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata["validation"]) + "\n")

        print(
            "Feature extraction complete: {:.3f} audio hours across {} train and {} validation utterances".format(
                total_frames * self.hop_length / self.sampling_rate / 3600,
                len(metadata["train"]),
                len(metadata["validation"]),
            )
        )
        return metadata

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
