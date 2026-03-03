from os.path import join, basename
from glob import glob
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from soundfile import read
from tqdm import tqdm
import pandas as pd
import librosa

from jiwer import wer
from pesq import pesq
from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std


def compute_audio_metrics(original, adversarial, purified, sr):
    """Compute all metrics comparing original/adversarial/purified signals."""
    original_16k = librosa.resample(original, orig_sr=sr, target_sr=16000) if sr != 16000 else original
    adversarial_16k = librosa.resample(adversarial, orig_sr=sr, target_sr=16000) if sr != 16000 else adversarial
    purified_16k = librosa.resample(purified, orig_sr=sr, target_sr=16000) if sr != 16000 else purified

    metrics = {
        "pesq": {
            "og-vs-adv": pesq(16000, original_16k, adversarial_16k, 'wb'),
            "og-vs-prf": pesq(16000, original_16k, purified_16k, 'wb'),
            "adv-vs-prf": pesq(16000, adversarial_16k, purified_16k, 'wb'),
        },
        "estoi": {
            "og-vs-adv": stoi(original, adversarial, sr, extended=True),
            "og-vs-prf": stoi(original, purified, sr, extended=True),
            "adv-vs-prf": stoi(adversarial, purified, sr, extended=True),
        },
    }

    n = adversarial - original
    si_sdr, si_sir, si_sar = energy_ratios(purified, original, n)
    metrics["si-sdr"] = si_sdr
    metrics["si-sir"] = si_sir
    metrics["si-sar"] = si_sar

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Directory containing the original (clean) audio")
    parser.add_argument("--adversarial_dir", type=str, required=True,
                        help="Directory containing the adversarial audio")
    parser.add_argument("--purified_parent_dir", type=str, required=True,
                        help="Parent directory whose subdirectories are one purifier each "
                             "(e.g. purified/sgmse, purified/mambattention)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Discover purifier subdirectories
    # ------------------------------------------------------------------

    purifier_dirs = sorted([
        d for d in Path(args.purified_parent_dir).iterdir()
        if d.is_dir()
    ])

    if not purifier_dirs:
        raise RuntimeError(f"No subdirectories found in {args.purified_parent_dir}")

    purifier_names = [d.name for d in purifier_dirs]
    print(f"Found {len(purifier_names)} purifier(s): {', '.join(purifier_names)}")

    # Results go in the run directory (parent of purified_parent_dir)
    parent_dir = str(Path(args.purified_parent_dir).parent)

    # ------------------------------------------------------------------
    # Build DataFrame column schema
    # ------------------------------------------------------------------

    nested_keys = ["pesq", "estoi"]
    pairs = ["og-vs-adv", "og-vs-prf", "adv-vs-prf"]
    si_metrics = ["si-sdr", "si-sir", "si-sar"]

    # One set of audio-quality columns per purifier; adversarial columns
    # only need to be stored once (og-vs-adv is the same regardless of purifier).
    data = {"filename": []}

    # Adversarial-only columns (computed once)
    for key in nested_keys:
        data[f"{key}_og-vs-adv"] = []
    # SI metrics don't apply to adversarial-only comparison, keep per-purifier

    # Per-purifier columns
    for name in purifier_names:
        for key in nested_keys:
            for pair in ["og-vs-prf", "adv-vs-prf"]:
                data[f"{key}_{pair}_{name}"] = []
        for m in si_metrics:
            data[f"{m}_{name}"] = []
        data[f"wer_adv-vs-og"] = []  # stored once, added below
        data[f"wer_prf-vs-og_{name}"] = []

    # wer_adv-vs-og should only appear once in the schema — clean up the duplicate
    # keys added in the loop above, keep a single column
    data = {k: v for k, v in data.items() if not (k == "wer_adv-vs-og" and data["filename"] == [])}
    data["wer_adv-vs-og"] = []

    # ------------------------------------------------------------------
    # Discover adversarial files (drive the loop)
    # ------------------------------------------------------------------

    adversarial_files = sorted(glob(join(args.adversarial_dir, '*.wav')))
    adversarial_files += sorted(glob(join(args.adversarial_dir, '**', '*.wav')))

    for adversarial_file in tqdm(adversarial_files, desc="Audio metrics"):
        filename = str(Path(adversarial_file).relative_to(args.adversarial_dir))
        original_filename = filename.split("_")[0] + ".wav" if 'dB' in filename else filename

        # Load original and adversarial once
        x, sr_x = read(join(args.original_dir, original_filename))
        y, sr_y = read(join(args.adversarial_dir, filename))
        assert sr_x == sr_y, f"Sampling rate mismatch for {filename}"

        data["filename"].append(filename)

        # Adversarial vs original PESQ/ESTOI (purifier-independent)
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        y_16k = librosa.resample(y, orig_sr=sr_y, target_sr=16000) if sr_y != 16000 else y
        for key, fn in [("pesq", lambda a, b: pesq(16000, a, b, 'wb')),
                        ("estoi", lambda a, b: stoi(x, y, sr_x, extended=True))]:
            if key == "pesq":
                data[f"{key}_og-vs-adv"].append(pesq(16000, x_16k, y_16k, 'wb'))
            else:
                data[f"{key}_og-vs-adv"].append(stoi(x, y, sr_x, extended=True))

        # Per-purifier metrics
        for name, pdir in zip(purifier_names, purifier_dirs):
            purified_path = join(str(pdir), filename)
            x_hat, sr_hat = read(purified_path)
            assert sr_x == sr_hat, f"Sampling rate mismatch for purified {filename} ({name})"

            metrics = compute_audio_metrics(x, y, x_hat, sr_x)

            for key in nested_keys:
                for pair in ["og-vs-prf", "adv-vs-prf"]:
                    data[f"{key}_{pair}_{name}"].append(metrics[key][pair])
            for m in si_metrics:
                data[f"{m}_{name}"].append(metrics[m])

    # ------------------------------------------------------------------
    # Transcription JSONs + WER
    # ------------------------------------------------------------------

    original_json_files = glob(join(args.original_dir, "*.json"))
    adversarial_json_files = glob(join(args.adversarial_dir, "*.json"))

    wer_adversarial_mean, wer_adversarial_std = float('nan'), float('nan')
    wer_purified_stats = {name: (float('nan'), float('nan')) for name in purifier_names}

    if len(original_json_files) != 1 or len(adversarial_json_files) != 1:
        print("Expected exactly one transcription JSON in original and adversarial dirs. "
              "Skipping WER computation.")
        # Fill columns with NaN
        n_files = len(data["filename"])
        data["wer_adv-vs-og"] = [np.nan] * n_files
        for name in purifier_names:
            data[f"wer_prf-vs-og_{name}"] = [np.nan] * n_files
    else:
        print("Loading transcription JSONs...")

        with open(original_json_files[0]) as f:
            original_data = json.load(f)
        with open(adversarial_json_files[0]) as f:
            adversarial_data = json.load(f)

        original_dict = {basename(k): v for k, v in original_data.items()}
        adversarial_dict = {basename(k): v for k, v in adversarial_data.items()}

        # Load purifier JSONs (one per purifier)
        purified_dicts = {}
        for name, pdir in zip(purifier_names, purifier_dirs):
            pjson = glob(join(str(pdir), "*.json"))
            if len(pjson) != 1:
                print(f"  Warning: expected 1 JSON in {pdir}, found {len(pjson)}. "
                      f"Skipping WER for {name}.")
                purified_dicts[name] = {}
            else:
                with open(pjson[0]) as f:
                    purified_dicts[name] = {basename(k): v for k, v in json.load(f).items()}

        wer_adversarial = []
        wer_purified_raw = {name: [] for name in purifier_names}
        merged = {}

        for file_id, original_text in original_data.items():
            adversarial_text = adversarial_dict.get(file_id, "")

            entry = {
                "original": original_text,
                "adversarial": adversarial_text,
            }
            for name in purifier_names:
                entry[f"purified_{name}"] = purified_dicts[name].get(file_id, "")
            merged[file_id] = entry

            if original_text.strip() == "":
                data["wer_adv-vs-og"].append(np.nan)
                for name in purifier_names:
                    data[f"wer_prf-vs-og_{name}"].append(np.nan)
                continue

            # Adversarial WER
            w = 1.0 if adversarial_text.strip() == "" else wer(original_text, adversarial_text)
            wer_adversarial.append(w)
            data["wer_adv-vs-og"].append(w)

            # Per-purifier WER
            for name in purifier_names:
                purified_text = purified_dicts[name].get(file_id, "")
                w = 1.0 if purified_text.strip() == "" else wer(original_text, purified_text)
                wer_purified_raw[name].append(w)
                data[f"wer_prf-vs-og_{name}"].append(w)

        # Save merged JSON
        merged_path = join(parent_dir, "merged_transcriptions.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged transcription file saved to: {merged_path}")

        wer_adversarial_mean, wer_adversarial_std = mean_std(np.array(wer_adversarial))
        for name in purifier_names:
            wer_purified_stats[name] = mean_std(np.array(wer_purified_raw[name]))

    # ------------------------------------------------------------------
    # Print and save results
    # ------------------------------------------------------------------

    df = pd.DataFrame(data)

    print("\n============ AVERAGE METRICS ============")

    print("\nWER:")
    print(f"  Adversarial vs Original:           {wer_adversarial_mean:.3f} ± {wer_adversarial_std:.3f}")
    for name in purifier_names:
        mean_v, std_v = wer_purified_stats[name]
        print(f"  Purified ({name}) vs Original:  {mean_v:.3f} ± {std_v:.3f}")

    for key in nested_keys:
        print(f"\n{key.upper()}:")
        col = f"{key}_og-vs-adv"
        mean_v, std_v = mean_std(df[col].to_numpy())
        print(f"  og-vs-adv:                         {mean_v:.3f} ± {std_v:.3f}")
        for name in purifier_names:
            for pair in ["og-vs-prf", "adv-vs-prf"]:
                col = f"{key}_{pair}_{name}"
                mean_v, std_v = mean_std(df[col].to_numpy())
                label = f"{pair} ({name})"
                print(f"  {label:<35} {mean_v:.3f} ± {std_v:.3f}")

    print("\nSI METRICS:")
    for m in si_metrics:
        for name in purifier_names:
            col = f"{m}_{name}"
            mean_v, std_v = mean_std(df[col].to_numpy())
            print(f"  {m} ({name}):{' ' * (10 - len(m))} {mean_v:.3f} ± {std_v:.3f}")
    print()

    # Save per-file CSV
    results_csv = join(parent_dir, "results_per_file.csv")
    df.to_csv(results_csv, index=False)
    print(f"Per-file results saved to: {results_csv}")

    # Save averaged results
    avg_path = join(parent_dir, "avg_results.txt")
    with open(avg_path, "w") as f:
        f.write("WER:\n")
        f.write(f"  Adversarial vs Original:           {wer_adversarial_mean:.3f} ± {wer_adversarial_std:.3f}\n")
        for name in purifier_names:
            mean_v, std_v = wer_purified_stats[name]
            f.write(f"  Purified ({name}) vs Original:  {mean_v:.3f} ± {std_v:.3f}\n")
        for key in nested_keys:
            f.write(f"\n{key.upper()}:\n")
            col = f"{key}_og-vs-adv"
            mean_v, std_v = mean_std(df[col].to_numpy())
            f.write(f"  og-vs-adv:                         {mean_v:.3f} ± {std_v:.3f}\n")
            for name in purifier_names:
                for pair in ["og-vs-prf", "adv-vs-prf"]:
                    col = f"{key}_{pair}_{name}"
                    mean_v, std_v = mean_std(df[col].to_numpy())
                    label = f"{pair} ({name})"
                    f.write(f"  {label:<35} {mean_v:.3f} ± {std_v:.3f}\n")
        f.write("\nSI METRICS:\n")
        for m in si_metrics:
            for name in purifier_names:
                col = f"{m}_{name}"
                mean_v, std_v = mean_std(df[col].to_numpy())
                f.write(f"  {m} ({name}):{' ' * (10 - len(m))} {mean_v:.3f} ± {std_v:.3f}\n")
    print(f"Average results saved to: {avg_path}")
