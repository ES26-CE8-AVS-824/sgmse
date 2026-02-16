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
    """Compute all metrics for the three signals with nested structure."""
    metrics = {
        "pesq": {},
        "estoi": {}
    }

    # Resample to 16k for PESQ
    original_16k = librosa.resample(original, orig_sr=sr, target_sr=16000) if sr != 16000 else original
    adversarial_16k = librosa.resample(adversarial, orig_sr=sr, target_sr=16000) if sr != 16000 else adversarial
    purified_16k = librosa.resample(purified, orig_sr=sr, target_sr=16000) if sr != 16000 else purified

    # --- PESQ ---
    metrics["pesq"]["og-vs-adv"] = pesq(16000, original_16k, adversarial_16k, 'wb')
    metrics["pesq"]["og-vs-prf"] = pesq(16000, original_16k, purified_16k, 'wb')
    metrics["pesq"]["adv-vs-prf"] = pesq(16000, adversarial_16k, purified_16k, 'wb')

    # --- ESTOI ---
    metrics["estoi"]["og-vs-adv"] = stoi(original, adversarial, sr, extended=True)
    metrics["estoi"]["og-vs-prf"] = stoi(original, purified, sr, extended=True)
    metrics["estoi"]["adv-vs-prf"] = stoi(adversarial, purified, sr, extended=True)

    # --- SI-SDR / SIR / SAR ---
    n = adversarial - original
    si_sdr, si_sir, si_sar = energy_ratios(purified, original, n)
    metrics["si-sdr"] = si_sdr
    metrics["si-sir"] = si_sir
    metrics["si-sar"] = si_sar

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True, help='Directory containing the original data')
    parser.add_argument("--adversarial_dir", type=str, required=True, help='Directory containing the adversarial data')
    parser.add_argument("--purified_dir", type=str, required=True, help='Directory containing the purified data')
    args = parser.parse_args()

    # ======================================================
    # Compute audio quality metrics
    # ======================================================

    # Prepare storage
    data = {"filename": []}
    # Initialize nested dictionaries in the DataFrame
    nested_keys = ["pesq", "estoi"]
    pairs = ["og-vs-adv", "og-vs-prf", "adv-vs-prf"]
    for key in nested_keys:
        for pair in pairs:
            data[f"{key}_{pair}"] = []
    # SI-SDR / SIR / SAR
    si_metrics = ["si-sdr", "si-sir", "si-sar"]
    for m in si_metrics:
        data[m] = []
    # WER
    wers = ["wer_adv-vs-og", "wer_prf-vs-og"]
    for w in wers:
        data[w] = []

    # Results will be saved to purified_dir's parent directory
    parent_dir = join(args.purified_dir, "..")

    # Discover adversarial files (used as reference for filenames)
    adversarial_files = sorted(glob(join(args.adversarial_dir, '*.wav')))
    adversarial_files += sorted(glob(join(args.adversarial_dir, '**', '*.wav')))

    for adversarial_file in tqdm(adversarial_files):
        filename = str(Path(adversarial_file).relative_to(args.adversarial_dir))
        if 'dB' in filename:
            original_filename = filename.split("_")[0] + ".wav"
        else:
            original_filename = filename

        # Load signals
        x, sr_x = read(join(args.original_dir, original_filename))
        y, sr_y = read(join(args.adversarial_dir, filename))
        x_hat, sr_x_hat = read(join(args.purified_dir, filename))
        assert sr_x == sr_y == sr_x_hat, f"Sampling rates do not match for {filename}"

        # Compute metrics
        metrics = compute_audio_metrics(x, y, x_hat, sr_x)

        # Store results
        data["filename"].append(filename)
        for key in nested_keys:
            for pair in pairs:
                data[f"{key}_{pair}"].append(metrics[key][pair])
        for m in si_metrics:
            data[m].append(metrics[m])


    # ======================================================
    # Merge transcription JSONs + Compute WER
    # ======================================================

    # Find JSON files
    original_json_files = glob(join(args.original_dir, "*.json"))
    adversarial_json_files = glob(join(args.adversarial_dir, "*.json"))
    purified_json_files = glob(join(args.purified_dir, "*.json"))

    if len(original_json_files) != 1 or len(adversarial_json_files) != 1 or len(purified_json_files) != 1:
        print("Expected exactly one transcription JSON file in each directory (original, adversarial, purified). Skipping WER computation.")
        wer_adversarial_mean, wer_adversarial_std = float('nan'), float('nan')
        wer_purified_mean, wer_purified_std = float('nan'), float('nan')
    else:
        print("Loading transcription JSONs...")

        with open(original_json_files[0], "r") as f:
            original_data = json.load(f)
        with open(adversarial_json_files[0], "r") as f:
            adversarial_data = json.load(f)
        with open(purified_json_files[0], "r") as f:
            purified_data = json.load(f)

        original_dict = {basename(k): v for k, v in original_data.items()}
        adversarial_dict = {basename(k): v for k, v in adversarial_data.items()}
        purified_dict = {basename(k): v for k, v in purified_data.items()}

        merged = {}
        wer_adversarial = []
        wer_purified = []

        for file_id, original_text in original_data.items():

            adversarial_text = adversarial_dict.get(file_id, "")
            purified_text = purified_dict.get(file_id, "")

            merged[file_id] = {
                "original": original_text,
                "adversarial": adversarial_text,
                "purified": purified_text
            }

            if original_text.strip() != "":

                w = 1.0 if adversarial_text.strip() == "" else wer(original_text, adversarial_text)
                wer_adversarial.append(w)
                data["wer_adv-vs-og"].append(w)

                w = 1.0 if purified_text.strip() != "" else wer(original_text, purified_text)
                wer_purified.append(w)
                data["wer_prf-vs-og"].append(w)
            else:
                data["wer_adv-vs-og"].append(np.nan)
                data["wer_prf-vs-og"].append(np.nan)


        # Save merged JSON in parent directory
        merged_path = join(parent_dir, "merged_transcriptions.json")

        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"Merged transcription file saved to: {merged_path}")

        # Compute mean/std
        wer_adversarial_mean, wer_adversarial_std = mean_std(pd.Series(wer_adversarial).to_numpy())
        wer_purified_mean, wer_purified_std = mean_std(pd.Series(wer_purified).to_numpy())

    # ======================================================
    # Print and save results
    # ======================================================

    # Create DataFrame
    df = pd.DataFrame(data)

    # Print average ± std for each metric, nicely nested
    print("\n============ AVERAGE METRICS ============")
    print("WER:")
    print(f"  Adversarial vs Original: {wer_adversarial_mean:.3f} ± {wer_adversarial_std:.3f}")
    print(f"  Purified vs Original: {wer_purified_mean:.3f} ± {wer_purified_std:.3f}")
    for key in nested_keys:
        print(f"\n{key.upper()}:")
        for pair in pairs:
            mean_val, std_val = mean_std(df[f"{key}_{pair}"].to_numpy())
            print(f"  {pair}: {mean_val:.3f} ± {std_val:.3f}")
    print("\nSI METRICS:")
    for m in si_metrics:
        mean_val, std_val = mean_std(df[m].to_numpy())
        print(f"  {m}: {mean_val:.3f} ± {std_val:.3f}")
    print()

    # Save per-file results
    df.to_csv(join(parent_dir, "results_per_file.csv"), index=False)

    # Save averages to file
    with open(join(parent_dir, "avg_results.txt"), "w") as f:
        f.write("WER:\n")
        f.write(f"  Adversarial vs Original: {wer_adversarial_mean:.3f} ± {wer_adversarial_std:.3f}\n")
        f.write(f"  Purified vs Original: {wer_purified_mean:.3f} ± {wer_purified_std:.3f}\n")
        for key in nested_keys:
            f.write(f"\n{key.upper()}:\n")
            for pair in pairs:
                mean_val, std_val = mean_std(df[f"{key}_{pair}"].to_numpy())
                f.write(f"  {pair}: {mean_val:.3f} ± {std_val:.3f}\n")
        f.write("\nSI METRICS:\n")
        for m in si_metrics:
            mean_val, std_val = mean_std(df[m].to_numpy())
            f.write(f"  {m}: {mean_val:.3f} ± {std_val:.3f}\n")
