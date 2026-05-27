import json
from argparse import ArgumentParser
from glob import glob
from os.path import join, basename
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from jiwer import wer
from pesq import pesq, PesqError
from pystoi import stoi
from soundfile import read
from tqdm import tqdm

from sgmse.util.other import energy_ratios, mean_std


def eval_frac_0_samples(hyps):
    """Fraction of hypotheses that are empty strings."""
    if not hyps:
        return float('nan')
    no_len_count = 0
    for hyp in hyps:
        if len(hyp) == 0:
            no_len_count += 1
    return no_len_count / len(hyps)


def compute_audio_metrics(original, adversarial, purified, sr, compute_si=True):
    """Compute all metrics comparing original/adversarial/purified signals."""
    original_16k = librosa.resample(original, orig_sr=sr, target_sr=16000) if sr != 16000 else original
    adversarial_16k = librosa.resample(adversarial, orig_sr=sr, target_sr=16000) if sr != 16000 else adversarial
    purified_16k = librosa.resample(purified, orig_sr=sr, target_sr=16000) if sr != 16000 else purified

    metrics = {
        "pesq": {
            "raw-vs-adv": pesq(16000, original_16k, adversarial_16k, 'wb', on_error=PesqError.RETURN_VALUES),
            "raw-vs-prf": pesq(16000, original_16k, purified_16k, 'wb', on_error=PesqError.RETURN_VALUES),
            "adv-vs-prf": pesq(16000, adversarial_16k, purified_16k, 'wb', on_error=PesqError.RETURN_VALUES),
        },
        "estoi": {
            "raw-vs-adv": stoi(original, adversarial, sr, extended=True),
            "raw-vs-prf": stoi(original, purified, sr, extended=True),
            "adv-vs-prf": stoi(adversarial, purified, sr, extended=True),
        },
    }

    if compute_si:
        n = adversarial - original
        si_sdr, si_sir, si_sar = energy_ratios(purified, original, n)
        metrics["si-sdr"] = si_sdr
        metrics["si-sir"] = si_sir
        metrics["si-sar"] = si_sar

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Directory containing the original (clean) audio and its transcription JSON")
    parser.add_argument("--adversarial_dir", type=str, required=True,
                        help="Directory containing the adversarial audio and its transcription JSON")
    parser.add_argument("--purified_parent_dir", type=str, required=True,
                        help="Parent directory whose subdirectories are one purifier each "
                             "(e.g. purified/sgmse, purified/mambattention)")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                        help="Directory containing the ground-truth transcription JSON (e.g. "
                             "transcriptions_vctk/). When provided, WER is computed as "
                             "raw-vs-gt, adv-vs-gt, and def-vs-gt against these references "
                             "instead of using the raw Whisper transcriptions as the reference.")
    parser.add_argument("--report_si_metrics", action="store_true",
                        help="When set, SI-SDR / SI-SIR / SI-SAR are computed and reported. "
                             "Omitted by default.")
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
    si_metrics = ["si-sdr", "si-sir", "si-sar"]

    # One set of audio-quality columns per purifier; adversarial columns
    # only need to be stored once (raw-vs-adv is the same regardless of purifier).
    data = {"filename": []}

    # Raw-vs-adversarial columns (computed once)
    for key in nested_keys:
        data[f"{key}_raw-vs-adv"] = []

    # Per-purifier columns
    for name in purifier_names:
        for key in nested_keys:
            for pair in ["raw-vs-prf", "adv-vs-prf"]:
                data[f"{key}_{pair}_{name}"] = []
        if args.report_si_metrics:
            for m in si_metrics:
                data[f"{m}_{name}"] = []

    # WER columns: one shared raw-vs-gt and adv-vs-gt, plus per-purifier def-vs-gt
    data["wer_raw-vs-gt"] = []
    data["wer_adv-vs-gt"] = []
    for name in purifier_names:
        data[f"wer_def-vs-gt_{name}"] = []

    # ------------------------------------------------------------------
    # Discover adversarial files (drive the loop)
    # ------------------------------------------------------------------

    adversarial_files = sorted(glob(join(args.adversarial_dir, '*.wav')))
    adversarial_files += sorted(glob(join(args.adversarial_dir, '**', '*.wav')))

    for adversarial_file in tqdm(adversarial_files, desc="Audio metrics"):
        filename = str(Path(adversarial_file).relative_to(args.adversarial_dir))
        original_filename = filename.split("_")[0] + ".wav" if 'dB' in filename else filename
        
        original_filename = filename.replace("_adv", "_nat") if 'adv' in filename else original_filename
        print(f"Processing: {filename} (original: {original_filename})")

        # Load original and adversarial once
        x, sr_x = read(join(args.original_dir, original_filename))
        y, sr_y = read(join(args.adversarial_dir, filename))
        assert sr_x == sr_y, f"Sampling rate mismatch for {filename}"

        data["filename"].append(filename)

        # Raw vs adversarial PESQ/ESTOI (purifier-independent)
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        y_16k = librosa.resample(y, orig_sr=sr_y, target_sr=16000) if sr_y != 16000 else y
        data["pesq_raw-vs-adv"].append(pesq(16000, x_16k, y_16k, 'wb', on_error=PesqError.RETURN_VALUES))
        data["estoi_raw-vs-adv"].append(stoi(x, y, sr_x, extended=True))

        # Per-purifier metrics
        for name, pdir in zip(purifier_names, purifier_dirs):
            purified_path = join(str(pdir), filename)
            x_hat, sr_hat = read(purified_path)
            #assert sr_x == sr_hat, f"Sampling rate mismatch for purified {filename} ({name})"
            x_hat = librosa.resample(x_hat, orig_sr=sr_hat, target_sr=sr_x) if sr_hat != sr_x else x_hat

            metrics = compute_audio_metrics(x, y, x_hat, sr_x,
                                            compute_si=args.report_si_metrics)

            for key in nested_keys:
                for pair in ["raw-vs-prf", "adv-vs-prf"]:
                    data[f"{key}_{pair}_{name}"].append(metrics[key][pair])
            if args.report_si_metrics:
                for m in si_metrics:
                    data[f"{m}_{name}"].append(metrics[m])

    # ------------------------------------------------------------------
    # Transcription JSONs + WER
    # ------------------------------------------------------------------

    # Resolve which directory holds the ground-truth references
    gt_dir = args.ground_truth_dir if args.ground_truth_dir else args.original_dir

    gt_json_files = glob(join(gt_dir, "*.json"))
    original_json_files = glob(join(args.original_dir, "*.json"))
    adversarial_json_files = glob(join(args.adversarial_dir, "*.json"))

    # Default NaN stats
    wer_raw_mean, wer_raw_std = float('nan'), float('nan')
    wer_adversarial_mean, wer_adversarial_std = float('nan'), float('nan')
    wer_purified_stats = {name: (float('nan'), float('nan')) for name in purifier_names}
    frac_zero_adversarial = float('nan')
    frac_zero_purified = {name: float('nan') for name in purifier_names}

    missing = []
    if len(gt_json_files) != 1:
        missing.append(f"ground-truth dir ({gt_dir})")
        print(f"Warning: expected exactly one transcription JSON in {gt_dir}, found {len(gt_json_files)}. "
              f"WER vs ground truth will not be computed.")
    if len(original_json_files) != 1:
        missing.append(f"original dir ({args.original_dir})")
        print(f"Warning: expected exactly one transcription JSON in {args.original_dir}, found {len(original_json_files)}. "
              f"WER vs ground truth will not be computed.")
    if len(adversarial_json_files) != 1:
        missing.append(f"adversarial dir ({args.adversarial_dir})")
        print(f"Warning: expected exactly one transcription JSON in {args.adversarial_dir}, found {len(adversarial_json_files)}. "
              f"WER vs ground truth will not be computed.")

    if missing:
        print("Expected exactly one transcription JSON in each of: "
              + ", ".join(missing) + ". Skipping WER computation.")
        n_files = len(data["filename"])
        data["wer_raw-vs-gt"] = [np.nan] * n_files
        data["wer_adv-vs-gt"] = [np.nan] * n_files
        for name in purifier_names:
            data[f"wer_def-vs-gt_{name}"] = [np.nan] * n_files
    else:
        print("Loading transcription JSONs...")

        with open(gt_json_files[0]) as f:
            gt_data = json.load(f)
        with open(original_json_files[0]) as f:
            original_data = json.load(f)
        with open(adversarial_json_files[0]) as f:
            adversarial_data = json.load(f)

        original_dict = {basename(k.replace("_nat", "")): v for k, v in original_data.items()}
        adversarial_dict = {basename(k.replace("_adv", "")): v for k, v in adversarial_data.items()}

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
                    purified_dicts[name] = {basename(k.replace("_adv", "")): v for k, v in json.load(f).items()}

        wer_raw_list = []
        wer_adversarial = []
        wer_purified_raw = {name: [] for name in purifier_names}
        adv_hyps = []
        purified_hyps = {name: [] for name in purifier_names}
        merged = {}

        # Iterate over ground-truth entries as the reference
        for file_id, gt_text in gt_data.items():
            file_id = file_id.replace("_nat", "")
            raw_text = original_dict.get(file_id, "")
            adversarial_text = adversarial_dict.get(file_id, "")

            entry = {
                "ground_truth": gt_text,
                "raw": raw_text,
                "adversarial": adversarial_text,
            }
            for name in purifier_names:
                entry[f"purified_{name}"] = purified_dicts[name].get(file_id, "")
            merged[file_id] = entry

            adv_hyps.append(adversarial_text)
            for name in purifier_names:
                purified_hyps[name].append(purified_dicts[name].get(file_id, ""))

            if gt_text.strip() == "":
                data["wer_raw-vs-gt"].append(np.nan)
                data["wer_adv-vs-gt"].append(np.nan)
                for name in purifier_names:
                    data[f"wer_def-vs-gt_{name}"].append(np.nan)
                continue

            # Raw WER vs ground truth
            w_raw = 1.0 if raw_text.strip() == "" else wer(gt_text, raw_text)
            wer_raw_list.append(w_raw)
            data["wer_raw-vs-gt"].append(w_raw)

            # Adversarial WER vs ground truth
            w_adv = 1.0 if adversarial_text.strip() == "" else wer(gt_text, adversarial_text)
            wer_adversarial.append(w_adv)
            data["wer_adv-vs-gt"].append(w_adv)

            # Per-purifier (defended) WER vs ground truth
            for name in purifier_names:
                purified_text = purified_dicts[name].get(file_id, "")
                w = 1.0 if purified_text.strip() == "" else wer(gt_text, purified_text)
                wer_purified_raw[name].append(w)
                data[f"wer_def-vs-gt_{name}"].append(w)

        # Save merged JSON
        merged_path = join(parent_dir, "merged_transcriptions.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged transcription file saved to: {merged_path}")

        wer_raw_mean, wer_raw_std = mean_std(np.array(wer_raw_list))
        wer_adversarial_mean, wer_adversarial_std = mean_std(np.array(wer_adversarial))
        for name in purifier_names:
            wer_purified_stats[name] = mean_std(np.array(wer_purified_raw[name]))

        frac_zero_adversarial = eval_frac_0_samples(adv_hyps)
        for name in purifier_names:
            frac_zero_purified[name] = eval_frac_0_samples(purified_hyps[name])

    # ------------------------------------------------------------------
    # Print and save results
    # ------------------------------------------------------------------

    df = pd.DataFrame(data)

    # Determine the longest label across all sections so numbers align globally
    all_labels = (
            ["raw-vs-gt", "adv-vs-gt"]
            + [f"def-vs-gt ({n})" for n in purifier_names]
            + ["raw-vs-adv"]
            + [f"{pair} ({n})" for n in purifier_names for pair in ["raw-vs-prf", "adv-vs-prf"]]
    )
    if args.report_si_metrics:
        all_labels += [f"{m} ({n})" for m in si_metrics for n in purifier_names]
    W = max(len(l) for l in all_labels) + 4  # +4 for the leading "  " and a gap

    def fmt(label, mean_v, std_v):
        full = f"  {label}"
        return f"{full:<{W}} {mean_v:.3f} ± {std_v:.3f}"

    lines = ["\n============ AVERAGE METRICS ============"]

    lines.append("\nWER:")
    lines.append(fmt("raw-vs-gt", wer_raw_mean, wer_raw_std))
    lines.append(fmt("adv-vs-gt", wer_adversarial_mean, wer_adversarial_std))
    for name in purifier_names:
        mean_v, std_v = wer_purified_stats[name]
        lines.append(fmt(f"def-vs-gt ({name})", mean_v, std_v))

    lines.append("\nFrac0:")
    lines.append(fmt("adv", frac_zero_adversarial, 0.0))
    for name in purifier_names:
        lines.append(fmt(f"def ({name})", frac_zero_purified[name], 0.0))

    for key in nested_keys:
        lines.append(f"\n{key.upper()}:")
        col = f"{key}_raw-vs-adv"
        mean_v, std_v = mean_std(df[col].to_numpy())
        lines.append(fmt("raw-vs-adv", mean_v, std_v))
        for name in purifier_names:
            for pair in ["raw-vs-prf", "adv-vs-prf"]:
                col = f"{key}_{pair}_{name}"
                mean_v, std_v = mean_std(df[col].to_numpy())
                lines.append(fmt(f"{pair} ({name})", mean_v, std_v))

    if args.report_si_metrics:
        lines.append("\nSI METRICS:")
        for m in si_metrics:
            for name in purifier_names:
                col = f"{m}_{name}"
                mean_v, std_v = mean_std(df[col].to_numpy())
                lines.append(fmt(f"{m} ({name})", mean_v, std_v))

    lines.append("")

    output = "\n".join(lines)
    print(output)

    # Save per-file CSV
    results_csv = join(parent_dir, "results_per_file.csv")
    df.to_csv(results_csv, index=False)
    print(f"Per-file results saved to: {results_csv}")

    # Save averaged results — reuse the same formatted lines
    avg_path = join(parent_dir, "avg_results.txt")
    with open(avg_path, "w") as f:
        f.write(output.strip() + "\n")
    print(f"Average results saved to: {avg_path}")