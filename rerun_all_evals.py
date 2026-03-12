#!/usr/bin/env python3
"""
Re-run eval for all existing experiment directories.

Iterates over trained_models_highlevel/test/, reads experiment_info.json
for each dir, and calls run_experiment.py --exp_dir ... --dataset ... --mode eval.

Usage (inside Docker):
    cd /workspace/externals/EgoMimic
    CUDA_VISIBLE_DEVICES=0 python3 rerun_all_evals.py --target_epoch 169
    CUDA_VISIBLE_DEVICES=0 python3 rerun_all_evals.py --target_epoch 169 --dry_run
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Re-run eval for all existing experiments")
    parser.add_argument("--target_epoch", type=int, default=169)
    parser.add_argument("--input_dir", type=str,
                        default="egomimic/trained_models_highlevel/test")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    trained_dir = os.path.join(repo_root, args.input_dir)

    if not os.path.isdir(trained_dir):
        print(f"ERROR: {trained_dir} not found")
        sys.exit(1)

    exp_dirs = sorted(
        d for d in os.listdir(trained_dir)
        if os.path.isdir(os.path.join(trained_dir, d))
    )

    to_run = []
    for dirname in exp_dirs:
        exp_dir = os.path.join(trained_dir, dirname)

        # Must have models/
        if not os.path.isdir(os.path.join(exp_dir, "models")):
            print(f"  SKIP (no models): {dirname}")
            continue

        # Must have experiment_info.json with dataset
        info_path = os.path.join(exp_dir, "experiment_info.json")
        if not os.path.exists(info_path):
            print(f"  SKIP (no experiment_info.json): {dirname}")
            continue

        with open(info_path) as f:
            info = json.load(f)

        dataset = info.get("dataset")
        if not dataset:
            print(f"  SKIP (no dataset in experiment_info.json): {dirname}")
            continue

        to_run.append((dirname, exp_dir, dataset))

    print(f"\nFound {len(to_run)} experiments to re-eval.\n")

    failed = []
    for i, (dirname, exp_dir, dataset) in enumerate(to_run):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(to_run)}] {dirname}")
        print(f"  dataset: {dataset}")
        print(f"{'=' * 70}")

        cmd = [
            sys.executable, "run_experiment.py",
            "--exp_dir", exp_dir,
            "--dataset", dataset,
            "--target_epoch", str(args.target_epoch),
            "--mode", "eval",
        ]
        print(f"  cmd: {' '.join(cmd)}\n")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"  ERROR: eval failed (exit {result.returncode})")
            failed.append(dirname)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"DRY RUN: {len(to_run)} experiments would be re-evaluated.")
    else:
        ok = len(to_run) - len(failed)
        print(f"Done: {ok}/{len(to_run)} succeeded.")
        if failed:
            print("Failed:")
            for d in failed:
                print(f"  {d}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
