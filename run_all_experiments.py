#!/usr/bin/env python3
"""
Run all EgoMimic experiments across multiple skills and GPU sets.

Launches pairs of experiments (with AVP + without AVP) in parallel on
different GPU sets, then runs eval/viz once training finishes.

Usage:
    # Run all 4 skills (8 experiments total: 4 with AVP, 4 without)
    python run_all_experiments.py --val_ratio 0.5 --target_epoch 169

    # Run only specific skills
    python run_all_experiments.py --skills kiwi turn_mug --val_ratio 0.5 --target_epoch 169

    # Run sequentially instead of all at once ( 2 experiments per skill, one skill at a time)
    python run_all_experiments.py --skills turn_mug turn_cup bimanual_apple --val_ratio 0.5 --target_epoch 169 --sequential

    # Dry run: print commands without executing
    python run_all_experiments.py --val_ratio 0.5 --target_epoch 169 --dry_run

    # Only create val splits (no training)
    python run_all_experiments.py --val_ratio 0.5 --splits_only

    # Custom GPU assignment (comma-separated pairs: avp_gpus:no_avp_gpus)
    python run_all_experiments.py --gpu_sets 0,1,2,3:4,5,6,7 --val_ratio 0.5 --target_epoch 169
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── Experiment definitions ─────────────────────────────────────────────────

SKILLS = {
    "kiwi": {
        "avp_config": "egomimic/configs/egomimic_lbm_avp_kiwi.json",
        "no_avp_config": "egomimic/configs/egomimic_lbm_no_avp_kiwi.json",
        "task_name": "PutKiwiInCenterOfTable",
    },
    "turn_mug": {
        "avp_config": "egomimic/configs/egomimic_lbm_avp_turn_mug.json",
        "no_avp_config": "egomimic/configs/egomimic_lbm_no_avp_turn_mug.json",
        "task_name": "TurnMugRightsideUp",
    },
    "turn_cup": {
        "avp_config": "egomimic/configs/egomimic_lbm_avp_turn_cup.json",
        "no_avp_config": "egomimic/configs/egomimic_lbm_no_avp_turn_cup.json",
        "task_name": "TurnCupUpsideDown",
    },
    "bimanual_apple": {
        "avp_config": "egomimic/configs/egomimic_lbm_avp_bimanual_apple.json",
        "no_avp_config": "egomimic/configs/egomimic_lbm_no_avp_bimanual_apple.json",
        "task_name": "BimanualPlaceAppleFromBowlOnCuttingBoard",
    },
    "bimanual_banana": {
        "avp_config": "egomimic/configs/egomimic_lbm_avp_bimanual_banana.json",
        "no_avp_config": "egomimic/configs/egomimic_lbm_no_avp_bimanual_banana.json",
        "task_name": "BimanualPlaceBananaFromBowlOnCuttingBoard",
    },
}


def create_val_splits(skills, val_ratio, val_seed=42):
    """Pre-create val splits for all selected skills (fast with external links)."""
    from update_train_val_split import update_train_val_split

    repo_root = os.path.dirname(os.path.abspath(__file__))

    for skill_name, skill in skills.items():
        config_path = os.path.join(repo_root, skill["avp_config"])
        with open(config_path) as f:
            config = json.load(f)

        original_data = config["train"]["data"]
        if not os.path.isabs(original_data):
            original_data = os.path.normpath(os.path.join(repo_root, original_data))

        task_stem = os.path.splitext(os.path.basename(original_data))[0]
        dataset_root = os.path.dirname(os.path.dirname(original_data))
        split_dir = os.path.join(dataset_root, f"{task_stem}_val{val_ratio}")
        split_hdf5 = os.path.join(split_dir, os.path.basename(original_data))

        if os.path.exists(split_hdf5):
            print(f"  [{skill_name}] Reusing existing split: {split_hdf5}")
        else:
            print(f"  [{skill_name}] Creating split: {split_hdf5}")
            os.makedirs(split_dir, exist_ok=True)
            update_train_val_split(original_data, val_ratio=val_ratio,
                                   seed=val_seed, output_path=split_hdf5)


def build_command(config, val_ratio, target_epoch, gpus, master_port, mode="all",
                  val_seed=42, extra_args=None):
    """Build a run_experiment.py command."""
    cmd = [
        sys.executable, "run_experiment.py",
        "--config", config,
        "--gpus", gpus,
        "--master_port", str(master_port),
        "--mode", mode,
    ]
    if val_ratio is not None:
        cmd.extend(["--val_ratio", str(val_ratio)])
        cmd.extend(["--val_seed", str(val_seed)])
    if target_epoch is not None:
        cmd.extend(["--target_epoch", str(target_epoch)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_experiment_pair(skill_name, skill, val_ratio, target_epoch, gpu_set_avp,
                        gpu_set_no_avp, port_avp, port_no_avp, mode="all",
                        val_seed=42, dry_run=False):
    """
    Launch a pair of experiments (with AVP + without AVP) in parallel.
    Returns list of (label, process) tuples.
    """
    experiments = []

    # With AVP
    cmd_avp = build_command(
        skill["avp_config"], val_ratio, target_epoch,
        gpu_set_avp, port_avp, mode=mode, val_seed=val_seed,
    )
    label_avp = f"{skill_name}_avp"

    # Without AVP
    cmd_no_avp = build_command(
        skill["no_avp_config"], val_ratio, target_epoch,
        gpu_set_no_avp, port_no_avp, mode=mode, val_seed=val_seed,
    )
    label_no_avp = f"{skill_name}_no_avp"

    for label, cmd in [(label_avp, cmd_avp), (label_no_avp, cmd_no_avp)]:
        print(f"\n  [{label}] {' '.join(cmd)}")

    if dry_run:
        return []

    # Launch both in parallel
    log_dir = os.path.join("experiment_logs", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for label, cmd in [(label_avp, cmd_avp), (label_no_avp, cmd_no_avp)]:
        log_path = os.path.join(log_dir, f"{label}_{datetime.now().strftime('%H%M%S')}.log")
        log_file = open(log_path, "w")
        print(f"  [{label}] Log: {log_path}")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((label, proc, log_file, log_path))

    return procs


def wait_for_processes(procs):
    """Wait for all processes to complete and report status."""
    results = {}
    while procs:
        for label, proc, log_file, log_path in list(procs):
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                results[label] = (ret, log_path)
                print(f"  [{label}] {status} (log: {log_path})")
                procs.remove((label, proc, log_file, log_path))
        if procs:
            time.sleep(30)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all EgoMimic experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skills", nargs="+", default=None,
                        choices=list(SKILLS.keys()),
                        help="Skills to run (default: all)")
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="Val split ratio (e.g. 0.5)")
    parser.add_argument("--val_seed", type=int, default=42,
                        help="Random seed for val split")
    parser.add_argument("--target_epoch", type=int, default=None,
                        help="Target training epoch")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "train", "eval", "viz"],
                        help="Run mode (default: all)")
    parser.add_argument("--gpu_sets", type=str, default="0,1,2,3:4,5,6,7",
                        help="GPU sets as avp_gpus:no_avp_gpus (default: 0,1,2,3:4,5,6,7)")
    parser.add_argument("--base_port", type=int, default=29500,
                        help="Base master port (incremented per experiment)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run skill pairs sequentially instead of all at once")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--splits_only", action="store_true",
                        help="Only create val splits, don't train")

    args = parser.parse_args()

    # Select skills
    selected = {k: SKILLS[k] for k in (args.skills or SKILLS.keys())}

    # Parse GPU sets
    gpu_parts = args.gpu_sets.split(":")
    if len(gpu_parts) != 2:
        parser.error("--gpu_sets must be in format 'avp_gpus:no_avp_gpus' (e.g. '0,1,2,3:4,5,6,7')")
    gpu_set_avp, gpu_set_no_avp = gpu_parts

    print(f"\n{'=' * 70}")
    print(f"EgoMimic Experiment Batch Runner")
    print(f"{'=' * 70}")
    print(f"  Skills:       {list(selected.keys())}")
    print(f"  Val ratio:    {args.val_ratio}")
    print(f"  Target epoch: {args.target_epoch}")
    print(f"  Mode:         {args.mode}")
    print(f"  GPU set AVP:  {gpu_set_avp}")
    print(f"  GPU set base: {gpu_set_no_avp}")
    print(f"  Sequential:   {args.sequential}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")
    print()

    # Step 1: Create val splits (fast, sequential)
    if args.val_ratio is not None:
        print("Step 1: Creating val splits...")
        create_val_splits(selected, args.val_ratio, args.val_seed)
        print()

    if args.splits_only:
        print("Done (splits only).")
        return

    # Step 2: Launch experiments
    print("Step 2: Launching experiments...")

    all_procs = []
    port = args.base_port

    skill_items = list(selected.items())

    if args.sequential:
        # Run one skill pair at a time (2 experiments per skill)
        for skill_name, skill in skill_items:
            print(f"\n--- {skill_name} ---")
            procs = run_experiment_pair(
                skill_name, skill, args.val_ratio, args.target_epoch,
                gpu_set_avp, gpu_set_no_avp, port, port + 1,
                mode=args.mode, val_seed=args.val_seed, dry_run=args.dry_run,
            )
            port += 2
            if procs:
                results = wait_for_processes(procs)
                all_procs.extend(results.items())
    else:
        # Launch all skill pairs at once
        for skill_name, skill in skill_items:
            print(f"\n--- {skill_name} ---")
            procs = run_experiment_pair(
                skill_name, skill, args.val_ratio, args.target_epoch,
                gpu_set_avp, gpu_set_no_avp, port, port + 1,
                mode=args.mode, val_seed=args.val_seed, dry_run=args.dry_run,
            )
            port += 2
            all_procs.extend(procs)

        if all_procs:
            print(f"\nWaiting for {len(all_procs)} experiments to complete...")
            results = wait_for_processes(all_procs)

    # Summary
    if not args.dry_run and (args.sequential or all_procs):
        print(f"\n{'=' * 70}")
        print("BATCH SUMMARY")
        print(f"{'=' * 70}")
        if args.sequential:
            for label, (ret, log_path) in all_procs:
                status = "OK" if ret == 0 else f"FAILED"
                print(f"  {status:8s} {label:30s} {log_path}")
        else:
            for label, (ret, log_path) in results.items():
                status = "OK" if ret == 0 else f"FAILED"
                print(f"  {status:8s} {label:30s} {log_path}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
