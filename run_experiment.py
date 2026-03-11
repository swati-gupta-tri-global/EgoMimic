#!/usr/bin/env python3
"""
Unified experiment runner for EgoMimic: train → evaluate → visualize.

Examples:
    # Full pipeline: train to epoch 169 then eval + viz on val split
    python run_experiment.py \
        --config egomimic/configs/egomimic_lbm_avp_kiwi.json \
        --target_epoch 169 --gpus 4,5,6,7

    # Train with a custom 50% val split (auto-creates split HDF5)
    python run_experiment.py \
        --config egomimic/configs/egomimic_lbm_avp_kiwi.json \
        --val_ratio 0.5 --target_epoch 169 --gpus 4,5,6,7

    # Eval-only on an existing checkpoint
    python run_experiment.py \
        --ckpt_path trained_models_highlevel/test/..._DT_.../models/model_epoch_epoch=169.ckpt \
        --mode eval

    # Eval-only, auto-discover checkpoint from experiment directory
    python run_experiment.py \
        --exp_dir trained_models_highlevel/test/..._DT_.../ \
        --target_epoch 169 --mode eval

    # Train only, no eval
    python run_experiment.py \
        --config egomimic/configs/egomimic_lbm_avp_kiwi.json \
        --target_epoch 300 --gpus 0,1,2,3 --mode train
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import egomimic


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_output_base(config_dict):
    """Resolve output_dir the same way training does (relative to egomimic package)."""
    output_base = config_dict.get("train", {}).get("output_dir", "../trained_models_highlevel")
    if not os.path.isabs(output_base):
        # Match egomimic/utils/train_utils.py:get_exp_dir — relative to egomimic module
        output_base = os.path.join(egomimic.__path__[0], output_base)
    return os.path.normpath(output_base)

def find_latest_exp_dir(output_base, exp_name):
    """Return the most-recently-modified experiment directory."""
    base = os.path.join(output_base, exp_name)
    if not os.path.isdir(base):
        return None
    dirs = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def find_checkpoint(exp_dir, target_epoch=None):
    """
    Find a checkpoint in *exp_dir*/models/.

    Returns (ckpt_path, actual_epoch).
    If *target_epoch* is None, returns the latest checkpoint.
    """
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models/ directory in {exp_dir}")

    ckpts = {}
    for f in os.listdir(models_dir):
        m = re.match(r"model_epoch_epoch=(\d+)\.ckpt", f)
        if m:
            ckpts[int(m.group(1))] = os.path.join(models_dir, f)

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {models_dir}")

    if target_epoch is not None and target_epoch in ckpts:
        return ckpts[target_epoch], target_epoch

    nearest = min(ckpts.keys(), key=lambda e: abs(e - (target_epoch or max(ckpts))))
    if target_epoch is not None and nearest != target_epoch:
        print(f"WARNING: Epoch {target_epoch} not found. Using nearest: epoch {nearest}")
        print(f"  Available epochs: {sorted(ckpts.keys())}")
    return ckpts[nearest], nearest


def exp_dir_from_ckpt(ckpt_path):
    """Derive the experiment directory from a checkpoint path."""
    # .../models/model_epoch_epoch=169.ckpt → ...
    return os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))


def run_training(config_path, gpus, master_port, description=None, extra_args=None):
    """Launch training via torchrun and return the process exit code."""
    gpu_list = gpus.split(",")
    n_gpus = len(gpu_list)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        f"--master_port={master_port}",
        "egomimic/scripts/pl_train.py",
        "--config", config_path,
        "--gpus-per-node", str(n_gpus),
        "--num-nodes", "1",
    ]
    if description:
        cmd.extend(["--description", description])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CUDA_VISIBLE_DEVICES={gpus}")
    print()

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Print training output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"\nERROR: Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # torchrun can exit 0 even when workers crash — check output for errors
    combined = (result.stdout or "") + (result.stderr or "")
    error_patterns = ["Traceback (most recent call last)", "run failed with error"]
    for pattern in error_patterns:
        if pattern in combined:
            print(f"\nERROR: Training crashed (found '{pattern}' in output)")
            print("Aborting eval/viz — fix the training error first.")
            sys.exit(1)


def run_inference_cmd(ckpt_path, dataset_path, output_dir,
                      num_frames=500, num_demos=-1, data_type=0,
                      val_split=True, visualize=False):
    """Run inference as a subprocess."""
    cmd = [
        sys.executable, "egomimic_inference.py",
        "--ckpt_path", ckpt_path,
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--num_frames", str(num_frames),
        "--num_demos", str(num_demos),
        "--data_type", str(data_type),
    ]
    if val_split:
        cmd.append("--val_split")
    if visualize:
        cmd.append("--visualize")

    label = "VISUALIZATION" if visualize else "EVALUATION"
    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Inference failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def print_results_summary(eval_dir, config_path, ckpt_path, epoch, val_ratio=None):
    """Print a clean summary of experiment results."""
    print(f"\n{'=' * 70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    if config_path:
        print(f"  Config:     {config_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch:      {epoch}")
    if val_ratio is not None:
        print(f"  Val ratio:  {val_ratio}")
    print(f"  Output dir: {eval_dir}")

    # Try to load metrics
    for mf in [
        os.path.join(eval_dir, "metrics", "aggregated_metrics_all_tasks.json"),
        os.path.join(eval_dir, "metrics", "metrics_summary.json"),
    ]:
        if os.path.exists(mf):
            with open(mf) as f:
                metrics = json.load(f)
            summary = metrics.get("aggregated_summary", metrics)
            print("\n  Metrics:")
            for k, v in sorted(summary.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
            break
    else:
        print("\n  (no metrics file found)")

    viz_dir = os.path.join(eval_dir, "viz")
    if os.path.isdir(viz_dir):
        mp4s = glob.glob(os.path.join(viz_dir, "*.mp4"))
        print(f"\n  Visualization: {len(mp4s)} MP4(s) in {viz_dir}")

    print(f"{'=' * 70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EgoMimic experiment runner: train → eval → visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- what to run ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (required for training)")
    parser.add_argument("--target_epoch", type=int, default=None,
                        help="Epoch to evaluate at (latest if omitted)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "train", "eval", "viz"],
                        help="all=train+eval+viz, train, eval, or viz")

    # --- existing experiment ---
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Explicit checkpoint path (skips training)")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Existing experiment directory (skips training)")

    # --- training options ---
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (default: '0')")
    parser.add_argument("--master_port", type=int, default=29501,
                        help="torchrun master port")

    # --- data split options ---
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="Val split ratio (e.g. 0.5). Auto-creates split HDF5 if needed.")
    parser.add_argument("--val_seed", type=int, default=42,
                        help="Random seed for val split (default: 42)")

    # --- eval options ---
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override eval dataset path")
    parser.add_argument("--num_viz_demos", type=int, default=3,
                        help="Number of demos to visualize (default: 3)")
    parser.add_argument("--num_frames", type=int, default=500,
                        help="Frames per demo for metrics (default: 500)")
    parser.add_argument("--data_type", type=int, default=0, choices=[0, 1],
                        help="0=robot, 1=hand (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override eval output directory")

    args = parser.parse_args()

    do_train = args.mode in ("all", "train")
    do_eval = args.mode in ("all", "eval")
    do_viz = args.mode in ("all", "viz")

    # ── 1. Load config ───────────────────────────────────────────────────
    config_dict = None
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)

    # ── 1b. Apply val split if requested ─────────────────────────────────
    if args.val_ratio is not None and config_dict:
        original_data = config_dict["train"]["data"]
        # Resolve relative to repo root
        if not os.path.isabs(original_data):
            original_data = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), original_data)
            )

        # Derive split output: {dataset_root}/{TaskName}_val{ratio}/{TaskName}.hdf5
        task_stem = os.path.splitext(os.path.basename(original_data))[0]
        dataset_root = os.path.dirname(os.path.dirname(original_data))
        split_dir = os.path.join(dataset_root, f"{task_stem}_val{args.val_ratio}")
        split_hdf5 = os.path.join(split_dir, os.path.basename(original_data))

        if not os.path.exists(split_hdf5):
            print(f"\n{'=' * 70}")
            print(f"CREATING VAL SPLIT (ratio={args.val_ratio})")
            print(f"{'=' * 70}")
            print(f"  Source: {original_data}")
            print(f"  Output: {split_hdf5}")
            from update_train_val_split import update_train_val_split
            os.makedirs(split_dir, exist_ok=True)
            update_train_val_split(
                original_data, val_ratio=args.val_ratio,
                seed=args.val_seed, output_path=split_hdf5,
            )
        else:
            print(f"Reusing existing split: {split_hdf5}")

        # Patch config in-memory so training uses the split dataset
        config_dict["train"]["data"] = split_hdf5

    # ── 1c. Build experiment description ──────────────────────────────────
    description = None
    if do_train and config_dict:
        desc_parts = []
        data_path = config_dict["train"]["data"]
        task_stem = os.path.splitext(os.path.basename(data_path))[0]
        desc_parts.append(task_stem)
        if config_dict["train"]["data_2"] is not None:
            desc_parts.append("avp")
        if args.val_ratio is not None:
            desc_parts.append(f"val{args.val_ratio}")
        description = "_".join(desc_parts)

    # ── 2. TRAIN ─────────────────────────────────────────────────────────
    if do_train:
        if args.config is None:
            parser.error("--config is required for training")

        # Write patched config to a temp file so training sees updated data paths
        train_config_path = args.config
        if args.val_ratio is not None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="egomimic_cfg_",
                dir=os.path.dirname(args.config), delete=False,
            )
            json.dump(config_dict, tmp, indent=4)
            tmp.close()
            train_config_path = tmp.name

        # Record existing exp dirs so we can find the new one after training
        output_base = resolve_output_base(config_dict)
        exp_name = config_dict.get("experiment", {}).get("name", "test")
        search_base = os.path.join(output_base, exp_name)
        pre_dirs = set(os.listdir(search_base)) if os.path.isdir(search_base) else set()

        try:
            run_training(train_config_path, args.gpus, args.master_port,
                         description=description)
        finally:
            # Clean up temp config
            if args.val_ratio is not None and os.path.exists(train_config_path):
                os.unlink(train_config_path)

        # Discover new experiment directory
        if os.path.isdir(search_base):
            post_dirs = set(os.listdir(search_base))
            new_dirs = post_dirs - pre_dirs
            if new_dirs:
                # When running parallel experiments, multiple new dirs may appear.
                # Filter by description prefix to pick the correct one.
                if description and len(new_dirs) > 1:
                    matched = [d for d in new_dirs if d.startswith(description + "_DT_")]
                    if matched:
                        new_dirs = matched
                # In multi-GPU training, non-rank-0 processes may create ghost
                # directories (with a slightly different timestamp) that lack a
                # models/ subdirectory.  Prefer dirs that actually contain models/.
                new_dirs_list = sorted(new_dirs)
                with_models = [d for d in new_dirs_list
                               if os.path.isdir(os.path.join(search_base, d, "models"))]
                best = max(with_models) if with_models else max(new_dirs_list)
                args.exp_dir = os.path.join(search_base, best)
                print(f"\nNew experiment directory: {args.exp_dir}")
            elif args.exp_dir is None:
                print(f"\nWARNING: No new experiment directory found in {search_base}")
                print(f"  pre_dirs ({len(pre_dirs)}), post_dirs ({len(post_dirs)})")
                print("Training may have failed to create an experiment directory.")
                if do_eval or do_viz:
                    print("Aborting eval/viz — no valid experiment directory found.")
                    sys.exit(1)

        # Save the final (possibly patched) config into the experiment dir
        if args.exp_dir and config_dict:
            run_config_path = os.path.join(args.exp_dir, "run_config.json")
            with open(run_config_path, "w") as f:
                json.dump(config_dict, f, indent=4)
            print(f"Saved run config to: {run_config_path}")

    # ── 3. FIND CHECKPOINT ───────────────────────────────────────────────
    ckpt_path = args.ckpt_path
    actual_epoch = args.target_epoch
    exp_dir = args.exp_dir

    if ckpt_path:
        exp_dir = exp_dir or exp_dir_from_ckpt(ckpt_path)
        if actual_epoch is None:
            m = re.search(r"epoch=(\d+)", ckpt_path)
            actual_epoch = int(m.group(1)) if m else 0
    elif exp_dir:
        ckpt_path, actual_epoch = find_checkpoint(exp_dir, args.target_epoch)
    elif do_eval or do_viz:
        # Try auto-discovering from config
        if config_dict:
            output_base = resolve_output_base(config_dict)
            exp_name = config_dict.get("experiment", {}).get("name", "test")
            exp_dir = find_latest_exp_dir(output_base, exp_name)
            if exp_dir:
                ckpt_path, actual_epoch = find_checkpoint(exp_dir, args.target_epoch)
        if ckpt_path is None:
            parser.error("Cannot find checkpoint. Provide --ckpt_path, --exp_dir, or --config")

    if (do_eval or do_viz) and ckpt_path is None:
        parser.error("No checkpoint found for evaluation. Provide --ckpt_path or --exp_dir")

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Epoch:      {actual_epoch}")
    print(f"Exp dir:    {exp_dir}")

    # ── 4. Determine eval dataset ────────────────────────────────────────
    dataset_path = args.dataset
    if dataset_path is None and config_dict:
        dataset_path = config_dict.get("train", {}).get("data")
        if dataset_path and not os.path.isabs(dataset_path):
            dataset_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), dataset_path)
            )

    if (do_eval or do_viz) and not dataset_path:
        parser.error("No dataset path found. Provide --dataset or ensure config has train.data")

    # ── 5. Determine eval output dir ─────────────────────────────────────
    eval_dir = args.output_dir
    if eval_dir is None and exp_dir:
        eval_dir = os.path.join(exp_dir, f"eval_ep{actual_epoch}")

    if eval_dir is None:
        config_stem = os.path.splitext(os.path.basename(args.config))[0] if args.config else "eval"
        eval_dir = f"eval_{config_stem}_ep{actual_epoch}"

    # ── 6. EVAL ──────────────────────────────────────────────────────────
    if do_eval:
        metrics_dir = os.path.join(eval_dir, "metrics")
        run_inference_cmd(
            ckpt_path, dataset_path, metrics_dir,
            num_frames=args.num_frames, num_demos=-1,
            data_type=args.data_type, val_split=True, visualize=False,
        )

    # ── 7. VIZ ───────────────────────────────────────────────────────────
    if do_viz:
        viz_dir = os.path.join(eval_dir, "viz")
        run_inference_cmd(
            ckpt_path, dataset_path, viz_dir,
            num_frames=200, num_demos=args.num_viz_demos,
            data_type=args.data_type, val_split=True, visualize=True,
        )

    # ── 8. SUMMARY ───────────────────────────────────────────────────────
    if do_eval or do_viz:
        print_results_summary(eval_dir, args.config, ckpt_path, actual_epoch,
                              val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
