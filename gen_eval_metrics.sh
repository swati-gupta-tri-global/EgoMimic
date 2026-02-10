#!/bin/bash
# Usage: 
#   ./gen_eval_metrics.sh                # Run full evaluation + visualization
#   ./gen_eval_metrics.sh --visualize-only  # Run only visualization

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Function to run inference
run_inference() {
    local ckpt_path="$1"
    local dataset_path="$2"
    local output_dir="$3"
    local visualize="${4:-false}"  # Optional, defaults to false
    local num_demos="${5:--1}"     # Optional, defaults to -1 (all demos)
    local num_frames="${6:-500}"   # Optional, defaults to 500
    
    local viz_flag=""
    if [ "$visualize" = "true" ]; then
        viz_flag="--visualize"
    fi
    
    if python3 /workspace/externals/EgoMimic/egomimic_inference.py \
        --ckpt_path "$ckpt_path" \
        --dataset_path "$dataset_path" \
        --output_dir "$output_dir" \
        --num_demos "$num_demos" \
        --num_frames "$num_frames" \
        --data_type 0 \
        --val_split \
        $viz_flag 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ Success: $output_dir" | tee -a "$LOG_FILE"
        return 0
    else
        echo "✗ Failed: $output_dir" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Main script logic
main() {
    # Parse arguments
    VISUALIZE_ONLY=false
    for arg in "$@"; do
        if [ "$arg" = "--visualize-only" ]; then
            VISUALIZE_ONLY=true
        fi
    done
    
    LOG_FILE="/workspace/externals/EgoMimic/eval_$(date +%Y%m%d_%H%M%S).log"
    
    if [ "$VISUALIZE_ONLY" = "true" ]; then
        echo "Running in VISUALIZE-ONLY mode at $(date), logging to $LOG_FILE" | tee -a "$LOG_FILE"
    else
        echo "Running FULL evaluation at $(date), logging to $LOG_FILE" | tee -a "$LOG_FILE"
    fi
    
    # LBM (In) + AVP validation on held-out LBM tasks 
    # if [ "$VISUALIZE_ONLY" = "false" ]; then
    #     echo "Running: LBM (In) + AVP validation on held-out LBM tasks" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-19-13-11/models/model_epoch_epoch=279.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/" \
    #         "/workspace/externals/EgoMimic/inf_lbmid_avp_eval_epoch279" \
    #         "false" \
    #         "-1" \
    #         "500"
    # fi

    # visualize specific tasks
    # echo "Running: LBM (In) + AVP - Visualize specific tasks" | tee -a "$LOG_FILE"
    # for task in TurnMugRightsideUp PutKiwiInCenterOfTable TurnCupUpsideDown; do
    #     echo "  Visualizing task: $task" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-19-13-11/models/model_epoch_epoch=279.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/${task}.hdf5" \
    #         "/workspace/externals/EgoMimic/inf_lbmid_avp_eval_epoch279" \
    #         "true" \
    #         "3" \
    #         "200"
    # done

    # LBM (In) model on held-out LBM tasks 
    # if [ "$VISUALIZE_ONLY" = "false" ]; then
    #     echo "Running: LBM (In) model on held-out LBM tasks" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-01-59-15/models/model_epoch_epoch=279.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out" \
    #         "/workspace/externals/EgoMimic/inf_lbmid_eval_epoch279" \
    #         "false" \
    #         "-1" \
    #         "500"
    # fi

    # visualize specific tasks
    # echo "Running: LBM (In) - Visualize specific tasks" | tee -a "$LOG_FILE"
    # for task in TurnMugRightsideUp PutKiwiInCenterOfTable TurnCupUpsideDown; do
    #     echo "  Visualizing task: $task" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-01-59-15/models/model_epoch_epoch=279.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/${task}.hdf5" \
    #         "/workspace/externals/EgoMimic/inf_lbmid_eval_epoch279" \
    #         "true" \
    #         "3" \
    #         "200"
    # done


    # LBM (In) + Egodex on held-out LBM tasks
    # Run: None_DT_2026-01-21-23-41-48
#     python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-21-23-41-48/models/model_epoch_epoch=99.ckpt \
#    --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/ \
#    --output_dir /workspace/externals/EgoMimic/inf_lbmid_egodex_eval_epoch99 --num_frames 500 --num_demos=-1 --val_split --data_type 0
    # echo "Running: LBM (In) + Egodex - Visualize specific tasks" | tee -a "$LOG_FILE"
    # for task in TurnMugRightsideUp PutKiwiInCenterOfTable TurnCupUpsideDown; do
    #     echo " Visualizing task: $task" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-21-23-41-48/models/model_epoch_epoch=119.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/${task}.hdf5" \
    #         "/workspace/externals/EgoMimic/inf_lbm_egodex_eval_epoch119" \
    #         "true" \
    #         "3" \
    #         "200"
    # done
    # if [ "$VISUALIZE_ONLY" = "false" ]; then
    #     echo "Running: LBM (In) + Egodex model on held-out LBM tasks" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-21-23-41-48/models/model_epoch_epoch=119.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out" \
    #         "/workspace/externals/EgoMimic/inf_lbm_egodex_eval_epoch119" \
    #         "false" \
    #         "-1" \
    #         "500"
    # fi

    # LBM (All) model on held-out LBM tasks
    # None_DT_2026-01-25-07-14-27
    # echo "Running: LBM (All) - Visualize specific tasks" | tee -a "$LOG_FILE"
    for task in TurnMugRightsideUp PutKiwiInCenterOfTable TurnCupUpsideDown; do
        echo " Visualizing task: $task" | tee -a "$LOG_FILE"
        run_inference \
            "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-25-07-14-27/models/model_epoch_epoch=169.ckpt" \
            "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/${task}.hdf5" \
            "/workspace/externals/EgoMimic/inf_lbmall_eval_epoch169" \
            "true" \
            "3" \
            "200"
    done

    # if [ "$VISUALIZE_ONLY" = "false" ]; then
    #     echo "Running: LBM (All) model on held-out LBM tasks" | tee -a "$LOG_FILE"
    #     run_inference \
    #         "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-25-07-14-27/models/model_epoch_epoch=169.ckpt" \
    #         "/workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out" \
    #         "/workspace/externals/EgoMimic/inf_lbmall_eval_epoch169" \
    #         "false" \
    #         "-1" \
    #         "500"
    # fi
#     python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-25-07-14-27/models/model_epoch_epoch=169.ckpt \
#    --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out/ \
#    --output_dir /workspace/externals/EgoMimic/inf_lbmall_eval_epoch169 --num_frames 500 --num_demos=-1 --val_split --data_type 0



    # # LBM (In) model on LBM all-tasks
    # python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-01-59-15/models/model_epoch_epoch=279.ckpt \
    #  --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/train_split_combined --output_dir /workspace/externals/EgoMimic/inf_lbmid_eval_all --num_frames 500 --num_demos=-1 --val_split --data_type 0

    
    # # LBM (In) + AVP model on LBM all-tasks
    # python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-19-13-11/models/model_epoch_epoch=279.ckpt \
    #  --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/train_split_combined --output_dir /workspace/externals/EgoMimic/inf_lbmid_avp_eval_all --num_frames 500 --num_demos=-1 --val_split --data_type 0

    # Egomimic model on egomimic data (smallcloth_fold):
    # python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-02-05-01-18-53/models/model_epoch_epoch=169.ckpt --dataset_path /workspace/externals/EgoMimic/datasets/egomimic/robot/smallclothfold_robot.hdf5 --output_dir /workspace/externals/EgoMimic/egomimic_eval_ckpt169 --num_frames 500 --num_demos=1 --val_split --data_type 0 --visualize
    
    # None_DT_2026-02-06-00-05-03
    # single lbm task (kiwi)
    # python3 /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-02-06-00-05-03/models/model_epoch_epoch\=169.ckpt --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/train_split_combined_equal_val_mask/PutKiwiInCenterOfTable.hdf5 --output_dir /workspace/externals/EgoMimic/lbm_kiwi_eval_ckpt169 --num_frames 500 --num_demos=3 --val_split --data_type 0 --visualize

    if [ "$VISUALIZE_ONLY" = "true" ]; then
        echo "Visualization completed at $(date)" | tee -a "$LOG_FILE"
    else
        echo "Eval completed at $(date)" | tee -a "$LOG_FILE"
    fi
    echo "Log saved to: $LOG_FILE"
}

# Run main function
main "$@"

# re-generate normalizer stats for past trainings
# EXP_DIRS=(
#     "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-19-13-11"  # LBM (In) + AVP
#     "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-01-59-15"  # LBM (In)
#     "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-21-23-41-48"  # LBM (In) + Egodex
#     "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-25-07-14-27"  # LBM (All)
#     "/workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-02-05-01-18-53"  # Egomimic (clothes)
# )

# # # Run generate_norm_stats.py for each
# for exp_dir in "${EXP_DIRS[@]}"; do
#     echo "=========================================="
#     echo "Processing: $exp_dir"
#     echo "=========================================="
#     python /workspace/externals/EgoMimic/egomimic/generate_norm_stats.py --exp_dir "$exp_dir"
#     echo ""
# done