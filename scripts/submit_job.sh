#!/bin/bash
#SBATCH --job-name=kpms_train_combined_OFA          # Job name
#SBATCH --partition=gpu_a100           # Partition 
#SBATCH --qos=gpu_training             # QoS 
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=8             # CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs (change to gpu:2 for multi-GPU)
#SBATCH --mem=124G                      # Memory per node
#SBATCH --time=11:00:00                # Time limit (HH:MM:SS)
#SBATCH --output=logs/kpms_%j.out      # Standard output log (%j = job ID)
#SBATCH --error=logs/kpms_%j.err       # Standard error log

# ============================================================================
# KeyPoint-MoSeq SLURM Job Submission Script
#
# This script submits KPMS training jobs to an HPC cluster using SLURM.
# Supports single jobs and array jobs for parameter grid search.
#
# Usage:
#   sbatch scripts/submit_job.sh
#
# For array jobs (test multiple parameters):
#   Uncomment the array job section below and adjust parameters
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="


# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# ============================================================================
# CONFIGURATION PARAMETERS
# Adjust these for your analysis
# ============================================================================

# Project paths
PROJECT_ROOT="/flashscratch/chouda/kpms_combined_OFA/kpms_kumarlab"  
POSE_DIR="${PROJECT_ROOT}/data/combined_OFA/poses_csv_10k"
VIDEO_DIR="${PROJECT_ROOT}/data/combined_OFA/videos"
#POSE_DIR="${PROJECT_ROOT}/examples/jabs600_v2/poses_csv"
#VIDEO_DIR="${PROJECT_ROOT}/examples/jabs600_v2/videos"

# Model hyperparameters
KAPPA=1e6                    # Stickiness parameter
ARHMM_ITERS=50              # AR-HMM iterations
FULL_MODEL_ITERS=300         # Full model iterations
MIXED_MAP_ITERS=8           # GPU memory management

# GPU configuration
NUM_GPUS=1                   # Number of GPUs to use

PROJECT_PATH="${PROJECT_ROOT}/results/combined_OFA_kappa_${KAPPA}"
#PROJECT_PATH="${PROJECT_ROOT}/results/test_${KAPPA}"

CONFIG_10k="${PROJECT_ROOT}/config/config_10k.yml"


# ============================================================================
# SINGLE JOB EXECUTION
# ============================================================================

# Change to project directory
cd $PROJECT_ROOT || exit 1

# Run training
echo "Starting KPMS training..."
echo "Pose directory: $POSE_DIR"
echo "Project path: $PROJECT_PATH"
echo "Video directory: $VIDEO_DIR"
echo "Kappa: $KAPPA"
echo "Config: $CONFIG_10k"
echo "=========================================="

python scripts/train_kpms.py \
    --pose-dir "$POSE_DIR" \
    --project-path "$PROJECT_PATH" \
    --video-dir "$VIDEO_DIR" \
    --kappa "$KAPPA" \
    --arhmm-iters "$ARHMM_ITERS" \
    --full-model-iters "$FULL_MODEL_ITERS" \
    --mixed-map-iters "$MIXED_MAP_ITERS" \
    --num-gpus "$NUM_GPUS"\
    --config "$CONFIG_10k"
   

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Results saved to: $PROJECT_PATH"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed! Check error logs."
    echo "=========================================="
    exit 1
fi

# ============================================================================
# ARRAY JOB EXAMPLE (Parameter Grid Search)
# Uncomment this section to test multiple parameter combinations
# ============================================================================

# #SBATCH --array=0-8                  # Array job with 9 configurations
#
# # Define parameter grid
# KAPPA_VALUES=(0.01 0.1 1.0)         # 3 kappa values
# MIXED_MAP_VALUES=(4 8 16)            # 3 memory management values
#
# # Calculate indices for this array task
# KAPPA_IDX=$((SLURM_ARRAY_TASK_ID / 3))
# MIXED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
#
# # Get parameter values for this task
# KAPPA=${KAPPA_VALUES[$KAPPA_IDX]}
# MIXED_MAP_ITERS=${MIXED_MAP_VALUES[$MIXED_IDX]}
#
# # Update project path to include parameters
# PROJECT_PATH="${PROJECT_ROOT}/results/kappa_${KAPPA}_mixed_${MIXED_MAP_ITERS}"
#
# echo "Array Job ID: $SLURM_ARRAY_TASK_ID"
# echo "Testing: kappa=$KAPPA, mixed_map_iters=$MIXED_MAP_ITERS"
#
# # Run training with current parameters
# python scripts/train_kpms.py \
#     --pose-dir "$POSE_DIR" \
#     --project-path "$PROJECT_PATH" \
#     --video-dir "$VIDEO_DIR" \
#     --kappa "$KAPPA" \
#     --arhmm-iters "$ARHMM_ITERS" \
#     --full-model-iters "$FULL_MODEL_ITERS" \
#     --mixed-map-iters "$MIXED_MAP_ITERS" \
#     --num-gpus "$NUM_GPUS"

# ============================================================================
# CLEANUP AND NOTIFICATIONS
# ============================================================================

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

# Optional: Copy results to backup location
# rsync -av "$PROJECT_PATH" /backup/location/

