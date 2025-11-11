#!/usr/bin/env python
"""
KeyPoint-MoSeq Training Script for HPC Batch Execution

This script enables non-interactive training of KPMS models, suitable for overnight
runs on HPC clusters. It properly configures JAX GPU memory management to avoid
preallocation issues.

Usage:
    python train_kpms.py --pose-dir <path> --project-path <path> --video-dir <path> [options]

Example:
    python train_kpms.py \
        --pose-dir examples/jabs600_v2/poses \
        --project-path results/my_analysis \
        --video-dir examples/jabs600_v2/videos \
        --kappa 0.1 \
        --arhmm-iters 10 \
        --full-model-iters 10
"""

import os
import sys

# ============================================================================
# CRITICAL: Set JAX environment variables BEFORE any imports
# These MUST be set before JAX is imported to prevent 75% GPU preallocation
# ============================================================================
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Now safe to import other libraries
import argparse
import pathlib
import logging
from typing import List

# JAX and KeyPoint-MoSeq imports
import jax
from jax_moseq.utils import set_mixed_map_iters, set_mixed_map_gpus
import keypoint_moseq as kpms

# Add project root to Python path
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.utils import (
    set_up_logging, print_gpu_usage, validate_data_quality,
    load_config, save_config, merge_config_with_args
)
from src.methods import load_and_format_data, perform_pca, fit_and_save_model, generate_plots_and_movies


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train KeyPoint-MoSeq model for behavioral analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--pose-dir',
        type=str,
        required=True,
        help='Directory containing pose CSV files'
    )
    parser.add_argument(
        '--project-path',
        type=str,
        required=True,
        help='Path to save project results and checkpoints'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='Directory containing corresponding video files'
    )

    # Model hyperparameters
    parser.add_argument(
        '--kappa',
        type=float,
        default=0.1,
        help='Stickiness parameter for behavioral bouts (higher = longer bouts)'
    )
    parser.add_argument(
        '--arhmm-iters',
        type=int,
        default=10,
        help='Number of AR-HMM iterations'
    )
    parser.add_argument(
        '--full-model-iters',
        type=int,
        default=10,
        help='Number of full model iterations'
    )
    parser.add_argument(
        '--mixed-map-iters',
        type=int,
        default=8,
        help='Mixed map iterations for GPU memory management (higher = less memory, slower)'
    )

    # GPU configuration
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for training'
    )

    # Optional configurations
    parser.add_argument(
        '--anterior-bodyparts',
        type=str,
        nargs='+',
        default=["BASE_NECK_INDEX"],
        help='Anterior bodyparts for orientation'
    )
    parser.add_argument(
        '--posterior-bodyparts',
        type=str,
        nargs='+',
        default=["BASE_TAIL_INDEX"],
        help='Posterior bodyparts for orientation'
    )
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip generating plots and movies (faster completion)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate setup without running training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config/default.yml)'
    )

    return parser.parse_args()


def initialize_project(project_path: pathlib.Path, video_dir: str,
                       bodyparts: List[str], skeleton: List[List[str]],
                       anterior_bodyparts: List[str], posterior_bodyparts: List[str],
                       logger: logging.Logger):
    """Initialize KeyPoint-MoSeq project with configurations."""
    logger.info(f"Initializing project at: {project_path}")

    # Setup project (always use overwrite=True to handle incomplete initializations)
    kpms.setup_project(
        project_path,
        video_dir=video_dir,
        bodyparts=bodyparts,
        skeleton=skeleton,
        overwrite=True
    )

    # Update configuration
    kpms.update_config(
        project_path,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        use_bodyparts=bodyparts,
    )

    logger.info("Project initialized successfully")


def main():
    """Main training pipeline."""
    args = parse_args()

    # Convert paths to pathlib objects
    project_path = pathlib.Path(args.project_path)
    pose_dir = args.pose_dir
    video_dir = args.video_dir

    # Create project directory
    project_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = project_path / "logs"
    set_up_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Load configuration and merge with CLI arguments
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config or 'config/default.yml'}")
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {e}")
        logger.warning("Using CLI arguments only")
        config = {}

    # Merge config with CLI arguments (CLI overrides config)
    config = merge_config_with_args(config, args)

    # Save training hyperparameters to separate file (KPMS uses config.yml for its own config)
    config_save_path = project_path / "training_params.yml"
    save_config(config, config_save_path)
    logger.info(f"Saved training parameters to: {config_save_path}")

    # Extract values from merged config
    kappa = config.get('kappa', args.kappa)
    arhmm_iters = config.get('arhmm_iters', args.arhmm_iters)
    full_model_iters = config.get('full_model_iters', args.full_model_iters)
    mixed_map_iters = config.get('mixed_map_iters', args.mixed_map_iters)
    num_gpus = config.get('num_gpus', args.num_gpus)
    bodyparts = config.get('bodyparts', [])
    skeleton = config.get('skeleton', [])
    anterior_bodyparts = config.get('anterior_bodyparts', args.anterior_bodyparts)
    posterior_bodyparts = config.get('posterior_bodyparts', args.posterior_bodyparts)

    # Log configuration
    logger.info("="*80)
    logger.info("KeyPoint-MoSeq Training Script")
    logger.info("="*80)
    logger.info(f"JAX device: {jax.devices()[0].platform}")
    logger.info(f"Number of devices: {len(jax.devices())}")
    logger.info(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    logger.info(f"XLA_PYTHON_CLIENT_ALLOCATOR: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")
    logger.info(f"Pose directory: {pose_dir}")
    logger.info(f"Project path: {project_path}")
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Kappa: {kappa}")
    logger.info(f"AR-HMM iterations: {arhmm_iters}")
    logger.info(f"Full model iterations: {full_model_iters}")
    logger.info(f"Mixed map iterations: {mixed_map_iters}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info("="*80)

    # Configure GPU memory management
    logger.info(f"Setting mixed_map_iters to {mixed_map_iters}")
    set_mixed_map_iters(mixed_map_iters)

    # Configure multi-GPU if requested
    if num_gpus > 1:
        logger.info(f"Configuring {num_gpus} GPUs for distributed training")
        set_mixed_map_gpus(num_gpus)

    # Check GPU usage
    if jax.devices()[0].platform != 'cpu':
        logger.info("Initial GPU status:")
        print_gpu_usage()

    # Initialize project only if not already initialized
    config_file = project_path / "config.yml"
    if config_file.exists():
        logger.info(f"Project already initialized at {project_path}, skipping setup")
    else:
        logger.info(f"Initializing new project at {project_path}")
        initialize_project(
            project_path, video_dir, bodyparts, skeleton,
            anterior_bodyparts, posterior_bodyparts, logger
        )

    # Configuration function
    config_kpms = lambda: kpms.load_config(project_path)

    if args.dry_run:
        logger.info("Dry run mode - skipping training")
        logger.info("Project initialization successful!")
        print("Dry run completed successfully. Project is ready for training.")
        return

    try:
        # Step 1: Load and format data
        logger.info("Step 1/4: Loading and formatting data...")
        data, metadata, coordinates = load_and_format_data(pose_dir, project_path)

        # Validate data quality
        logger.info("Validating data quality...")
        quality_report = validate_data_quality(
            coordinates,
            metadata.get('confidences', {})
        )
        for filename, metrics in quality_report.items():
            logger.info(f"  {filename}: {metrics['total_frames']} frames, "
                       f"mean confidence: {metrics['mean_confidence']:.3f}")

        if jax.devices()[0].platform != 'cpu':
            logger.info("GPU status after data loading:")
            print_gpu_usage()

        # Step 2: Perform PCA
        logger.info("Step 2/4: Performing PCA analysis...")
        pca, n_components_90 = perform_pca(data, config_kpms, project_path)

        # Update config with optimal latent dimension
        logger.info(f"Updating latent dimension to {n_components_90} "
                   "(components explaining >90% variance)")
        kpms.update_config(project_path, latent_dim=n_components_90)

        if jax.devices()[0].platform != 'cpu':
            logger.info("GPU status after PCA:")
            print_gpu_usage()

        # Step 3: Fit AR-HMM model
        logger.info(f"Step 3/4: Fitting AR-HMM model (kappa={kappa})...")
        model, model_name, results = fit_and_save_model(
            data, metadata, pca, config_kpms, project_path,
            kappa=kappa,
            arhmm_iters=arhmm_iters,
            full_model_iters=full_model_iters
        )

        logger.info(f"Model saved as: {model_name}")

        # Step 4: Generate visualizations (optional)
        if not args.skip_visualizations:
            logger.info("Step 4/4: Generating visualizations...")
            generate_plots_and_movies(model_name, results, coordinates,
                                     project_path, config_kpms)
        else:
            logger.info("Step 4/4: Skipping visualizations (--skip-visualizations)")

        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {project_path}")
        logger.info(f"Model name: {model_name}")
        logger.info("="*80)

        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {project_path}")
        print(f"Model name: {model_name}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        print(f"\nERROR: Training failed. Check logs at {log_dir}")
        sys.exit(1)


if __name__ == '__main__':
    main()
