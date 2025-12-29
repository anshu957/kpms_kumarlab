#!/usr/bin/env python
"""
Pose Data Preprocessing Script

Converts H5 pose estimation files to CSV format for KPMS analysis.
Run this once before training to convert raw pose data.

Usage:
    python preprocess_poses.py --input data/exp1/h5/ --output data/exp1/csv/

Example:
    python preprocess_poses.py \
        --input data/experiment_1/raw_h5/ \
        --output data/experiment_1/processed_csv/ \
        --pose-version v6 \
        --validate \
        --n-jobs 8
"""

import sys
import pathlib
import argparse
import logging

# Add project root to Python path
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.utils import load_config, set_up_logging, generate_subset_config, log_sample_data
from src.preprocessing import h5_to_csv_poses_parallel, create_keypoint_subset, SUBSET_CONFIGS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert H5 pose files to CSV format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing H5 pose files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for CSV files'
    )

    # Optional arguments
    parser.add_argument(
        '--pose-version',
        type=str,
        default=None,
        choices=['v2', 'v6'],
        help='JABS pose version (overrides config)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate converted CSV files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing CSV files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config/default.yml)'
    )
    parser.add_argument(
        '--subsets',
        type=str,
        choices=['none', '10k', '8k', 'both'],
        default='none',
        help='Generate keypoint subsets (10k, 8k, or both)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=3,
        help='Number of random videos to sample for logging (default: 3)'
    )

    return parser.parse_args()


def main():
    """Main preprocessing pipeline."""
    args = parse_args()

    # Setup logging
    log_dir = pathlib.Path(args.output) / "logs"
    set_up_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Log configuration
    logger.info("="*60)
    logger.info("Pose Data Preprocessing")
    logger.info("="*60)
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config or 'config/default.yml'}")
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {e}")
        logger.warning("Using default parameters")
        config = {}

    # CLI arguments override config
    pose_version = args.pose_version or config.get('pose_version', 'v6')
    validate_output = args.validate or config.get('validate_output', True)
    overwrite = args.overwrite or config.get('overwrite', False)

    logger.info(f"Pose version: {pose_version}")
    logger.info(f"Validate output: {validate_output}")
    logger.info(f"Overwrite existing: {overwrite}")
    logger.info(f"Parallel workers: {args.n_jobs or 'auto (CPU count)'}")
    logger.info(f"Sample videos for logging: {args.n_samples}")
    logger.info("="*60)

    # Convert H5 to CSV using parallel processing
    try:
        logger.info("Starting parallel conversion...")
        converted_files = h5_to_csv_poses_parallel(
            folder_path=args.input,
            dest_path=args.output,
            file_pattern="*.h5",
            pose_version=pose_version,
            overwrite=overwrite,
            validate_output=validate_output,
            n_jobs=args.n_jobs
        )

        logger.info("="*60)
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Converted {len(converted_files)} files")
        logger.info(f"Output directory: {args.output}")
        logger.info("="*60)

        print(f"\nConversion completed!")
        print(f"Converted {len(converted_files)} files")
        print(f"Output: {args.output}")

        # Generate keypoint subsets if requested
        if args.subsets != 'none':
            logger.info("="*60)
            logger.info("Generating keypoint subsets...")
            logger.info("="*60)

            # Determine which subsets to create
            subsets_to_create = []
            if args.subsets == 'both':
                subsets_to_create = ['10k', '8k']
            else:
                subsets_to_create = [args.subsets]

            # Get project root for config generation
            project_root = pathlib.Path(__file__).parent.parent
            default_config_path = project_root / "config" / "default.yml"

            for subset_type in subsets_to_create:
                try:
                    logger.info(f"\nCreating {subset_type} subset...")

                    # Create subset CSV files
                    output_base = pathlib.Path(args.output)
                    subset_dir = output_base.parent / f"{output_base.name}_{subset_type}"

                    subset_files = create_keypoint_subset(
                        source_dir=args.output,
                        dest_dir=str(subset_dir),
                        subset_type=subset_type,
                        overwrite=overwrite
                    )

                    logger.info(f"Created {len(subset_files)} {subset_type} CSV files")
                    print(f"  {subset_type}: {len(subset_files)} files → {subset_dir}")

                    # Generate corresponding config file
                    config_output_path = project_root / "config" / f"config_{subset_type}.yml"
                    subset_config = SUBSET_CONFIGS[subset_type]

                    generate_subset_config(
                        source_config_path=str(default_config_path),
                        output_config_path=str(config_output_path),
                        remove_indices=subset_config['remove_indices'],
                        subset_name=subset_type
                    )

                    logger.info(f"Generated config file: {config_output_path}")
                    print(f"  Config: {config_output_path}")

                except Exception as e:
                    logger.error(f"Failed to create {subset_type} subset: {e}", exc_info=True)
                    print(f"  ERROR: Failed to create {subset_type} subset")
                    continue

            logger.info("="*60)
            logger.info("Subset generation completed!")
            logger.info("="*60)
            print("\nSubset generation completed!")

        # Log sample data for verification
        if args.n_samples > 0:
            logger.info("\n" + "="*60)
            logger.info("Logging sample data for verification...")
            logger.info("="*60)
            
            # Collect subset directories
            subset_dirs = {}
            if args.subsets != 'none':
                output_base = pathlib.Path(args.output)
                subsets_to_check = []
                if args.subsets == 'both':
                    subsets_to_check = ['10k', '8k']
                else:
                    subsets_to_check = [args.subsets]
                
                for subset_type in subsets_to_check:
                    subset_dir = output_base.parent / f"{output_base.name}_{subset_type}"
                    if subset_dir.exists():
                        subset_dirs[subset_type] = str(subset_dir)
            
            log_sample_data(
                h5_dir=args.input,
                csv_dir=args.output,
                subset_dirs=subset_dirs,
                subset_configs=SUBSET_CONFIGS,
                pose_version=pose_version,
                n_samples=args.n_samples
            )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        print(f"\nERROR: Preprocessing failed. Check logs at {log_dir}")
        sys.exit(1)


if __name__ == '__main__':
    main()
