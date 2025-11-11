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
        --validate
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
from src.utils import load_config, set_up_logging
from src.preprocessing import h5_to_csv_poses


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
    logger.info("="*60)

    # Convert H5 to CSV
    try:
        logger.info("Starting conversion...")
        converted_files = h5_to_csv_poses(
            folder_path=args.input,
            dest_path=args.output,
            file_pattern="*.h5",
            pose_version=pose_version,
            overwrite=overwrite,
            validate_output=validate_output
        )

        logger.info("="*60)
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Converted {len(converted_files)} files")
        logger.info(f"Output directory: {args.output}")
        logger.info("="*60)

        print(f"\nConversion completed!")
        print(f"Converted {len(converted_files)} files")
        print(f"Output: {args.output}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        print(f"\nERROR: Preprocessing failed. Check logs at {log_dir}")
        sys.exit(1)


if __name__ == '__main__':
    main()
