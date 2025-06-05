"""Utility functions for KeyPoint-MoSeq behavioral analysis pipeline.

This module contains helper functions for data loading, GPU monitoring,
and logging configuration.

Usage in Jupyter notebook:
    from src.utils import load_keypoints_pd, print_gpu_usage, set_up_logging
"""

from typing import Dict, Tuple
import glob
import numpy as np
import os
import tqdm
import pandas as pd
import subprocess
import datetime
import logging
import pathlib

logger = logging.getLogger(__name__)


def load_keypoints_pd(
    dir_name: str,
    file_pattern: str = "*.csv",
    chunk_size: int = 1000
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load keypoint data from CSV files in a directory.

    Args:
        dir_name: Directory containing CSV files
        file_pattern: Glob pattern for matching files (default: "*.csv")
        chunk_size: Size of chunks for reading large CSV files (default: 1000)

    Returns:
        Tuple containing (coordinates, confidences) dictionaries
    """
    if not os.path.exists(dir_name):
        raise FileNotFoundError(f"Directory not found: {dir_name}")

    keypoint_files = glob.glob(os.path.join(dir_name, file_pattern))

    if not keypoint_files:
        raise ValueError(
            f"No CSV files found in {dir_name} matching pattern {file_pattern}")

    logger.info(f"Found {len(keypoint_files)} CSV files to process")

    coordinates = {}
    confidences = {}

    for filepath in tqdm.tqdm(keypoint_files, desc="Loading keypoint files"):
        try:
            # Read the CSV file in chunks to handle large files
            chunk_iterator = pd.read_csv(
                filepath, skiprows=1, header=None, chunksize=chunk_size
            )
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            continue

        name = os.path.basename(filepath)

        # Initialize lists to accumulate results from chunks
        coords_list = []
        confs_list = []

        for chunk in chunk_iterator:
            try:
                data = chunk.values.astype(np.float64)

                # Reshape data: (n_frames, n_keypoints, 3)
                # Expect format: x1, y1, conf1, x2, y2, conf2, ...
                if data.shape[1] % 3 != 0:
                    logger.error(
                        f"Invalid data format in {filepath}: expected columns divisible by 3")
                    break

                n_keypoints = data.shape[1] // 3
                data = data.reshape(data.shape[0], n_keypoints, 3)

                # Extract coordinates and swap x and y (JABS format specific)
                coords = data[:, :, :2][:, :, ::-1]  # Swap x and y
                coords_list.append(coords)

                # Extract confidences
                confs = data[:, :, 2]
                confs_list.append(confs)

            except Exception as e:
                logger.error(f"Error processing chunk in {filepath}: {e}")
                continue

        # Concatenate results from all chunks
        if coords_list:
            coordinates[name] = np.concatenate(coords_list, axis=0)
            confidences[name] = np.concatenate(confs_list, axis=0)
            logger.debug(f"Loaded {name}: {coordinates[name].shape[0]} frames")
        else:
            logger.warning(f"No valid data found in {filepath}")

    if not coordinates:
        raise ValueError(f"No valid keypoint data loaded from {dir_name}")

    total_files = len(coordinates)
    total_frames = sum(coord.shape[0] for coord in coordinates.values())
    logger.info(
        f"Successfully loaded {total_files} files with {total_frames} total frames")

    return coordinates, confidences


def print_gpu_usage() -> None:
    """Print current GPU usage statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.info("GPU Usage Information:")
            logger.info(result.stdout)
        else:
            logger.error(
                f"nvidia-smi failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi command timed out")
    except FileNotFoundError:
        logger.warning(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except Exception as e:
        logger.error(f"Unexpected error getting GPU usage: {e}")


def set_up_logging(
    log_dir: pathlib.Path,
    log_level: str = "INFO"
) -> None:
    """Set up logging configuration for the analysis pipeline.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_dir / f"kpms_analysis_{timestamp}.log"

    # Set up basic logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ],
        force=True
    )

    logger.info(f"Logging configured. Log file: {log_filename}")


def validate_data_quality(
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    min_confidence: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """Validate the quality of loaded keypoint data.

    Args:
        coordinates: Dictionary mapping filenames to coordinate arrays
        confidences: Dictionary mapping filenames to confidence arrays
        min_confidence: Minimum acceptable confidence threshold (default: 0.1)

    Returns:
        Dictionary with quality metrics for each file
    """
    quality_report = {}

    for filename in coordinates.keys():
        coords = coordinates[filename]
        confs = confidences[filename]

        # Calculate basic metrics
        nan_coords = np.isnan(coords).sum()
        nan_confs = np.isnan(confs).sum()
        low_conf_points = (confs < min_confidence).sum()

        nan_coord_pct = (nan_coords / coords.size) * \
            100 if coords.size > 0 else 0
        nan_conf_pct = (nan_confs / confs.size) * 100 if confs.size > 0 else 0
        low_conf_pct = (low_conf_points / confs.size) * \
            100 if confs.size > 0 else 0

        mean_confidence = np.nanmean(confs)

        quality_report[filename] = {
            'total_frames': coords.shape[0],
            'total_keypoints': coords.shape[1],
            'nan_coordinates_pct': nan_coord_pct,
            'nan_confidence_pct': nan_conf_pct,
            'low_confidence_pct': low_conf_pct,
            'mean_confidence': mean_confidence
        }

        # Log warnings for poor quality data
        if nan_coord_pct > 50:
            logger.warning(
                f"{filename}: High NaN coordinates percentage: {nan_coord_pct:.2f}%")
        if nan_conf_pct > 50:
            logger.warning(
                f"{filename}: High NaN confidence percentage: {nan_conf_pct:.2f}%")
        if low_conf_pct > 50:
            logger.warning(
                f"{filename}: High low-confidence percentage: {low_conf_pct:.2f}%")
        if mean_confidence < min_confidence:
            logger.warning(
                f"{filename}: Low mean confidence: {mean_confidence:.3f}")

    return quality_report
