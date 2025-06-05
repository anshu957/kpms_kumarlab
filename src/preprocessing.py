"""Data preprocessing functions for KeyPoint-MoSeq behavioral analysis pipeline.

This module contains functions for converting and preprocessing pose estimation data,
particularly for converting H5 format files to CSV format compatible with KPMS.
"""

from typing import Optional, List
import pandas as pd
import h5py
import numpy as np
import os
import logging
import pathlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


def h5_to_csv_poses(
    folder_path: str,
    dest_path: str,
    file_pattern: str = "*.h5",
    pose_version: str = "v6",
    overwrite: bool = False,
    validate_output: bool = True
) -> List[str]:
    """Convert H5 pose estimation files to CSV format for KPMS analysis.

    Args:
        folder_path: Directory containing H5 pose files
        dest_path: Destination directory for CSV files
        file_pattern: Glob pattern for matching H5 files (default: "*.h5")
        pose_version: Version of pose estimation format ("v2" or "v6", default: "v6")
        overwrite: Whether to overwrite existing CSV files (default: False)
        validate_output: Whether to validate converted CSV files (default: True)

    Returns:
        List of successfully converted file paths

    Raises:
        FileNotFoundError: If source directory doesn't exist
        ValueError: If invalid pose version or no H5 files found
        RuntimeError: If conversion fails
    """
    try:
        # Validate inputs
        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Source directory not found: {folder_path}")

        if pose_version not in ["v2", "v6"]:
            raise ValueError(
                f"Invalid pose version: {pose_version}. Must be 'v2' or 'v6'")

        # Create destination directory
        dest_path_obj = pathlib.Path(dest_path)
        dest_path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created destination directory: {dest_path}")

        # Find H5 files
        import glob
        h5_files = glob.glob(os.path.join(folder_path, file_pattern))

        if not h5_files:
            raise ValueError(
                f"No H5 files found in {folder_path} matching pattern {file_pattern}")

        logger.info(f"Found {len(h5_files)} H5 files to convert")

        converted_files = []
        failed_files = []

        for file_path in tqdm(h5_files, desc="Converting H5 to CSV"):
            try:
                filename = os.path.basename(file_path)
                csv_filename = filename.replace(".h5", ".csv")
                csv_path = dest_path_obj / csv_filename

                # Check if file already exists and overwrite flag
                if csv_path.exists() and not overwrite:
                    logger.info(f"Skipping existing file: {csv_filename}")
                    converted_files.append(str(csv_path))
                    continue

                # Convert single file
                success = _convert_single_h5_file(
                    file_path, csv_path, pose_version, validate_output
                )

                if success:
                    converted_files.append(str(csv_path))
                    logger.debug(f"Successfully converted: {filename}")
                else:
                    failed_files.append(filename)

            except Exception as e:
                logger.error(f"Failed to convert {filename}: {e}")
                failed_files.append(filename)
                continue

        # Summary
        success_count = len(converted_files)
        total_count = len(h5_files)
        logger.info(
            f"Conversion complete: {success_count}/{total_count} files successful")

        if failed_files:
            logger.warning(
                f"Failed to convert {len(failed_files)} files: {failed_files}")

        if success_count == 0:
            raise RuntimeError("No files were successfully converted")

        return converted_files

    except Exception as e:
        logger.error(f"H5 to CSV conversion failed: {e}")
        raise RuntimeError(f"H5 to CSV conversion failed: {e}") from e


def _convert_single_h5_file(
    h5_path: str,
    csv_path: pathlib.Path,
    pose_version: str,
    validate_output: bool = True
) -> bool:
    """Convert a single H5 pose file to CSV format.

    Args:
        h5_path: Path to the input H5 file
        csv_path: Path for the output CSV file
        pose_version: Version of pose estimation format ("v2" or "v6")
        validate_output: Whether to validate the converted CSV file

    Returns:
        True if conversion successful, False otherwise
    """
    h5_file = None
    try:
        # Open H5 file
        try:
            h5_file = h5py.File(h5_path, "r")
        except Exception as e:
            logger.error(f"Cannot open H5 file {h5_path}: {e}")
            return False

        # Validate H5 file structure
        if "poseest" not in h5_file:
            logger.error(
                f"Invalid H5 structure in {h5_path}: missing 'poseest' group")
            return False

        poseest_group = h5_file["poseest"]

        if "points" not in poseest_group or "confidence" not in poseest_group:
            logger.error(
                f"Invalid H5 structure in {h5_path}: missing required datasets")
            return False

        points = poseest_group["points"]
        confidence = poseest_group["confidence"]

        # Get data dimensions based on pose version
        if pose_version == "v6":
            n_frames = points.shape[0]
            n_keypoints = points.shape[2]
            # Shape should be (n_frames, 1, n_keypoints, 2)
            if len(points.shape) != 4 or points.shape[1] != 1 or points.shape[3] != 2:
                logger.error(
                    f"Invalid v6 points shape in {h5_path}: {points.shape}")
                return False
        elif pose_version == "v2":
            n_frames = points.shape[0]
            n_keypoints = points.shape[1]
            # Shape should be (n_frames, n_keypoints, 2)
            if len(points.shape) != 3 or points.shape[2] != 2:
                logger.error(
                    f"Invalid v2 points shape in {h5_path}: {points.shape}")
                return False
        else:
            logger.error(f"Unsupported pose version: {pose_version}")
            return False

        # Validate confidence shape
        expected_conf_shape = (n_frames, 1, n_keypoints) if pose_version == "v6" else (
            n_frames, n_keypoints)
        if confidence.shape != expected_conf_shape:
            logger.error(
                f"Invalid confidence shape in {h5_path}: {confidence.shape}, expected: {expected_conf_shape}")
            return False

        logger.debug(
            f"Processing {n_frames} frames with {n_keypoints} keypoints")

        # Initialize output array: (n_frames, n_keypoints * 3)
        # Format: x1, y1, conf1, x2, y2, conf2, ...
        output_array = np.zeros((n_frames, n_keypoints * 3))

        # Extract and reformat data
        for i in range(n_keypoints):
            if pose_version == "v6":
                # Extract coordinates (swap x and y for JABS format)
                x_coords = points[:, 0, i, 1]  # y becomes x
                y_coords = points[:, 0, i, 0]  # x becomes y
                confs = confidence[:, 0, i]
            else:  # v2
                # Extract coordinates (swap x and y for JABS format)
                x_coords = points[:, i, 1]  # y becomes x
                y_coords = points[:, i, 0]  # x becomes y
                confs = confidence[:, i]

            # Store in output array
            output_array[:, 3 * i] = x_coords
            output_array[:, 3 * i + 1] = y_coords
            output_array[:, 3 * i + 2] = confs

        # Validate data quality
        if np.all(np.isnan(output_array)):
            logger.warning(f"All data is NaN in {h5_path}")

        nan_percentage = (np.isnan(output_array).sum() /
                          output_array.size) * 100
        if nan_percentage > 90:
            logger.warning(
                f"High NaN percentage ({nan_percentage:.1f}%) in {h5_path}")

        # Save to CSV
        df = pd.DataFrame(output_array)
        df.to_csv(csv_path, index=False, header=False)

        # Validate output if requested
        if validate_output:
            if not _validate_csv_output(csv_path, n_frames, n_keypoints):
                logger.error(f"CSV validation failed for {csv_path}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error converting {h5_path}: {e}")
        return False

    finally:
        if h5_file is not None:
            h5_file.close()


def _validate_csv_output(
    csv_path: pathlib.Path,
    expected_frames: int,
    expected_keypoints: int
) -> bool:
    """Validate the converted CSV file.

    Args:
        csv_path: Path to the CSV file to validate
        expected_frames: Expected number of frames
        expected_keypoints: Expected number of keypoints

    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, header=None)

        # Check dimensions
        if df.shape[0] != expected_frames:
            logger.error(
                f"Frame count mismatch in {csv_path}: got {df.shape[0]}, expected {expected_frames}")
            return False

        if df.shape[1] != expected_keypoints * 3:
            logger.error(
                f"Column count mismatch in {csv_path}: got {df.shape[1]}, expected {expected_keypoints * 3}")
            return False

        # Check for completely empty data
        if df.isnull().all().all():
            logger.error(f"CSV file {csv_path} contains only null values")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating CSV file {csv_path}: {e}")
        return False


def batch_convert_poses(
    source_directories: List[str],
    dest_directory: str,
    pose_version: str = "v6",
    preserve_structure: bool = True
) -> dict:
    """Convert poses from multiple source directories.

    Args:
        source_directories: List of source directories containing H5 files
        dest_directory: Destination directory for all converted CSV files
        pose_version: Version of pose estimation format (default: "v6")
        preserve_structure: Whether to preserve source directory structure (default: True)

    Returns:
        Dictionary with conversion results for each source directory

    Raises:
        ValueError: If no source directories provided
    """
    if not source_directories:
        raise ValueError("No source directories provided")

    results = {}

    for source_dir in source_directories:
        try:
            logger.info(f"Processing source directory: {source_dir}")

            if preserve_structure:
                # Create subdirectory in destination
                subdir_name = os.path.basename(source_dir.rstrip(os.sep))
                dest_path = os.path.join(dest_directory, subdir_name)
            else:
                dest_path = dest_directory

            converted_files = h5_to_csv_poses(
                source_dir, dest_path, pose_version=pose_version
            )

            results[source_dir] = {
                'status': 'success',
                'converted_files': converted_files,
                'count': len(converted_files)
            }

        except Exception as e:
            logger.error(f"Failed to process {source_dir}: {e}")
            results[source_dir] = {
                'status': 'failed',
                'error': str(e),
                'count': 0
            }

    # Summary
    total_files = sum(r.get('count', 0) for r in results.values())
    successful_dirs = sum(1 for r in results.values()
                          if r['status'] == 'success')

    logger.info(
        f"Batch conversion complete: {successful_dirs}/{len(source_directories)} directories, {total_files} total files")

    return results


def get_pose_file_info(h5_path: str) -> Optional[dict]:
    """Get information about a pose estimation H5 file.

    Args:
        h5_path: Path to the H5 file

    Returns:
        Dictionary with file information, or None if file cannot be read
    """
    try:
        with h5py.File(h5_path, "r") as h5_file:
            if "poseest" not in h5_file:
                return None

            poseest = h5_file["poseest"]

            if "points" not in poseest or "confidence" not in poseest:
                return None

            points_shape = poseest["points"].shape
            conf_shape = poseest["confidence"].shape

            # Determine pose version based on shape
            if len(points_shape) == 4 and points_shape[1] == 1:
                pose_version = "v6"
                n_frames = points_shape[0]
                n_keypoints = points_shape[2]
            elif len(points_shape) == 3:
                pose_version = "v2"
                n_frames = points_shape[0]
                n_keypoints = points_shape[1]
            else:
                pose_version = "unknown"
                n_frames = points_shape[0] if len(points_shape) > 0 else 0
                n_keypoints = 0

            info = {
                'filename': os.path.basename(h5_path),
                'pose_version': pose_version,
                'n_frames': n_frames,
                'n_keypoints': n_keypoints,
                'points_shape': points_shape,
                'confidence_shape': conf_shape,
                'file_size_mb': os.path.getsize(h5_path) / (1024 * 1024)
            }

            return info

    except Exception as e:
        logger.error(f"Error reading pose file info from {h5_path}: {e}")
        return None
