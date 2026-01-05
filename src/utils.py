"""Utility functions for KeyPoint-MoSeq behavioral analysis pipeline.

This module contains helper functions for data loading, GPU monitoring,
logging configuration, and config file management.

Usage in Jupyter notebook:
    from src.utils import load_keypoints_pd, print_gpu_usage, set_up_logging, load_config
"""

from typing import Dict, Tuple, Any
import datetime
import glob
import h5py
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import random
import subprocess
import tqdm
import yaml

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
                filepath, header=None, chunksize=chunk_size
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


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads config/default.yml

    Returns:
        Dictionary with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        # Default to config/default.yml in project root
        project_root = pathlib.Path(__file__).parent.parent
        config_path = project_root / "config" / "default.yml"

    config_path = pathlib.Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: pathlib.Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config file
        
    Raises:
        OSError: If file cannot be written
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Custom YAML dumper to preserve formatting similar to default.yml
        class ConfigDumper(yaml.SafeDumper):
            pass
        
        def represent_str(dumper, data):
            # Use quoted strings only for string values that need it
            if '\n' in data or data == '':
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            # Quote strings that look like bodypart names or version strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
        
        def represent_list(dumper, data):
            # Use flow style (inline brackets) for lists of exactly 2 items (skeleton edges)
            if len(data) == 2 and all(isinstance(item, str) for item in data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            # Use block style for other lists
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        ConfigDumper.add_representer(str, represent_str)
        ConfigDumper.add_representer(list, represent_list)

        with open(output_path, 'w') as f:
            # Add header comment
            f.write("# KeyPoint-MoSeq Configuration\n")
            f.write("# Generated subset configuration\n\n")
            
            # Dump the configuration with proper indentation
            yaml_content = yaml.dump(config, Dumper=ConfigDumper, 
                     default_flow_style=False, sort_keys=False, 
                     indent=2, allow_unicode=True)
            
            # Post-process to improve formatting - remove quotes from keys and adjust list indentation
            lines = yaml_content.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Remove quotes from keys (but keep quotes on values)
                if ':' in line and not line.strip().startswith('-'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0].replace('"', '').replace("'", '')
                        value_part = parts[1]
                        line = key_part + ':' + value_part
                
                # Adjust list indentation to match default.yml (add extra 2 spaces)
                if line.startswith('- '):
                    # Top-level list item
                    line = '  ' + line
                elif line.startswith('  - ') and not line.startswith('    '):
                    # Already indented list item, ensure consistent spacing
                    pass
                
                formatted_lines.append(line)
            
            f.write('\n'.join(formatted_lines))
        
        logger.info(f"Saved configuration to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving config to {output_path}: {e}")
        raise


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Merge configuration with command-line arguments.

    CLI arguments override config file values.

    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments (argparse Namespace)

    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()

    # Map CLI args to config keys
    arg_to_config = {
        'kappa': 'kappa',
        'arhmm_iters': 'arhmm_iters',
        'full_model_iters': 'full_model_iters',
        'mixed_map_iters': 'mixed_map_iters',
        'num_gpus': 'num_gpus',
        'pose_version': 'pose_version',
        'anterior_bodyparts': 'anterior_bodyparts',
        'posterior_bodyparts': 'posterior_bodyparts',
    }

    # Override config with CLI arguments
    for arg_name, config_key in arg_to_config.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                merged[config_key] = arg_value
                logger.debug(f"CLI override: {config_key} = {arg_value}")

    return merged


def log_sample_data(
    h5_dir: str,
    csv_dir: str,
    subset_dirs: Dict[str, str],
    subset_configs: Dict[str, Dict[str, Any]] = None,
    pose_version: str = "v6",
    n_samples: int = 3,
    n_keypoints_to_show: int = 3
) -> None:
    """Log sample coordinates from H5, CSV, and subset files for verification.
    
    Randomly samples videos and displays coordinates from the first frame
    to help verify conversion correctness.
    
    Args:
        h5_dir: Directory containing H5 files
        csv_dir: Directory containing 12-keypoint CSV files
        subset_dirs: Dictionary mapping subset names to directories (e.g., {'10k': path, '8k': path})
        subset_configs: Dictionary mapping subset names to their configurations (optional)
        pose_version: Version of pose estimation format ("v2" or "v6", default: "v6")
        n_samples: Number of random videos to sample (default: 3)
        n_keypoints_to_show: Number of keypoints to show per file (default: 3)
    """
    logger.info("="*60)
    logger.info("Sample Data Verification")
    logger.info("="*60)
    
    try:
        # Find H5 files
        h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
        if not h5_files:
            logger.warning(f"No H5 files found in {h5_dir}")
            return
        
        # Randomly sample files
        sample_files = random.sample(h5_files, min(n_samples, len(h5_files)))
        
        for h5_path in sample_files:
            filename = os.path.basename(h5_path)
            csv_filename = filename.replace(".h5", ".csv")
            
            logger.info(f"\n--- Sample: {filename} ---")
            
            # Read H5 file
            try:
                with h5py.File(h5_path, "r") as h5_file:
                    if "poseest" not in h5_file:
                        logger.warning(f"No poseest group in {filename}")
                        continue
                    
                    poseest = h5_file["poseest"]
                    points = poseest["points"]
                    confidence = poseest["confidence"]
                    
                    logger.info(f"H5 file (frame 0, first {n_keypoints_to_show} keypoints):")
                    
                    for i in range(min(n_keypoints_to_show, points.shape[2] if pose_version == "v6" else points.shape[1])):
                        if pose_version == "v6":
                            x = points[0, 0, i, 0]
                            y = points[0, 0, i, 1]
                            conf = confidence[0, 0, i]
                        else:  # v2
                            x = points[0, i, 0]
                            y = points[0, i, 1]
                            conf = confidence[0, i]
                        
                        logger.info(f"  Keypoint {i}: x={x:.2f}, y={y:.2f}, conf={conf:.3f}")
            
            except Exception as e:
                logger.error(f"Error reading H5 file {filename}: {e}")
                continue
            
            # Read CSV file (12 keypoints)
            csv_path = os.path.join(csv_dir, csv_filename)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, header=None)
                    logger.info(f"CSV file (frame 0, first {n_keypoints_to_show} keypoints, after x/y swap):")
                    
                    for i in range(min(n_keypoints_to_show, 12)):
                        x = df.iloc[0, 3*i]
                        y = df.iloc[0, 3*i + 1]
                        conf = df.iloc[0, 3*i + 2]
                        logger.info(f"  Keypoint {i}: x={x:.2f}, y={y:.2f}, conf={conf:.3f}")
                
                except Exception as e:
                    logger.error(f"Error reading CSV file {csv_filename}: {e}")
            else:
                logger.warning(f"CSV file not found: {csv_filename}")
            
            # Read subset files
            if subset_configs is None:
                subset_configs = {}
            
            for subset_name, subset_dir in subset_dirs.items():
                if not os.path.exists(subset_dir):
                    continue
                
                subset_csv_path = os.path.join(subset_dir, csv_filename)
                if os.path.exists(subset_csv_path):
                    try:
                        config = subset_configs.get(subset_name, {})
                        remove_indices = config.get('remove_indices', [])
                        n_keypoints = config.get('n_keypoints', 0)
                        
                        df_subset = pd.read_csv(subset_csv_path, header=None)
                        logger.info(f"{subset_name} subset (frame 0, first {n_keypoints_to_show} keypoints):")
                        logger.info(f"  Removed keypoint indices: {remove_indices}")
                        
                        for i in range(min(n_keypoints_to_show, n_keypoints)):
                            x = df_subset.iloc[0, 3*i]
                            y = df_subset.iloc[0, 3*i + 1]
                            conf = df_subset.iloc[0, 3*i + 2]
                            logger.info(f"  Keypoint {i}: x={x:.2f}, y={y:.2f}, conf={conf:.3f}")
                    
                    except Exception as e:
                        logger.error(f"Error reading {subset_name} subset file: {e}")
                else:
                    logger.warning(f"{subset_name} subset file not found: {csv_filename}")
        
        logger.info("="*60)
        logger.info("Sample data verification complete")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Error in log_sample_data: {e}")


def generate_subset_config(
    source_config_path: str,
    output_config_path: str,
    remove_indices: list,
    subset_name: str
) -> None:
    """Generate config file for keypoint subset.

    Creates a new config file with filtered bodyparts and skeleton connections
    based on removed keypoint indices.

    Args:
        source_config_path: Path to source config file (default 12-keypoint config)
        output_config_path: Path to save subset config file
        remove_indices: List of keypoint indices to remove (0-based)
        subset_name: Name of subset (e.g., '10k', '8k')

    Raises:
        FileNotFoundError: If source config doesn't exist
        ValueError: If config format is invalid
    """
    try:
        # Load source config
        source_config = load_config(source_config_path)

        if 'bodyparts' not in source_config:
            raise ValueError("Source config missing 'bodyparts' field")

        bodyparts = source_config['bodyparts']
        n_keypoints = len(bodyparts)

        # Filter bodyparts (remove specified indices)
        filtered_bodyparts = [
            bp for i, bp in enumerate(bodyparts) if i not in remove_indices
        ]

        logger.info(
            f"Filtered bodyparts: {n_keypoints} → {len(filtered_bodyparts)}")

        # Build name-to-index mapping for original bodyparts
        name_to_idx = {bp: i for i, bp in enumerate(bodyparts)}

        # Filter skeleton connections (remove edges with removed bodyparts)
        if 'skeleton' in source_config:
            original_skeleton = source_config['skeleton']
            filtered_skeleton = []

            for edge in original_skeleton:
                if len(edge) != 2:
                    continue

                bp1, bp2 = edge
                idx1 = name_to_idx.get(bp1)
                idx2 = name_to_idx.get(bp2)

                # Keep edge only if both bodyparts are retained
                if idx1 is not None and idx2 is not None:
                    if idx1 not in remove_indices and idx2 not in remove_indices:
                        filtered_skeleton.append(edge)

            logger.info(
                f"Filtered skeleton: {len(original_skeleton)} → {len(filtered_skeleton)} edges")
        else:
            filtered_skeleton = []

        # Filter anterior/posterior bodyparts if they exist
        filtered_anterior = []
        if 'anterior_bodyparts' in source_config:
            filtered_anterior = [
                bp for bp in source_config['anterior_bodyparts']
                if bp in filtered_bodyparts
            ]

        filtered_posterior = []
        if 'posterior_bodyparts' in source_config:
            filtered_posterior = [
                bp for bp in source_config['posterior_bodyparts']
                if bp in filtered_bodyparts
            ]

        # Create new config
        subset_config = source_config.copy()
        subset_config['bodyparts'] = filtered_bodyparts
        subset_config['skeleton'] = filtered_skeleton

        if filtered_anterior:
            subset_config['anterior_bodyparts'] = filtered_anterior
        if filtered_posterior:
            subset_config['posterior_bodyparts'] = filtered_posterior

        # Add metadata
        subset_config['_subset_info'] = {
            'name': subset_name,
            'n_keypoints': len(filtered_bodyparts),
            'removed_indices': remove_indices,
            'source_config': str(source_config_path)
        }

        # Save subset config
        output_path = pathlib.Path(output_config_path)
        save_config(subset_config, output_path)

        logger.info(
            f"Generated {subset_name} config: {len(filtered_bodyparts)} keypoints")

    except Exception as e:
        logger.error(f"Failed to generate subset config: {e}")
        raise
