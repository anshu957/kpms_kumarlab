"""Core methods for KeyPoint-MoSeq behavioral analysis pipeline.

This module contains the main pipeline functions for loading data, performing PCA,
fitting AR-HMM models, and generating visualizations.

Usage in Jupyter notebook:
    from src.methods import load_and_format_data, perform_pca, fit_and_save_model
"""

from typing import Tuple, Dict, Any, Callable
import keypoint_moseq as kpms
import logging
import numpy as np
import pathlib
from src.utils import load_keypoints_pd, print_gpu_usage

logger = logging.getLogger(__name__)


def load_and_format_data(
    pose_dir: str,
    project_path: pathlib.Path
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """Load keypoints and format data for KPMS analysis.

    Args:
        pose_dir: Directory containing pose CSV files
        project_path: Path to the project directory

    Returns:
        Tuple containing:
            - data: Formatted data dictionary for KPMS
            - metadata: Metadata dictionary
            - coordinates: Raw coordinates dictionary

    Raises:
        FileNotFoundError: If pose directory doesn't exist
        ValueError: If no valid pose files found
        RuntimeError: If data formatting fails
    """
    try:
        logger.info(f"Loading keypoints from directory: {pose_dir}")

        if not pathlib.Path(pose_dir).exists():
            raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

        coordinates, confidences = load_keypoints_pd(pose_dir)

        if not coordinates:
            raise ValueError(f"No valid pose files found in {pose_dir}")

        total_frames = sum(coord.shape[0] for coord in coordinates.values())
        logger.info(f"Total number of frames: {total_frames}")

        # Count frames with NaN confidence
        nan_frames = sum(np.isnan(conf).sum() for conf in confidences.values())
        nan_percentage = (nan_frames / total_frames) * \
            100 if total_frames > 0 else 0
        logger.info(
            f"Number of frames with NaN confidence: {nan_frames} ({nan_percentage:.2f}%)")

        if nan_percentage > 50:
            logger.warning(
                f"High percentage of NaN confidence values: {nan_percentage:.2f}%")

        def config_func() -> Dict[str, Any]:
            """Load configuration from project path."""
            return kpms.load_config(project_path)

        logger.info("Formatting data for KPMS analysis...")
        data, metadata = kpms.format_data(
            coordinates, confidences, **config_func()
        )

        logger.info("Data formatting completed successfully")
        return data, metadata, coordinates

    except Exception as e:
        logger.error(f"Failed to load and format data: {e}")
        raise RuntimeError(f"Data loading failed: {e}") from e


def perform_pca(
    data: Dict[str, Any],
    config_func: Callable[[], Dict[str, Any]],
    project_path: pathlib.Path
) -> Tuple[Any, int]:
    """Perform Principal Component Analysis on pose data.

    Args:
        data: Formatted data dictionary from load_and_format_data
        config_func: Function that returns configuration dictionary
        project_path: Path to the project directory

    Returns:
        Tuple containing:
            - pca: Fitted PCA object
            - n_components_90: Number of components explaining >90% variance
    """
    logger.info("Starting PCA analysis...")

    # Perform PCA
    pca = kpms.fit_pca(**data, **config_func())
    logger.info("PCA fitting completed")

    # Calculate number of components for >90% variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = int(np.where(cumsum_var >= 0.9)[0][0] + 1)

    logger.info(
        f"Number of PCA components explaining >90% variance: {n_components_90}")

    # Save the PCA object
    logger.info("Saving PCA model...")
    kpms.save_pca(pca, project_path)

    # Generate analysis and plots
    logger.info("Generating PCA analysis plots...")
    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_path, savefig=True)
    kpms.plot_pcs(pca, project_dir=project_path, **config_func(), savefig=True)

    logger.info("PCA analysis completed successfully")
    return pca, n_components_90


def fit_and_save_model(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    pca: Any,
    config_func: Callable[[], Dict[str, Any]],
    project_path: pathlib.Path,
    kappa: float = 0.1,
    arhmm_iters: int = 10,
    full_model_iters: int = 10
) -> Tuple[Any, str, Dict[str, Any]]:
    """Fit AR-HMM model and save results.

    Args:
        data: Formatted data dictionary
        metadata: Metadata dictionary
        pca: Fitted PCA object
        config_func: Function that returns configuration dictionary
        project_path: Path to the project directory
        kappa: Stickiness parameter for behavioral bouts (default: 0.1)
        arhmm_iters: Number of AR-HMM iterations (default: 10)
        full_model_iters: Number of full model iterations (default: 10)

    Returns:
        Tuple containing:
            - model: Fitted model object
            - model_name: Name of the saved model
            - results: Extracted results dictionary

    Raises:
        ValueError: If invalid parameters provided
        RuntimeError: If model fitting fails
    """
    try:
        # Validate parameters
        if kappa <= 0:
            raise ValueError(f"Kappa must be positive, got: {kappa}")
        if arhmm_iters <= 0 or full_model_iters <= 0:
            raise ValueError("Number of iterations must be positive")

        logger.info(f"Initializing model with kappa={kappa}")
        logger.info(
            f"AR-HMM iterations: {arhmm_iters}, Full model iterations: {full_model_iters}")

        # Initialize model
        model = kpms.init_model(data, pca=pca, **config_func())

        # Update kappa and fit AR-HMM
        model = kpms.update_hypparams(model, kappa=kappa)
        logger.info("Starting AR-HMM fitting...")

        model, model_name = kpms.fit_model(
            model,
            data,
            metadata,
            project_path,
            ar_only=True,
            num_iters=arhmm_iters,
            parallel_message_passing=False,
        )

        logger.info("AR-HMM fitting completed")
        logger.info("GPU usage after fitting AR-HMM:")
        print_gpu_usage()

        # Fit full model with reduced kappa
        reduced_kappa = 0.1 * kappa
        logger.info(
            f"Starting full model fitting with reduced kappa={reduced_kappa}")

        model = kpms.update_hypparams(model, kappa=reduced_kappa)

        model = kpms.fit_model(
            model,
            data,
            metadata,
            project_path,
            model_name,
            ar_only=False,
            start_iter=arhmm_iters,
            num_iters=arhmm_iters + full_model_iters,
            parallel_message_passing=False,
        )[0]

        logger.info("Full model fitting completed")
        logger.info("GPU usage after fitting full model:")
        print_gpu_usage()

        # Post-process and save results
        logger.info("Reindexing syllables and extracting results...")
        kpms.reindex_syllables_in_checkpoint(project_path, model_name)

        results = kpms.extract_results(
            model, metadata, project_path, model_name)
        kpms.save_results_as_csv(results, project_path, model_name)

        logger.info(
            f"Model training completed successfully. Model saved as: {model_name}")
        return model, model_name, results

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        raise RuntimeError(f"Model fitting failed: {e}") from e


def generate_plots_and_movies(
    model_name: str,
    results: Dict[str, Any],
    coordinates: Dict[str, np.ndarray],
    project_path: pathlib.Path,
    config_func: Callable[[], Dict[str, Any]] = None
) -> None:
    """Generate plots and movies from analysis results.

    Args:
        model_name: Name of the fitted model
        results: Results dictionary from fit_and_save_model
        coordinates: Raw coordinates dictionary
        project_path: Path to the project directory
        config_func: Optional function that returns configuration dictionary

    Raises:
        RuntimeError: If visualization generation fails
    """
    try:
        logger.info("Starting visualization generation...")

        if config_func is None:
            def config_func() -> Dict[str, Any]:
                return kpms.load_config(project_path)

        # Generate trajectory plots
        logger.info("Generating trajectory plots...")
        kpms.generate_trajectory_plots(
            coordinates, results, project_path, model_name, **config_func()
        )
        logger.info("Trajectory plots generated successfully")

        # Generate grid movies
        logger.info("Generating grid movies...")
        kpms.generate_grid_movies(
            results, project_path, model_name, coordinates=coordinates, **config_func()
        )
        logger.info("Grid movies generated successfully")

        # Generate similarity dendrogram
        logger.info("Generating similarity dendrogram...")
        kpms.plot_similarity_dendrogram(
            coordinates, results, project_path, model_name, **config_func()
        )
        logger.info("Similarity dendrogram generated successfully")

        logger.info("All visualizations generated successfully")

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise RuntimeError(f"Visualization generation failed: {e}") from e


def run_complete_pipeline(
    pose_dir: str,
    project_path: str,
    video_dir: str,
    bodyparts: list,
    skeleton: list,
    kappa: float = 1e7,
    arhmm_iters: int = 20,
    full_model_iters: int = 20,
    mixed_map_iters: int = 8
) -> Tuple[Any, str, Dict[str, Any]]:
    """Run the complete KPMS analysis pipeline.

    Args:
        pose_dir: Directory containing pose CSV files
        project_path: Path to the project directory
        video_dir: Directory containing video files
        bodyparts: List of bodypart names
        skeleton: List of skeleton connections
        kappa: Stickiness parameter (default: 1e7)
        arhmm_iters: Number of AR-HMM iterations (default: 20)
        full_model_iters: Number of full model iterations (default: 20)
        mixed_map_iters: Mixed map iterations for memory management (default: 8)

    Returns:
        Tuple containing (model, model_name, results)

    Raises:
        RuntimeError: If any pipeline step fails
    """
    try:
        from jax_moseq.utils import set_mixed_map_iters

        logger.info("Starting complete KPMS pipeline...")
        logger.info(f"Project path: {project_path}")
        logger.info(f"Pose directory: {pose_dir}")
        logger.info(f"Video directory: {video_dir}")

        # Set memory management
        set_mixed_map_iters(mixed_map_iters)

        # Setup project
        project_path_obj = pathlib.Path(project_path)
        kpms.setup_project(
            project_path_obj,
            video_dir=video_dir,
            bodyparts=bodyparts,
            skeleton=skeleton
        )

        # Update configuration
        kpms.update_config(
            project_path_obj,
            anterior_bodyparts=["BASE_NECK_INDEX"],
            posterior_bodyparts=["BASE_TAIL_INDEX"],
            use_bodyparts=bodyparts,
        )

        def config_func() -> Dict[str, Any]:
            return kpms.load_config(project_path_obj)

        # Execute pipeline steps
        data, metadata, coordinates = load_and_format_data(
            pose_dir, project_path_obj)
        pca, n_components_90 = perform_pca(data, config_func, project_path_obj)
        model, model_name, results = fit_and_save_model(
            data, metadata, pca, config_func, project_path_obj,
            kappa=kappa, arhmm_iters=arhmm_iters, full_model_iters=full_model_iters
        )
        generate_plots_and_movies(
            model_name, results, coordinates, project_path_obj, config_func)

        logger.info("Complete pipeline executed successfully")
        return model, model_name, results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise RuntimeError(f"Pipeline execution failed: {e}") from e
