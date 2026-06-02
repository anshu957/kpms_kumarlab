"""Tests for pipeline orchestration helpers."""

import importlib
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import yaml


class TestMethods(unittest.TestCase):
    """Test method-level regressions."""

    def test_run_complete_pipeline_accepts_all_load_outputs(self):
        """run_complete_pipeline should accept the full four-value loader tuple."""
        fake_kpms = types.SimpleNamespace(
            setup_project=MagicMock(),
            update_config=MagicMock(),
            load_config=MagicMock(return_value={}),
        )
        fake_jax_utils = types.SimpleNamespace(
            set_mixed_map_iters=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "keypoint_moseq": fake_kpms,
                "jax_moseq": types.SimpleNamespace(utils=fake_jax_utils),
                "jax_moseq.utils": fake_jax_utils,
            },
        ):
            sys.modules.pop("src.methods", None)
            methods = importlib.import_module("src.methods")

            with (
                patch.object(
                    methods,
                    "load_and_format_data",
                    return_value=("data", "metadata", "coordinates", "confidences"),
                ) as load_and_format_data,
                patch.object(
                    methods,
                    "perform_pca",
                    return_value=("pca", 3),
                ),
                patch.object(
                    methods,
                    "fit_and_save_model",
                    return_value=("model", "model_name", {"syllables": []}),
                ),
                patch.object(methods, "generate_plots_and_movies") as generate_plots_and_movies,
            ):
                model, model_name, results = methods.run_complete_pipeline(
                    pose_dir="/tmp/poses",
                    project_path="/tmp/project",
                    video_dir="/tmp/videos",
                    bodyparts=["BASE_NECK_INDEX", "BASE_TAIL_INDEX"],
                    skeleton=[["BASE_NECK_INDEX", "BASE_TAIL_INDEX"]],
                )

        self.assertEqual((model, model_name, results), ("model", "model_name", {"syllables": []}))
        load_and_format_data.assert_called_once()
        generate_plots_and_movies.assert_called_once()
        self.assertEqual(generate_plots_and_movies.call_args.args[2], "coordinates")

    def _import_methods(self, config):
        """Import src.methods with keypoint_moseq stubbed to return ``config``."""
        fake_kpms = types.SimpleNamespace(
            load_config=MagicMock(return_value=config),
            generate_trajectory_plots=MagicMock(),
            generate_grid_movies=MagicMock(),
            plot_similarity_dendrogram=MagicMock(),
        )
        fake_jax_utils = types.SimpleNamespace(set_mixed_map_iters=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "keypoint_moseq": fake_kpms,
                "jax_moseq": types.SimpleNamespace(utils=fake_jax_utils),
                "jax_moseq.utils": fake_jax_utils,
            },
        ):
            sys.modules.pop("src.methods", None)
            methods = importlib.import_module("src.methods")
        return methods, fake_kpms

    def test_generate_plots_skips_grid_movies_when_skip_videos(self):
        """skip_videos should bypass grid movies but keep pose-based plots."""
        methods, fake_kpms = self._import_methods({"video_dir": "/tmp/videos"})

        status = methods.generate_plots_and_movies(
            "model_name", {"syllables": []}, {"vid": "coords"},
            pathlib.Path("/tmp/project"), skip_videos=True,
        )

        fake_kpms.generate_grid_movies.assert_not_called()
        fake_kpms.generate_trajectory_plots.assert_called_once()
        fake_kpms.plot_similarity_dendrogram.assert_called_once()
        self.assertNotIn("grid_movies", status)

    def test_generate_plots_skips_grid_movies_without_video_dir(self):
        """Grid movies should be skipped automatically when no video_dir is configured."""
        methods, fake_kpms = self._import_methods({})

        status = methods.generate_plots_and_movies(
            "model_name", {"syllables": []}, {"vid": "coords"},
            pathlib.Path("/tmp/project"),
        )

        fake_kpms.generate_grid_movies.assert_not_called()
        self.assertNotIn("grid_movies", status)

    def test_generate_plots_runs_grid_movies_with_video_dir(self):
        """Grid movies should run when a video_dir is configured and not skipped."""
        methods, fake_kpms = self._import_methods({"video_dir": "/tmp/videos"})

        status = methods.generate_plots_and_movies(
            "model_name", {"syllables": []}, {"vid": "coords"},
            pathlib.Path("/tmp/project"),
        )

        fake_kpms.generate_grid_movies.assert_called_once()
        self.assertIn("grid_movies", status)

    def test_config_10k_kappa_loads_as_numeric(self):
        """The 10-keypoint config should deserialize kappa as a numeric value."""
        config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "config_10k.yml"

        with config_path.open() as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config["kappa"], float)
        self.assertEqual(config["kappa"], 1000000.0)


if __name__ == "__main__":
    unittest.main()
