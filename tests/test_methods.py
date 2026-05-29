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

    def test_config_10k_kappa_loads_as_numeric(self):
        """The 10-keypoint config should deserialize kappa as a numeric value."""
        config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "config_10k.yml"

        with config_path.open() as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config["kappa"], float)
        self.assertEqual(config["kappa"], 1000000.0)


if __name__ == "__main__":
    unittest.main()
