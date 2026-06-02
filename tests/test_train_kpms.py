"""Tests for the training CLI entrypoint."""

import importlib
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestTrainKpms(unittest.TestCase):
    """Test training script argument handling."""

    def import_train_module(self):
        """Import scripts.train_kpms with external dependencies stubbed."""
        fake_kpms = types.SimpleNamespace(
            setup_project=MagicMock(),
            update_config=MagicMock(),
        )
        fake_jax = types.SimpleNamespace(
            devices=MagicMock(return_value=[types.SimpleNamespace(platform="cpu")]),
        )
        fake_jax_utils = types.SimpleNamespace(
            set_mixed_map_iters=MagicMock(),
            set_mixed_map_gpus=MagicMock(),
        )
        fake_utils = types.SimpleNamespace(
            set_up_logging=MagicMock(),
            print_gpu_usage=MagicMock(),
            validate_data_quality=MagicMock(),
            load_config=MagicMock(return_value={}),
            save_config=MagicMock(),
            merge_config_with_args=MagicMock(side_effect=lambda config, args: config),
        )
        fake_methods = types.SimpleNamespace(
            load_and_format_data=MagicMock(),
            perform_pca=MagicMock(),
            fit_and_save_model=MagicMock(),
            generate_plots_and_movies=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "jax": fake_jax,
                "jax_moseq": types.SimpleNamespace(utils=fake_jax_utils),
                "jax_moseq.utils": fake_jax_utils,
                "keypoint_moseq": fake_kpms,
                "src.utils": fake_utils,
                "src.methods": fake_methods,
            },
        ):
            sys.modules.pop("scripts.train_kpms", None)
            module = importlib.import_module("scripts.train_kpms")

        return module, fake_kpms

    def test_resolve_video_dir_requires_videos_by_default(self):
        """Training should require videos unless --skip-videos is used."""
        module, _ = self.import_train_module()
        args = types.SimpleNamespace(video_dir=None, skip_videos=False)

        with self.assertRaisesRegex(ValueError, "Video directory is required"):
            module.resolve_video_dir(args, {})

    def test_resolve_video_dir_allows_skipping_videos(self):
        """Video directory should be optional (and ignored) when videos are skipped."""
        module, _ = self.import_train_module()
        args = types.SimpleNamespace(video_dir="/tmp/videos", skip_videos=True)

        self.assertIsNone(module.resolve_video_dir(args, {}))

    def test_resolve_video_dir_uses_config_value(self):
        """Configured video_dir should satisfy the video requirement."""
        module, _ = self.import_train_module()
        args = types.SimpleNamespace(video_dir=None, skip_videos=False)

        self.assertEqual(
            module.resolve_video_dir(args, {"video_dir": "/tmp/videos"}),
            "/tmp/videos",
        )

    def test_initialize_project_omits_video_dir_when_absent(self):
        """Project setup should not pass a video_dir when none was provided."""
        module, fake_kpms = self.import_train_module()
        logger = logging.getLogger("test")

        module.initialize_project(
            project_path=module.pathlib.Path("/tmp/project"),
            video_dir=None,
            bodyparts=["A", "B"],
            skeleton=[["A", "B"]],
            anterior_bodyparts=["A"],
            posterior_bodyparts=["B"],
            logger=logger,
        )

        fake_kpms.setup_project.assert_called_once_with(
            module.pathlib.Path("/tmp/project"),
            bodyparts=["A", "B"],
            skeleton=[["A", "B"]],
            overwrite=True,
        )


if __name__ == "__main__":
    unittest.main()
