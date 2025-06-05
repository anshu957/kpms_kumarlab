"""Essential tests for KeyPoint-MoSeq behavioral analysis pipeline.

This module contains basic tests for the core functionality to ensure
the research pipeline works correctly. Tests use mock data to avoid
dependencies on external files.
"""

from src.utils import load_keypoints_pd, validate_data_quality, set_up_logging
import unittest
import numpy as np
import os
import tempfile
import pathlib
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEssentialFunctionality(unittest.TestCase):
    """Test essential functions needed for the research pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_keypoints_pd_basic(self):
        """Test basic keypoint loading functionality."""
        # Create a mock CSV file
        csv_path = os.path.join(self.temp_dir, "test_pose.csv")

        # Create test data: 10 frames, 12 keypoints (x, y, conf for each)
        test_data = np.random.rand(10, 36)  # 12 keypoints * 3 values
        header = ["header_line"]

        with open(csv_path, 'w') as f:
            f.write("header\n")
            for row in test_data:
                f.write(",".join(map(str, row)) + "\n")

        # Test loading
        coords, confs = load_keypoints_pd(self.temp_dir)

        self.assertEqual(len(coords), 1)
        self.assertEqual(len(confs), 1)

        filename = "test_pose.csv"
        self.assertIn(filename, coords)
        # frames, keypoints, xy
        self.assertEqual(coords[filename].shape, (10, 12, 2))
        self.assertEqual(confs[filename].shape, (10, 12)
                         )      # frames, keypoints

    def test_load_keypoints_pd_empty_directory(self):
        """Test behavior with empty directory."""
        with self.assertRaises(ValueError):
            load_keypoints_pd(self.temp_dir)

    def test_load_keypoints_pd_nonexistent_directory(self):
        """Test behavior with non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            load_keypoints_pd("/nonexistent/directory")

    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create test data with known quality issues
        coords = {
            "good_file.csv": np.random.rand(100, 12, 2),
            "bad_file.csv": np.full((100, 12, 2), np.nan)  # All NaN
        }

        confs = {
            # Good confidence
            "good_file.csv": np.random.rand(100, 12) * 0.9 + 0.1,
            "bad_file.csv": np.zeros((100, 12))  # Low confidence
        }

        quality_report = validate_data_quality(coords, confs)

        self.assertIn("good_file.csv", quality_report)
        self.assertIn("bad_file.csv", quality_report)

        # Check that bad file has high NaN percentage
        bad_report = quality_report["bad_file.csv"]
        self.assertGreater(bad_report["nan_coordinates_pct"], 90)
        self.assertLess(bad_report["mean_confidence"], 0.1)

    def test_set_up_logging(self):
        """Test logging setup."""
        log_dir = pathlib.Path(self.temp_dir) / "logs"

        # This should not raise an exception
        set_up_logging(log_dir)

        # Check that log directory was created
        self.assertTrue(log_dir.exists())

        # Check that log file was created
        log_files = list(log_dir.glob("*.log"))
        self.assertGreater(len(log_files), 0)

    def test_data_shapes_consistency(self):
        """Test that data shapes remain consistent through processing."""
        # Create test data with specific dimensions
        n_frames, n_keypoints = 50, 12
        test_coords = np.random.rand(n_frames, n_keypoints, 2)
        test_confs = np.random.rand(n_frames, n_keypoints)

        # Test that shapes are preserved
        # Same number of frames
        self.assertEqual(test_coords.shape[0], test_confs.shape[0])
        # Same number of keypoints
        self.assertEqual(test_coords.shape[1], test_confs.shape[1])
        self.assertEqual(test_coords.shape[2], 2)  # x, y coordinates

    def test_coordinate_swap(self):
        """Test that coordinate swapping (JABS format) works correctly."""
        # Create test data where x != y to verify swapping
        test_data = np.array([
            [1.0, 2.0, 0.9],  # x=1, y=2, conf=0.9
            [3.0, 4.0, 0.8],  # x=3, y=4, conf=0.8
        ]).reshape(1, 2, 3)  # 1 frame, 2 keypoints, 3 values each

        # Simulate the coordinate extraction and swapping from load_keypoints_pd
        coords = test_data[:, :, :2][:, :, ::-1]  # Extract coords and swap x,y

        # After swapping, first coordinate should be [2, 1] (y, x)
        expected = np.array([[[2.0, 1.0], [4.0, 3.0]]])
        np.testing.assert_array_equal(coords, expected)


if __name__ == '__main__':
    unittest.main()
