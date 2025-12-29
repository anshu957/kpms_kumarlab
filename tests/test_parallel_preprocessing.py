"""Tests for parallel preprocessing functionality.

This module contains tests for the parallel H5 to CSV conversion functions.
"""

import unittest
import numpy as np
import os
import tempfile
import pathlib
import h5py
import sys
import shutil
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import h5_to_csv_poses_parallel, h5_to_csv_poses


class TestParallelPreprocessing(unittest.TestCase):
    """Test parallel preprocessing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.h5_dir = os.path.join(self.temp_dir, "h5")
        self.csv_dir = os.path.join(self.temp_dir, "csv")
        self.csv_parallel_dir = os.path.join(self.temp_dir, "csv_parallel")
        
        os.makedirs(self.h5_dir)
        os.makedirs(self.csv_dir)
        os.makedirs(self.csv_parallel_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_h5_file(self, filename, n_frames=10, n_keypoints=12, pose_version="v6"):
        """Create a mock H5 file for testing."""
        filepath = os.path.join(self.h5_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            poseest = f.create_group('poseest')
            
            if pose_version == "v6":
                # Shape: (n_frames, 1, n_keypoints, 2)
                points = np.random.rand(n_frames, 1, n_keypoints, 2) * 100
                confidence = np.random.rand(n_frames, 1, n_keypoints)
            else:  # v2
                # Shape: (n_frames, n_keypoints, 2)
                points = np.random.rand(n_frames, n_keypoints, 2) * 100
                confidence = np.random.rand(n_frames, n_keypoints)
            
            poseest.create_dataset('points', data=points)
            poseest.create_dataset('confidence', data=confidence)
        
        return filepath

    def test_parallel_conversion_basic(self):
        """Test basic parallel conversion functionality."""
        # Create mock H5 files
        for i in range(3):
            self._create_mock_h5_file(f"test_{i}.h5")
        
        # Convert using parallel processing
        converted_files = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            validate_output=True,
            n_jobs=2
        )
        
        # Check that all files were converted
        self.assertEqual(len(converted_files), 3)
        
        # Check that CSV files exist
        for i in range(3):
            csv_path = os.path.join(self.csv_parallel_dir, f"test_{i}.csv")
            self.assertTrue(os.path.exists(csv_path))

    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel conversion produces same results as sequential."""
        # Create mock H5 files
        for i in range(3):
            self._create_mock_h5_file(f"test_{i}.h5", n_frames=5, n_keypoints=12)
        
        # Convert using sequential processing
        sequential_files = h5_to_csv_poses(
            folder_path=self.h5_dir,
            dest_path=self.csv_dir,
            pose_version="v6",
            validate_output=True
        )
        
        # Convert using parallel processing
        parallel_files = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            validate_output=True,
            n_jobs=2
        )
        
        # Check that same number of files were converted
        self.assertEqual(len(sequential_files), len(parallel_files))
        
        # Compare file contents
        for i in range(3):
            csv_seq = os.path.join(self.csv_dir, f"test_{i}.csv")
            csv_par = os.path.join(self.csv_parallel_dir, f"test_{i}.csv")
            
            df_seq = pd.read_csv(csv_seq, header=None)
            df_par = pd.read_csv(csv_par, header=None)
            
            # Check that dataframes are identical
            self.assertEqual(df_seq.shape, df_par.shape)
            np.testing.assert_allclose(df_seq.values, df_par.values, rtol=1e-8)

    def test_parallel_with_existing_files(self):
        """Test parallel conversion with existing files (overwrite=False)."""
        # Create mock H5 files
        self._create_mock_h5_file("test_1.h5")
        self._create_mock_h5_file("test_2.h5")
        
        # First conversion
        converted_files_1 = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            overwrite=False,
            n_jobs=2
        )
        
        self.assertEqual(len(converted_files_1), 2)
        
        # Second conversion (should skip existing files)
        converted_files_2 = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            overwrite=False,
            n_jobs=2
        )
        
        # Should still report 2 files (skipped existing ones)
        self.assertEqual(len(converted_files_2), 2)

    def test_parallel_with_n_jobs_none(self):
        """Test parallel conversion with n_jobs=None (auto-detect CPU count)."""
        # Create mock H5 file
        self._create_mock_h5_file("test.h5")
        
        # Convert with n_jobs=None
        converted_files = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            n_jobs=None
        )
        
        self.assertEqual(len(converted_files), 1)
        self.assertTrue(os.path.exists(os.path.join(self.csv_parallel_dir, "test.csv")))

    def test_parallel_error_handling(self):
        """Test error handling in parallel conversion."""
        # Create one valid H5 file
        self._create_mock_h5_file("valid.h5")
        
        # Create an invalid H5 file
        invalid_path = os.path.join(self.h5_dir, "invalid.h5")
        with open(invalid_path, 'w') as f:
            f.write("This is not a valid H5 file")
        
        # Should not raise an exception, but should report failures
        converted_files = h5_to_csv_poses_parallel(
            folder_path=self.h5_dir,
            dest_path=self.csv_parallel_dir,
            pose_version="v6",
            n_jobs=2
        )
        
        # At least one file should be converted successfully
        self.assertGreaterEqual(len(converted_files), 1)


if __name__ == '__main__':
    unittest.main()
