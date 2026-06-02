#!/usr/bin/env python
"""
Video Diagnostic Tool for KeyPoint-MoSeq Pipeline

Tests all videos in a directory for decoding issues that might cause
problems during visualization generation.

Usage:
    python scripts/test_videos.py --video-dir data/exp1/videos/
    python scripts/test_videos.py --video-dir data/exp1/videos/ --sample-rate 100
"""

import argparse
import pathlib
import sys
from typing import Dict, List, Tuple
import logging

try:
    import vidio
except ImportError:
    print("ERROR: vidio library not found. Install with: pip install vidio")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_video_file(
    video_path: pathlib.Path,
    sample_rate: int = 50,
    max_frames: int = None
) -> Tuple[bool, Dict]:
    """Test a single video file for decoding issues.

    Args:
        video_path: Path to video file
        sample_rate: Test every Nth frame (default: 50)
        max_frames: Maximum number of frames to test (default: all)

    Returns:
        Tuple of (success, info_dict) where info_dict contains:
        - total_frames: Total number of frames in video
        - tested_frames: Number of frames tested
        - failed_frames: List of frame indices that failed
        - error_messages: List of error messages
    """
    info = {
        'total_frames': 0,
        'tested_frames': 0,
        'failed_frames': [],
        'error_messages': []
    }

    try:
        # Open video
        video = vidio.VideoReader(str(video_path))
        info['total_frames'] = len(video)

        # Determine frames to test
        if max_frames:
            frames_to_test = min(info['total_frames'], max_frames)
        else:
            frames_to_test = info['total_frames']

        # Test frames at intervals
        tested = 0
        for frame_idx in range(0, frames_to_test, sample_rate):
            try:
                _ = video[frame_idx]
                tested += 1
            except Exception as e:
                info['failed_frames'].append(frame_idx)
                error_msg = f"Frame {frame_idx}: {str(e)}"
                info['error_messages'].append(error_msg)

        info['tested_frames'] = tested

        # Also test the last frame specifically
        if frames_to_test > 0 and (frames_to_test - 1) % sample_rate != 0:
            try:
                _ = video[frames_to_test - 1]
                info['tested_frames'] += 1
            except Exception as e:
                last_frame = frames_to_test - 1
                info['failed_frames'].append(last_frame)
                error_msg = f"Frame {last_frame}: {str(e)}"
                info['error_messages'].append(error_msg)

        video.close()

        success = len(info['failed_frames']) == 0
        return success, info

    except Exception as e:
        info['error_messages'].append(f"Failed to open video: {str(e)}")
        return False, info


def find_video_files(video_dir: pathlib.Path) -> List[pathlib.Path]:
    """Find all video files in a directory.

    Args:
        video_dir: Directory to search

    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg']
    video_files = []

    for ext in video_extensions:
        video_files.extend(video_dir.rglob(f'*{ext}'))

    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description='Test videos for decoding issues',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing videos to test')
    parser.add_argument('--sample-rate', type=int, default=50,
                       help='Test every Nth frame (lower = more thorough but slower)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to test per video')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output for each video')

    args = parser.parse_args()

    video_dir = pathlib.Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory does not exist: {video_dir}")
        sys.exit(1)

    logger.info("="*80)
    logger.info("Video Diagnostic Tool")
    logger.info("="*80)
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Sample rate: every {args.sample_rate} frames")
    if args.max_frames:
        logger.info(f"Max frames per video: {args.max_frames}")
    logger.info("")

    # Find all videos
    logger.info("Searching for video files...")
    video_files = find_video_files(video_dir)

    if not video_files:
        logger.warning("No video files found!")
        sys.exit(0)

    logger.info(f"Found {len(video_files)} video file(s)")
    logger.info("")

    # Test each video
    results = {}
    failed_videos = []

    for i, video_path in enumerate(video_files, 1):
        rel_path = video_path.relative_to(video_dir)
        logger.info(f"[{i}/{len(video_files)}] Testing: {rel_path}")

        success, info = test_video_file(video_path, args.sample_rate, args.max_frames)
        results[str(rel_path)] = (success, info)

        if not success:
            failed_videos.append(str(rel_path))

        if args.verbose or not success:
            logger.info(f"  Total frames: {info['total_frames']}")
            logger.info(f"  Tested frames: {info['tested_frames']}")
            if info['failed_frames']:
                logger.warning(f"  Failed frames: {info['failed_frames'][:10]}")
                if len(info['failed_frames']) > 10:
                    logger.warning(f"  ... and {len(info['failed_frames']) - 10} more")
                for err_msg in info['error_messages'][:3]:
                    logger.warning(f"    {err_msg}")
                if len(info['error_messages']) > 3:
                    logger.warning(f"    ... and {len(info['error_messages']) - 3} more errors")
            else:
                logger.info("  Status: ✓ All tested frames decoded successfully")
        else:
            logger.info("  Status: ✓ OK")

        logger.info("")

    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total videos tested: {len(video_files)}")
    logger.info(f"Successful: {len(video_files) - len(failed_videos)}")
    logger.info(f"Failed: {len(failed_videos)}")

    if failed_videos:
        logger.warning("\nVideos with decoding issues:")
        for video in failed_videos:
            logger.warning(f"  - {video}")

        logger.warning("\nThese videos may cause issues during visualization generation.")
        logger.warning("Possible solutions:")
        logger.warning("  1. Re-encode videos using ffmpeg with standard codecs")
        logger.warning("  2. Use --skip-videos flag during training")
        logger.warning("  3. Remove problematic videos from the dataset")
        logger.warning("\nExample re-encoding command:")
        logger.warning("  ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 18 output.mp4")
        sys.exit(1)
    else:
        logger.info("\n✓ All videos passed decoding tests!")
        logger.info("Videos should work correctly with grid movie generation.")
        sys.exit(0)


if __name__ == '__main__':
    main()
