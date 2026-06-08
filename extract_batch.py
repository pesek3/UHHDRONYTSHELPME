#!/usr/bin/env python3
"""
Batch extract frames from multiple video files.
Usage: python extract_batch.py --videos video1.mp4 video2.mp4 --rooms CORRIDOR P1 --interval 5
"""

import argparse
from pathlib import Path

from src.drone_rescue.extract_frames import extract_frames


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract frames from multiple video files"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video file paths",
    )
    parser.add_argument(
        "--rooms",
        nargs="+",
        required=True,
        help="Room names for each video (must match --videos count)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = every frame)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output base directory (default: data/training/rooms)",
    )
    
    args = parser.parse_args()
    
    if len(args.videos) != len(args.rooms):
        print(f"Error: {len(args.videos)} videos but {len(args.rooms)} rooms specified")
        return 1
    
    total_extracted = 0
    
    print(f"🎬 Batch processing {len(args.videos)} video(s)...\n")
    
    for video_path, room_name in zip(args.videos, args.rooms):
        print(f"\n{'='*60}")
        count = extract_frames(
            video_path=Path(video_path),
            room_name=room_name,
            output_dir=args.output_dir,
            interval=args.interval,
            verbose=True,
        )
        total_extracted += count
    
    print(f"\n{'='*60}")
    print(f"\n🎉 All done! Total frames extracted: {total_extracted}")
    
    return 0


if __name__ == "__main__":
    exit(main())
