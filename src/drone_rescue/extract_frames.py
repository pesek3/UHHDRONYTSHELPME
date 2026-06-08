#!/usr/bin/env python3
"""
Extract frames from video files to create training data images.
Usage: python -m drone_rescue.extract_frames <video_path> <room_name> [--interval N] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    room_name: str,
    output_dir: Path | None = None,
    interval: int = 1,
    verbose: bool = True,
) -> int:
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the video file
        room_name: Name of the room (used for folder organization)
        output_dir: Base output directory (defaults to data/training/rooms)
        interval: Extract every Nth frame (1 = every frame, 5 = every 5th frame)
        verbose: Print progress information
    
    Returns:
        Number of frames extracted
    """
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 0
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "training" / "rooms"
    
    # Create room directory
    room_dir = output_dir / room_name
    room_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"📹 Extracting frames from: {video_path}")
        print(f"📁 Saving to: {room_dir}")
        print(f"⏸️  Frame interval: every {interval} frame(s)")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return 0
    
    frame_count = 0
    extracted_count = 0
    
    # Get video properties for info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"📊 Video info: {total_frames} frames @ {fps:.1f} fps")
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % interval == 0:
            # Generate filename with zero-padded numbers
            filename = f"{room_name}_{extracted_count:06d}.jpg"
            filepath = room_dir / filename
            
            # Save frame
            cv2.imwrite(str(filepath), frame)
            extracted_count += 1
            
            if verbose and extracted_count % 50 == 0:
                print(f"  ✓ Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    if verbose:
        print(f"✅ Done! Extracted {extracted_count} frames to {room_dir}")
    
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files for training data"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("room", help="Room name (CORRIDOR, P1, P2, PRINTERS, RAI, etc.)")
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
    
    video_path = Path(args.video)
    count = extract_frames(
        video_path=video_path,
        room_name=args.room,
        output_dir=args.output_dir,
        interval=args.interval,
    )
    
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
