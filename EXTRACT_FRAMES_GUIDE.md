# 🎬 Video Frame Extraction Guide

Extract frames from room videos to create training data for your drone rescue ML models.

## Setup

The script uses OpenCV which is already in your `requirements.txt`. Make sure your environment is activated:

```bash
source .venv/Scripts/activate  # or .venv\Scripts\activate on Windows
```

## Usage

### Single Video

Extract all frames from a single video to a room directory:

```bash
python -m drone_rescue.extract_frames video.mp4 CORRIDOR
```

Options:
- `--interval N`: Extract every Nth frame (default: 1 = every frame)
  - Use `--interval 5` to extract every 5th frame (faster processing, fewer images)
- `--output-dir PATH`: Custom output directory (default: `data/training/rooms/`)

Example:
```bash
python -m drone_rescue.extract_frames recording_corridor.mp4 CORRIDOR --interval 3
```

This will save frames to `data/training/rooms/CORRIDOR/` with names like:
- `CORRIDOR_000000.jpg`
- `CORRIDOR_000001.jpg`
- etc.

### Batch Processing Multiple Videos

Process multiple videos at once:

```bash
python extract_batch.py \
  --videos video_corridor.mp4 video_p1.mp4 video_p2.mp4 \
  --rooms CORRIDOR P1 P2 \
  --interval 2
```

This will:
- Extract every 2nd frame from each video
- Save to the respective room directory
- Show progress for each video

## Tips

- **Every Frame**: Use `--interval 1` (default) to get maximum training data
- **Reduce Size**: Use `--interval 5` or higher to reduce the number of images and processing time
- **Sample Quality**: Videos at higher resolution and frame rate will give better training data
- **Organized Names**: Frames are named with the room and a zero-padded counter for easy sorting

## Example Workflow

```bash
# Extract lots of data: every frame
python -m drone_rescue.extract_frames corridor_full_scan.mp4 CORRIDOR

# Extract less data: every 3rd frame (faster, but still good coverage)
python -m drone_rescue.extract_frames p1_walk.mp4 P1 --interval 3

# Batch process everything
python extract_batch.py \
  --videos recordings/corridor.mp4 recordings/p1.mp4 recordings/p2.mp4 recordings/printers.mp4 recordings/rai.mp4 \
  --rooms CORRIDOR P1 P2 PRINTERS RAI \
  --interval 2
```

The extracted frames will be saved to your `data/training/rooms/[ROOM]/` directories and are ready for use with your vision model training!
