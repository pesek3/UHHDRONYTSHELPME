from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CollectionSummary:
    room: str
    saved: int
    output_dir: Path


def collect_photos(room: str, output_root: Path, count: int, interval: float) -> CollectionSummary:
    try:
        from djitellopy import Tello
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    output_dir = output_root / room.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    tello = Tello()
    saved = 0

    try:
        tello.connect()
        tello.streamon()
        frame_reader = tello.get_frame_read()
        time.sleep(2.0)

        for index in range(count):
            frame = frame_reader.frame
            if frame is None:
                time.sleep(interval)
                continue

            filename = output_dir / f"{room.lower()}_{int(time.time())}_{index:03d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
            time.sleep(interval)
    finally:
        try:
            tello.streamoff()
        finally:
            tello.end()

    return CollectionSummary(room=room.upper(), saved=saved, output_dir=output_dir)
