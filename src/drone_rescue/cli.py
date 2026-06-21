from __future__ import annotations

import argparse
from pathlib import Path

from .planner import plan_exploration
from .rooms import Room, parse_room


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
ROOM_MODEL = MODELS_ROOT / "room_classifier.joblib"
PEOPLE_MODEL = MODELS_ROOT / "people_regressor.joblib"


def cmd_simulate(args: argparse.Namespace) -> None:
    start = parse_room(args.start)
    plan = plan_exploration(start)
    print(f"Start room: {plan.start}")
    print("Visit order:")
    for index, room in enumerate(plan.visit_order, start=1):
        print(f"  {index}. {room}")
    print("Flight path:")
    print("  " + " -> ".join(plan.path))


def cmd_collect(args: argparse.Namespace) -> None:
    from .tello import collect_photos

    summary = collect_photos(
        room=parse_room(args.room).value,
        output_root=DATA_ROOT / "raw",
        count=args.count,
        interval=args.interval,
    )
    print(f"Saved {summary.saved} photos to {summary.output_dir}")


def cmd_train_room(_: argparse.Namespace) -> None:
    from .vision import train_room_classifier

    stats = train_room_classifier(DATA_ROOT / "training" / "rooms", ROOM_MODEL)
    print(f"Room classifier saved to {ROOM_MODEL}")
    print(f"Samples: {stats['samples']}")
    print(f"Labels: {', '.join(stats['labels'])}")
    print(f"Validation accuracy: {stats['accuracy']:.2f}")


def cmd_train_people(_: argparse.Namespace) -> None:
    from .vision import train_people_regressor

    data_dir = DATA_ROOT / "training" / "people_counts"
    stats = train_people_regressor(data_dir, data_dir / "labels.csv", PEOPLE_MODEL)
    print(f"People regressor saved to {PEOPLE_MODEL}")
    print(f"Samples: {stats['samples']}")
    print(f"Validation MAE: {stats['mae']:.2f} people")


def cmd_train_door(args: argparse.Namespace) -> None:
    """Extract frames from a door-transition video and retrain room classifier.

    The function splits the video frames in half: first half labelled as `from_room`,
    second half as `to_room`. Frames are saved under `data/training/rooms/<ROOM>/`.
    """
    import cv2
    from .vision import train_room_classifier

    video_path = Path(args.video)
    from_room = args.from_room
    to_room = args.to_room
    interval = max(1, int(args.interval))

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise SystemExit("Unable to read video or video has no frames")

    split_idx = total_frames // 2

    out_root = DATA_ROOT / "training" / "rooms"
    out_from = out_root / from_room
    out_to = out_root / to_room
    out_from.mkdir(parents=True, exist_ok=True)
    out_to.mkdir(parents=True, exist_ok=True)

    saved_from = 0
    saved_to = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval != 0:
            idx += 1
            continue

        if idx <= split_idx:
            out_path = out_from / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_from += 1
        else:
            out_path = out_to / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_to += 1

        idx += 1

    cap.release()

    print(f"Saved {saved_from} frames to {out_from}")
    print(f"Saved {saved_to} frames to {out_to}")

    print("Retraining room classifier with new frames...")
    stats = train_room_classifier(out_root, ROOM_MODEL)
    print(f"Room classifier saved to {ROOM_MODEL}")
    print(f"Samples: {stats['samples']}")
    print(f"Labels: {', '.join(stats['labels'])}")
    print(f"Validation accuracy: {stats['accuracy']:.2f}")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .vision import analyze_image

    result = analyze_image(Path(args.image), ROOM_MODEL, PEOPLE_MODEL)
    print(f"Image: {args.image}")
    print(f"Room: {result.room or 'unknown'}")
    if result.confidence is not None:
        print(f"Room confidence: {result.confidence:.2f}")
    print(f"People count: {result.people_count}")
    print(f"People source: {result.source}")


def cmd_mission(args: argparse.Namespace) -> None:
    from .vision import analyze_image

    result = analyze_image(Path(args.start_image), ROOM_MODEL, PEOPLE_MODEL)
    if result.room is None:
        raise SystemExit("No room model found. Train it first with: python -m drone_rescue train-room")

    start = parse_room(result.room)
    plan = plan_exploration(start)

    print(f"Recognized start room: {start}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence:.2f}")
    print(f"People in start image: {result.people_count}")
    print("Planned exploration path:")
    print("  " + " -> ".join(plan.path))


def cmd_gui(_: argparse.Namespace) -> None:
    from .gui import main as gui_main

    gui_main()


# Global abort flag
_abort_mission = False


def _setup_abort_listener():
    """Setup keyboard listener for 'X' key to abort mission."""
    global _abort_mission
    try:
        from pynput import keyboard
        
        def on_press(key):
            global _abort_mission
            try:
                if key.char and key.char.lower() == 'x':
                    _abort_mission = True
                    print("\n🛑 ABORT SIGNAL RECEIVED! Landing drone...")
            except AttributeError:
                pass
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    except ImportError:
        print("⚠️  pynput not installed. Using Ctrl+C for emergency stop instead.")
        print("   Install with: pip install pynput")
        return None


def cmd_auto_mission(args: argparse.Namespace) -> None:
    global _abort_mission
    raise SystemExit(
        "auto-mission is disabled for manual-room mode. "
        "Run: python -m drone_rescue gui, then use Scan People 360 and WASD controls."
    )
    from .vision import analyze_frames_360
    from .navigation import (
        align_to_doorway_and_enter,
        capture_360_scan,
        capture_360_scan_until_doorway,
    )
    from .planner import plan_exploration
    from .rooms import parse_room
    from djitellopy import Tello
    import time

    tello = None
    abort_listener = None
    
    try:
        _abort_mission = False
        
        # Setup abort listener
        print("⌨️  Press 'X' anytime to abort mission and land drone safely")
        abort_listener = _setup_abort_listener()
        
        tello = Tello()
        tello.connect()
        tello.streamon()
        frame_reader = tello.get_frame_read()
        time.sleep(2.0)

        print("🚁 Starting autonomous search and rescue mission...")
        tello.takeoff()
        print("🛫 Drone has taken off")
        
        # Get initial room from 360-degree scan (or use provided start room)
        if _abort_mission:
            print("❌ Abort triggered before mission start")
            return
        
        # Determine starting room
        if hasattr(args, 'start') and args.start:
            # Use provided start room, skip scan
            start = parse_room(args.start)
            result = type('obj', (object,), {'room': start.value, 'people_count': 0, 'confidence': 1.0})()
            print(f"📍 Starting in: {result.room} (provided, skipping scan)")
        else:
            # Scan to detect starting room
            print("\n📍 Analyzing starting room with 360° scan...")
            frames = capture_360_scan(tello, frame_reader)
            result = analyze_frames_360(frames, ROOM_MODEL, PEOPLE_MODEL)
            print(f"📍 Starting in: {result.room or 'Unknown'}")
            start = parse_room(result.room) if result.room else Room.RAI
        
        print(f"👥 People detected: {result.people_count}")
        if hasattr(result, 'confidence') and result.confidence:
            print(f"Room confidence: {result.confidence:.2f}")
        
        if result.room:
            start = parse_room(result.room) if isinstance(result.room, str) else result.room
            plan = plan_exploration(start)
            print(f"📋 Exploration plan: {' -> '.join(room.value for room in plan.path)}")
            
            visited_rooms = {start}
            
            # Autonomous navigation loop: Two-scan approach
            # Scan 1: identify room + people
            # Scan 2: find doorway
            # Move through doorway
            for target_room in plan.visit_order[1:]:  # Skip start room
                if _abort_mission:
                    print("❌ Abort triggered during navigation")
                    break
                    
                print(f"\n🎯 Moving towards: {target_room}")
                
                # Scan 1: Identify current room and people
                print("📸 Scan 1: Identifying room and people count...")
                frames = capture_360_scan(tello, frame_reader)
                room_result = analyze_frames_360(frames, ROOM_MODEL, PEOPLE_MODEL)
                print(f"  Room: {room_result.room} | People: {room_result.people_count} | Conf: {room_result.confidence:.2f}")
                
                if room_result.room:
                    curr_room = parse_room(room_result.room)
                    print(f"  ✓ Current room confirmed: {curr_room.value}")
                else:
                    print(f"  ⚠️  Could not identify room!")
                    continue

                # Scan 2: find doorway and stop rotating as soon as it is visible
                print("📸 Scan 2: Finding doorway...")
                doorway_scan = capture_360_scan_until_doorway(tello, frame_reader)
                print(
                    "  Doorway scan: "
                    f"{len(doorway_scan.frames)} frames, {doorway_scan.rotation_degrees} degrees rotated"
                )

                if doorway_scan.doorway_found and doorway_scan.doorway_state is not None:
                    doorway_state = doorway_scan.doorway_state
                    print(
                        "  ✓ Doorway detected; stopping rotation "
                        f"at center_x={doorway_state.doorway_center_x:.2f}"
                    )

                    align_to_doorway_and_enter(tello, doorway_state)
                    time.sleep(1.0)
                    
                    # Wait and scan to confirm new room
                    time.sleep(2.0)
                    print("📸 Confirming new room...")
                    frames = capture_360_scan(tello, frame_reader)
                    new_room_result = analyze_frames_360(frames, ROOM_MODEL, PEOPLE_MODEL)
                    
                    if new_room_result.room and parse_room(new_room_result.room) not in visited_rooms:
                        new_room = parse_room(new_room_result.room)
                        visited_rooms.add(new_room)
                        print(f"✅ ENTERED {new_room_result.room}! People: {new_room_result.people_count}")
                    else:
                        print(f"⚠️  Still in {new_room_result.room if new_room_result.room else 'unknown'} (may need retry)")
                else:
                    print(f"  ✗ NO DOORWAY DETECTED in this scan!")
                    print(f"    (Try reducing obstacle/edge thresholds or checking camera angle)")

                if _abort_mission:
                    break
                
                if _abort_mission:
                    break
        
        if not _abort_mission:
            print("\n🎉 Mission complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected - landing drone...")
        _abort_mission = True
    except Exception as e:
        print(f"❌ Mission error: {e}")
        _abort_mission = True
    finally:
        # Graceful shutdown
        if tello is not None:
            try:
                if _abort_mission:
                    print("\n🛬 Landing drone...")
                    tello.land()
                    time.sleep(2.0)
                print("📹 Stopping video stream...")
                tello.streamoff()
                print("🔌 Disconnecting from drone...")
                tello.end()
                print("✅ Drone safely disconnected")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
                try:
                    tello.emergency()
                    print("🆘 Emergency stop sent")
                except:
                    pass


def analyze_image_from_frame(frame, room_model_path, people_model_path):
    """Analyze a frame directly instead of loading from file."""
    from .vision import extract_features, AnalysisResult
    from pathlib import Path
    import joblib
    import cv2
    import numpy as np
    
    try:
        model = joblib.load(str(room_model_path))
        features = extract_features(frame)
        features_2d = features.reshape(1, -1)
        room_pred = model.predict(features_2d)[0]
        confidence = float(model.predict_proba(features_2d).max())
        
        # For now, estimate people count from brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people_count = max(0, int(gray.mean() / 40) - 2)  # Simple heuristic
        
        return AnalysisResult(
            room=room_pred,
            people_count=people_count,
            confidence=confidence,
            source="frame_analysis"
        )
    except Exception as e:
        print(f"Frame analysis error: {e}")
        return AnalysisResult(room=None, people_count=0, confidence=None, source="error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drone-rescue")
    subparsers = parser.add_subparsers(required=True)

    simulate = subparsers.add_parser("simulate", help="Plan exploration from a start room.")
    simulate.add_argument("--start", choices=[room.value for room in Room], required=True)
    simulate.set_defaults(func=cmd_simulate)

    collect = subparsers.add_parser("collect", help="Collect labelled room photos from DJI Tello.")
    collect.add_argument("--room", choices=[room.value for room in Room], required=True)
    collect.add_argument("--count", type=int, default=20)
    collect.add_argument("--interval", type=float, default=0.5)
    collect.set_defaults(func=cmd_collect)

    train_room = subparsers.add_parser("train-room", help="Train the room classifier.")
    train_room.set_defaults(func=cmd_train_room)

    train_people = subparsers.add_parser("train-people", help="Train the people-count regressor.")
    train_people.set_defaults(func=cmd_train_people)

    train_door = subparsers.add_parser("train-door", help="Extract frames from a door video and retrain room classifier.")
    train_door.add_argument("--video", required=True, help="Path to the door-transition video file")
    train_door.add_argument("--from-room", required=True, choices=[room.value for room in Room], help="Label for the starting room")
    train_door.add_argument("--to-room", required=True, choices=[room.value for room in Room], help="Label for the target room")
    train_door.add_argument("--interval", type=int, default=5, help="Save one frame every N frames")
    train_door.set_defaults(func=cmd_train_door)

    analyze = subparsers.add_parser("analyze", help="Analyze one image.")
    analyze.add_argument("image")
    analyze.set_defaults(func=cmd_analyze)

    mission = subparsers.add_parser("mission", help="Recognize start room and plan the rescue mission.")
    mission.add_argument("start_image")
    mission.set_defaults(func=cmd_mission)

    gui = subparsers.add_parser("gui", help="Open the route planning GUI.")
    gui.set_defaults(func=cmd_gui)

    auto_mission = subparsers.add_parser("auto-mission", help="Disabled; use gui for manual-room mode.")
    auto_mission.add_argument("--start", choices=[room.value for room in Room], default="RAI", help="Starting room (default: RAI)")
    auto_mission.set_defaults(func=cmd_auto_mission)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
