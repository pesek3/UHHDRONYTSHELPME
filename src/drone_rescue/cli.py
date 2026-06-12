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


def cmd_auto_mission(args: argparse.Namespace) -> None:
    from .vision import analyze_image
    from .navigation import get_navigation_state, generate_movement_command, apply_movement_command
    from .planner import plan_exploration
    from .rooms import parse_room
    from djitellopy import Tello
    import time

    try:
        tello = Tello()
        tello.connect()
        tello.streamon()
        frame_reader = tello.get_frame_read()
        time.sleep(2.0)

        print("🚁 Starting autonomous search and rescue mission...")
        
        # Get initial room from first frame
        frame = frame_reader.frame
        result = analyze_image_from_frame(frame, ROOM_MODEL, PEOPLE_MODEL)
        print(f"📍 Starting in: {result.room or 'Unknown'}")
        print(f"👥 People detected: {result.people_count}")
        
        if result.room:
            start = parse_room(result.room)
            plan = plan_exploration(start)
            print(f"📋 Exploration plan: {' -> '.join(plan.path)}")
            
            prev_frame = None
            frames_in_state = 0
            visited_rooms = {start}
            
            # Autonomous navigation loop
            for target_room in plan.visit_order[1:]:  # Skip start room
                print(f"\n🎯 Moving to: {target_room}")
                frames_in_state = 0
                
                while len(visited_rooms) < len(plan.visit_order):
                    frame = frame_reader.frame
                    if frame is None:
                        continue
                    
                    # Analyze navigation state
                    nav_state = get_navigation_state(frame, prev_frame)
                    command = generate_movement_command(nav_state)
                    
                    print(f"  Action: {nav_state.recommended_action} | Doorway: {nav_state.doorway_detected} | Obstacle: {nav_state.obstacle_distance:.2f}")
                    
                    # Send command to drone
                    apply_movement_command(tello, command)
                    
                    # Check if we've reached a new room
                    frame = frame_reader.frame
                    if frame is not None:
                        room_result = analyze_image_from_frame(frame, ROOM_MODEL, PEOPLE_MODEL)
                        if room_result.room and parse_room(room_result.room) not in visited_rooms:
                            visited_rooms.add(parse_room(room_result.room))
                            print(f"✅ Reached {room_result.room}! People: {room_result.people_count}")
                            break
                    
                    frames_in_state += 1
                    prev_frame = frame
                    time.sleep(0.1)
        
        print("\n🎉 Mission complete!")
        
    except Exception as e:
        print(f"❌ Mission error: {e}")
    finally:
        try:
            tello.streamoff()
        finally:
            tello.end()


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

    analyze = subparsers.add_parser("analyze", help="Analyze one image.")
    analyze.add_argument("image")
    analyze.set_defaults(func=cmd_analyze)

    mission = subparsers.add_parser("mission", help="Recognize start room and plan the rescue mission.")
    mission.add_argument("start_image")
    mission.set_defaults(func=cmd_mission)

    gui = subparsers.add_parser("gui", help="Open the route planning GUI.")
    gui.set_defaults(func=cmd_gui)

    auto_mission = subparsers.add_parser("auto-mission", help="Run fully autonomous search and rescue mission.")
    auto_mission.set_defaults(func=cmd_auto_mission)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
