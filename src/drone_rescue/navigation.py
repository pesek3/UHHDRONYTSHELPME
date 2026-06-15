"""
Autonomous navigation module for drone movement and obstacle avoidance.
Uses vision-based navigation to detect doorways and navigate between rooms.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationState:
    can_move_forward: bool
    doorway_detected: bool
    doorway_center_x: float  # Normalized position (0-1)
    obstacle_distance: float  # Estimated distance (0-1 scale)
    recommended_action: str  # "forward", "turn_left", "turn_right", "stop"


@dataclass(frozen=True)
class DoorwayScanResult:
    frames: list[np.ndarray]
    doorway_state: NavigationState | None
    rotation_degrees: int

    @property
    def doorway_found(self) -> bool:
        return self.doorway_state is not None


def detect_edges_and_openings(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect edges and potential openings (doorways) in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    return edges, edges_dilated


def detect_vertical_opening(edges: np.ndarray) -> tuple[bool, float]:
    """
    Detect a vertical opening (doorway) in the center of the frame.
    Returns (opening_detected, center_x_normalized)
    AGGRESSIVE: Lower thresholds to find any opening.
    """
    height, width = edges.shape
    
    # Focus on lower half of image (eye level and below)
    lower_half = edges[height // 2:, :]

    # Count edges in vertical slices
    slice_width = max(16, width // 16)
    edge_densities = []

    for x in range(0, width, slice_width):
        x_end = min(x + slice_width, width)
        slice_area = lower_half[:, x:x_end]
        if slice_area.size == 0:
            edge_densities.append(1.0)
            continue
        edge_count = np.sum(slice_area) / 255.0
        density = edge_count / slice_area.size
        edge_densities.append(density)

    if len(edge_densities) == 0:
        return False, 0.5

    avg_density = float(np.mean(edge_densities))
    # AGGRESSIVE: Use much lower threshold (0.5 instead of 0.7)
    thresh = max(0.005, avg_density * 0.5)

    # Find contiguous low-density regions (candidate doorways)
    low_regions = []
    start = None
    for i, d in enumerate(edge_densities):
        if d < thresh:
            if start is None:
                start = i
        else:
            if start is not None:
                low_regions.append((start, i - 1))
                start = None
    if start is not None:
        low_regions.append((start, len(edge_densities) - 1))

    # Require minimum width of 1 slice (very lenient)
    best_region = None
    best_width = 0
    for s, e in low_regions:
        width_slices = e - s + 1
        if width_slices >= 1 and width_slices > best_width:
            best_region = (s, e)
            best_width = width_slices

    if best_region is None:
        return False, 0.5

    s, e = best_region
    center_slice = (s + e) / 2.0
    doorway_center_x = ((center_slice * slice_width) + slice_width / 2.0) / width
    return True, float(max(0.0, min(1.0, doorway_center_x)))


def detect_obstacle_density(edges: np.ndarray) -> float:
    """
    Estimate obstacle density in the frame.
    Returns 0.0 (clear path) to 1.0 (blocked).
    """
    height, width = edges.shape
    total_pixels = height * width
    edge_pixels = np.sum(edges) / 255.0
    density = edge_pixels / total_pixels
    
    return min(1.0, density / 0.3)  # Normalize


def analyze_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """
    Detect motion in the frame using optical flow.
    Returns motion magnitude (0.0 = no motion, 1.0 = high motion).
    """
    if prev_frame is None or curr_frame is None:
        return 0.0
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Calculate magnitude of motion
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mean_magnitude = np.mean(magnitude)
    
    return min(1.0, mean_magnitude / 10.0)


def get_navigation_state(frame: np.ndarray, prev_frame: np.ndarray = None) -> NavigationState:
    """
    Analyze current frame and return navigation state with recommendations.
    """
    edges, edges_dilated = detect_edges_and_openings(frame)
    doorway_detected, doorway_center_x = detect_vertical_opening(edges)
    obstacle_density = detect_obstacle_density(edges)
    motion = analyze_motion(prev_frame, frame) if prev_frame is not None else 0.0
    
    can_move_forward = obstacle_density < 0.5 and doorway_detected
    
    # Determine recommended action
    if not can_move_forward and obstacle_density > 0.6:
        recommended_action = "stop"
    elif doorway_detected:
        # Check if we need to turn toward doorway
        center_offset = abs(doorway_center_x - 0.5)  # 0.5 is frame center
        if center_offset > 0.15:
            recommended_action = "turn_right" if doorway_center_x > 0.5 else "turn_left"
        else:
            recommended_action = "forward"
    else:
        recommended_action = "search"  # Rotate to find doorway
    
    return NavigationState(
        can_move_forward=can_move_forward,
        doorway_detected=doorway_detected,
        doorway_center_x=doorway_center_x,
        obstacle_distance=obstacle_density,
        recommended_action=recommended_action,
    )


def generate_movement_command(nav_state: NavigationState) -> dict[str, int]:
    """
    Convert navigation state to drone movement commands.
    Returns dict with keys: forward/back, left/right, up/down, rotate.
    Values are -100 to 100 (percent of max speed).
    """
    # Command values are interpreted as velocities for send_rc_control:
    # left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity
    command = {
        "forward": 0,  # forward velocity (cm/s)
        "back": 0,
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0,
        "rotate": 0,  # yaw velocity (-100..100)
    }

    if nav_state.recommended_action == "forward":
        command["forward"] = 30
    elif nav_state.recommended_action == "turn_left":
        command["rotate"] = -40
    elif nav_state.recommended_action == "turn_right":
        command["rotate"] = 40
    elif nav_state.recommended_action == "search":
        # slow yaw to look for doorway without overshooting
        command["rotate"] = 30
    elif nav_state.recommended_action == "stop":
        pass

    return command


def apply_movement_command(tello, command: dict[str, int], duration: float = 0.1) -> None:
    """
    Apply movement command to the drone.
    
    Args:
        tello: Tello drone object from djitellopy
        command: Movement command dict from generate_movement_command
        duration: Time to apply command (seconds)
    """
    try:
        # Use discrete move/rotate commands for reliability on flaky connections
        # Map requested values to safe small distances/degrees
        actions = []
        if command.get("forward", 0) > 0:
            dist = min(30, int(command.get("forward", 0)))
            actions.append(("move_forward", dist))
        elif command.get("back", 0) > 0:
            dist = min(30, int(command.get("back", 0)))
            actions.append(("move_back", dist))

        if command.get("left", 0) > 0:
            dist = min(30, int(command.get("left", 0)))
            actions.append(("move_left", dist))
        elif command.get("right", 0) > 0:
            dist = min(30, int(command.get("right", 0)))
            actions.append(("move_right", dist))

        if command.get("up", 0) > 0:
            dist = min(30, int(command.get("up", 0)))
            actions.append(("move_up", dist))
        elif command.get("down", 0) > 0:
            dist = min(30, int(command.get("down", 0)))
            actions.append(("move_down", dist))

        if command.get("rotate", 0) != 0:
            deg = min(45, abs(int(command.get("rotate", 0))))
            if command.get("rotate", 0) > 0:
                actions.append(("rotate_clockwise", deg))
            else:
                actions.append(("rotate_counter_clockwise", deg))

        if actions:
            print("⤴️  Executing actions:", ", ".join(f"{a[0]}({a[1]})" for a in actions))
        for act, val in actions:
            try:
                getattr(tello, act)(int(val))
            except Exception as e:
                print(f"⚠️  Action {act}({val}) failed: {e}")

    except Exception as e:
        print(f"Movement command failed: {e}")


def should_move_to_next_room(nav_state: NavigationState, frames_in_state: int) -> bool:
    """
    Determine if drone has reached a new room.
    Returns True if conditions suggest room transition occurred.
    """
    # If we haven't seen doorway and obstacle density drops, might be in new room
    return (
        nav_state.obstacle_distance < 0.3 and 
        frames_in_state > 10  # Need consistent frames
    )


def capture_360_scan(tello, frame_reader, interval: float = 0.3) -> list[np.ndarray]:
    """
    Perform a 360-degree rotation while capturing frames.
    
    Args:
        tello: Tello drone object from djitellopy
        frame_reader: Frame reader from tello.get_frame_read()
        interval: Time interval between frame captures (seconds)
    
    Returns:
        List of frames captured during the 360-degree rotation
    """
    import time
    
    frames = []
    total_rotation = 0
    # Use smaller steps to improve reliability
    rotation_per_step = 30  # degrees per step (12 steps = 360)
    max_retries = 3

    print("🔄 Starting 360-degree scan...")

    # Capture initial frame
    frame = frame_reader.frame
    if frame is not None:
        frames.append(frame.copy())

    # Perform rotations with retries and fallback
    steps = int(360 // rotation_per_step)
    for i in range(steps):
        success = False
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                tello.rotate_clockwise(rotation_per_step)
                success = True
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2 * attempt)

        # Fallback: try brief yaw via send_rc_control if rotate failed
        if not success and hasattr(tello, "send_rc_control"):
            try:
                yaw_speed = 30 if rotation_per_step > 0 else -30
                tello.send_rc_control(0, 0, 0, int(yaw_speed))
                time.sleep(interval)
                tello.send_rc_control(0, 0, 0, 0)
                success = True
            except Exception as e:
                last_err = e

        if not success:
            print(f"❌ 360-degree scan failed: rotation step {i+1} error: {last_err}")
            break

        total_rotation += rotation_per_step
        time.sleep(interval)

        frame = frame_reader.frame
        if frame is not None:
            frames.append(frame.copy())

        print(f"  📸 Captured frame {len(frames)} at {total_rotation}°")

    if len(frames) > 0:
        print(f"✅ 360-degree scan complete. Captured {len(frames)} frames")
    else:
        print("⚠️ 360-degree scan returned no frames")

    return frames


def capture_360_scan_until_doorway(
    tello,
    frame_reader,
    interval: float = 0.3,
    rotation_per_step: int = 30,
    center_tolerance: float = 0.22,
    max_obstacle_density: float = 0.75,
) -> DoorwayScanResult:
    """
    Rotate up to 360 degrees while looking for a doorway, stopping as soon as one is visible.

    This is meant for the second scan in the mission flow. The regular 360 scan is still used
    for room/person recognition; this scan exits early so the drone can face the doorway and enter.
    """
    import time

    frames: list[np.ndarray] = []
    total_rotation = 0
    max_retries = 3
    if rotation_per_step == 0:
        raise ValueError("rotation_per_step must be non-zero")
    steps = int(360 // abs(rotation_per_step))

    print("Starting doorway scan; stopping when a doorway is visible...")

    def check_current_frame() -> NavigationState | None:
        frame = frame_reader.frame
        if frame is None:
            return None

        frames.append(frame.copy())
        nav_state = get_navigation_state(frame, None)
        if not nav_state.doorway_detected:
            return None

        centered_enough = abs(nav_state.doorway_center_x - 0.5) <= center_tolerance
        clear_enough = nav_state.obstacle_distance <= max_obstacle_density
        if centered_enough or clear_enough:
            return nav_state
        return None

    doorway_state = check_current_frame()
    if doorway_state is not None:
        print(f"Doorway already visible at center_x={doorway_state.doorway_center_x:.2f}")
        return DoorwayScanResult(frames, doorway_state, total_rotation)

    for i in range(steps):
        success = False
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                if rotation_per_step > 0:
                    tello.rotate_clockwise(abs(rotation_per_step))
                else:
                    tello.rotate_counter_clockwise(abs(rotation_per_step))
                success = True
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2 * attempt)

        if not success and hasattr(tello, "send_rc_control"):
            try:
                yaw_speed = 30 if rotation_per_step > 0 else -30
                tello.send_rc_control(0, 0, 0, int(yaw_speed))
                time.sleep(interval)
                tello.send_rc_control(0, 0, 0, 0)
                success = True
            except Exception as e:
                last_err = e

        if not success:
            print(f"Doorway scan failed: rotation step {i+1} error: {last_err}")
            break

        total_rotation += abs(rotation_per_step)
        time.sleep(interval)

        doorway_state = check_current_frame()
        if doorway_state is not None:
            print(
                "Doorway found; stopping scan "
                f"at {total_rotation} degrees, center_x={doorway_state.doorway_center_x:.2f}"
            )
            return DoorwayScanResult(frames, doorway_state, total_rotation)

        print(f"  Doorway not found after {total_rotation} degrees")

    print("Doorway scan completed without a doorway lock")
    return DoorwayScanResult(frames, None, total_rotation)


def align_to_doorway_and_enter(
    tello,
    doorway_state: NavigationState,
    forward_distance_cm: int = 50,
    center_deadband: float = 0.12,
) -> None:
    """Face the doorway from the scan result and move through it."""
    center_error = doorway_state.doorway_center_x - 0.5
    if abs(center_error) > center_deadband:
        turn_degrees = min(35, max(15, int(abs(center_error) * 90)))
        if center_error < 0:
            print(f"Aligning left by {turn_degrees} degrees")
            tello.rotate_counter_clockwise(turn_degrees)
        else:
            print(f"Aligning right by {turn_degrees} degrees")
            tello.rotate_clockwise(turn_degrees)
    else:
        print("Doorway is centered enough to enter")

    print(f"Moving forward {forward_distance_cm} cm through doorway")
    tello.move_forward(forward_distance_cm)
