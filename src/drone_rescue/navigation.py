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
    """
    height, width = edges.shape
    
    # Focus on lower half of image (eye level and below)
    lower_half = edges[height // 2:, :]
    
    # Count edges in vertical slices
    slice_width = 20
    edge_counts = []
    
    for x in range(0, width, slice_width):
        x_end = min(x + slice_width, width)
        edge_count = np.sum(lower_half[:, x:x_end]) / 255.0
        edge_counts.append(edge_count)
    
    # Find the area with fewest edges (potential doorway)
    if len(edge_counts) == 0:
        return False, 0.5
    
    min_edges_idx = np.argmin(edge_counts)
    min_edges = edge_counts[min_edges_idx]
    
    # If minimum edge count is low, we might have a doorway
    doorway_detected = min_edges < np.mean(edge_counts) * 0.5
    doorway_center_x = (min_edges_idx * slice_width + slice_width // 2) / width
    
    return doorway_detected, doorway_center_x


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
    command = {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0,
        "rotate": 0,  # Positive = clockwise, negative = counter-clockwise
    }
    
    if nav_state.recommended_action == "forward":
        command["forward"] = 50  # Move forward at 50% speed
    elif nav_state.recommended_action == "turn_left":
        command["rotate"] = -30  # Turn left
    elif nav_state.recommended_action == "turn_right":
        command["rotate"] = 30  # Turn right
    elif nav_state.recommended_action == "search":
        command["rotate"] = 20  # Rotate to search for opening
    elif nav_state.recommended_action == "stop":
        pass  # All zeros - no movement
    
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
        # Send movement commands
        # Note: djitellopy uses different command names
        if command["forward"] > 0:
            tello.move_forward(int(command["forward"]))
        elif command["back"] > 0:
            tello.move_back(int(command["back"]))
        
        if command["left"] > 0:
            tello.move_left(int(command["left"]))
        elif command["right"] > 0:
            tello.move_right(int(command["right"]))
        
        if command["up"] > 0:
            tello.move_up(int(command["up"]))
        elif command["down"] > 0:
            tello.move_down(int(command["down"]))
        
        if command["rotate"] != 0:
            # Positive = rotate clockwise
            tello.rotate_clockwise(int(abs(command["rotate"]))) if command["rotate"] > 0 else tello.rotate_counter_clockwise(int(abs(command["rotate"])))
    
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
