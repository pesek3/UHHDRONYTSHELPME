import numpy as np

from drone_rescue.navigation import (
    NavigationState,
    capture_360_scan_until_doorway,
)


class FakeTello:
    def __init__(self, frame_reader):
        self.frame_reader = frame_reader
        self.clockwise_rotations: list[int] = []
        self.forward_moves: list[int] = []

    def rotate_clockwise(self, degrees: int) -> None:
        self.clockwise_rotations.append(degrees)
        self.frame_reader.advance()

    def move_forward(self, distance: int) -> None:
        self.forward_moves.append(distance)


class FakeFrameReader:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    @property
    def frame(self):
        return self.frames[self.index]

    def advance(self):
        self.index = min(self.index + 1, len(self.frames) - 1)


def test_doorway_scan_stops_when_doorway_becomes_visible(monkeypatch):
    frames = [
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.ones((20, 20, 3), dtype=np.uint8),
        np.full((20, 20, 3), 2, dtype=np.uint8),
    ]
    reader = FakeFrameReader(frames)
    tello = FakeTello(reader)

    states = [
        NavigationState(False, False, 0.5, 0.2, "search"),
        NavigationState(True, True, 0.48, 0.2, "forward"),
        NavigationState(True, True, 0.50, 0.2, "forward"),
    ]

    def fake_get_navigation_state(frame, prev_frame=None):
        return states[min(reader.index, len(states) - 1)]

    monkeypatch.setattr(
        "drone_rescue.navigation.get_navigation_state",
        fake_get_navigation_state,
    )

    result = capture_360_scan_until_doorway(
        tello,
        reader,
        interval=0,
        rotation_per_step=30,
    )

    assert result.doorway_found
    assert result.rotation_degrees == 30
    assert tello.clockwise_rotations == [30]
