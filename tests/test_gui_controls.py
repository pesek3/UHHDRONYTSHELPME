from __future__ import annotations

import numpy as np

from drone_rescue.gui import DroneRescueApp


class FakeWidget:
    def __init__(self, widget_class: str = "Frame") -> None:
        self.widget_class = widget_class

    def winfo_class(self) -> str:
        return self.widget_class


class FakeEvent:
    def __init__(self, keysym: str, widget_class: str = "Frame") -> None:
        self.keysym = keysym
        self.widget = FakeWidget(widget_class)


def test_wasd_keys_trigger_manual_moves() -> None:
    app = object.__new__(DroneRescueApp)
    moves: list[str] = []
    app._manual_move = moves.append
    app._run_360_scan = lambda: None

    assert app._handle_keypress(FakeEvent("w")) == "break"
    assert app._handle_keypress(FakeEvent("A")) == "break"

    assert moves == ["forward", "left"]


def test_j_key_triggers_scan() -> None:
    app = object.__new__(DroneRescueApp)
    scans: list[bool] = []
    app._manual_move = lambda move: None
    app._run_360_scan = lambda: scans.append(True)

    assert app._handle_keypress(FakeEvent("j")) == "break"

    assert scans == [True]


def test_keypress_ignores_text_fields() -> None:
    app = object.__new__(DroneRescueApp)
    moves: list[str] = []
    app._manual_move = moves.append
    app._run_360_scan = lambda: None

    assert app._handle_keypress(FakeEvent("w", widget_class="Entry")) is None

    assert moves == []


def test_save_scan_frames_writes_photos(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("drone_rescue.gui.SCAN_PHOTO_ROOT", tmp_path)
    app = object.__new__(DroneRescueApp)
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.full((4, 4, 3), 255, dtype=np.uint8),
    ]

    scan_dir = app._save_scan_frames(frames)

    assert scan_dir is not None
    assert scan_dir.parent == tmp_path
    assert sorted(path.name for path in scan_dir.glob("*.jpg")) == [
        "frame_01.jpg",
        "frame_02.jpg",
    ]
