from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2

from .planner import plan_exploration
from .rooms import ROOM_GRAPH, ROOMS, Room

try:
    from djitellopy import Tello
except ImportError:  # pragma: no cover
    Tello = None

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageTk = None

VIDEO_WIDTH = 360
VIDEO_HEIGHT = 240


ROOM_POSITIONS: dict[Room, tuple[int, int]] = {
    Room.CORRIDOR: (90, 360),
    Room.PRINTERS: (90, 90),
    Room.P1: (330, 210),
    Room.RAI: (330, 360),
    Room.P2: (330, 90),
}


class DroneRescueApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Drone Rescue Planner")
        self.geometry("760x520")
        self.minsize(700, 480)

        self.start_room = tk.StringVar(value=Room.RAI.value)
        self.status = tk.StringVar(value="Ready")
        self.path_labels: list[int] = []

        self.tello: Tello | None = None
        self.frame_reader = None
        self.camera_running = False
        self.video_photo: tk.PhotoImage | None = None
        self.camera_status = tk.StringVar(value="Camera: stopped")

        self._build_layout()
        self._draw_room_graph()
        self.run_simulation()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=18)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Mission Planner", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(sidebar, text="Start room").grid(row=1, column=0, sticky="w", pady=(22, 6))

        start_picker = ttk.Combobox(
            sidebar,
            textvariable=self.start_room,
            values=[room.value for room in Room],
            state="readonly",
            width=18,
        )
        start_picker.grid(row=2, column=0, sticky="ew")
        start_picker.bind("<<ComboboxSelected>>", lambda _: self.run_simulation())

        ttk.Button(sidebar, text="Simulate Route", command=self.run_simulation).grid(
            row=3, column=0, sticky="ew", pady=(14, 0)
        )

        self.camera_button = ttk.Button(sidebar, text="Start camera", command=self._toggle_camera)
        self.camera_button.grid(row=4, column=0, sticky="ew", pady=(14, 0))

        ttk.Separator(sidebar).grid(row=5, column=0, sticky="ew", pady=18)

        ttk.Label(sidebar, text="Visit order", font=("Segoe UI", 11, "bold")).grid(
            row=6, column=0, sticky="w"
        )
        self.visit_order = tk.Listbox(sidebar, height=7, activestyle="none")
        self.visit_order.grid(row=7, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(sidebar, text="Flight path", font=("Segoe UI", 11, "bold")).grid(
            row=8, column=0, sticky="w", pady=(18, 6)
        )
        self.path_text = tk.Text(sidebar, width=28, height=7, wrap="word")
        self.path_text.grid(row=9, column=0, sticky="nsew")
        self.path_text.configure(state="disabled")

        main = ttk.Frame(self, padding=(0, 18, 18, 18))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(2, weight=0)
        main.rowconfigure(3, weight=0)

        self.canvas = tk.Canvas(main, background="#eef4f8", highlightthickness=1, highlightbackground="#d8dde6")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.video_label = tk.Label(
            main,
            text="Camera feed stopped",
            background="#000000",
            foreground="#ffffff",
            anchor="center",
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        camera_status_bar = ttk.Label(main, textvariable=self.camera_status, anchor="w")
        camera_status_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        status_bar = ttk.Label(main, textvariable=self.status, anchor="w")
        status_bar.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _draw_room_graph(self) -> None:
        self.canvas.delete("all")
        self.path_labels.clear()

        drawn_edges: set[frozenset[Room]] = set()
        for room, neighbors in ROOM_GRAPH.items():
            x1, y1 = ROOM_POSITIONS[room]
            for neighbor in neighbors:
                edge = frozenset((room, neighbor))
                if edge in drawn_edges:
                    continue
                drawn_edges.add(edge)
                x2, y2 = ROOM_POSITIONS[neighbor]
                self.canvas.create_line(x1, y1, x2, y2, width=4, fill="#c9d2df", tags="edge")

        for room, (x, y) in ROOM_POSITIONS.items():
            self.canvas.create_oval(
                x - 46,
                y - 34,
                x + 46,
                y + 34,
                fill="#f3fbff",
                outline="#4d7290",
                width=2,
                tags=(f"room:{room.value}", "room"),
            )
            self.canvas.create_text(x, y - 7, text=room.value, font=("Segoe UI", 12, "bold"))
            self.canvas.create_text(
                x,
                y + 14,
                text=ROOMS[room].description,
                font=("Segoe UI", 8),
                fill="#3e5465",
                width=104,
            )

    def run_simulation(self) -> None:
        start = Room(self.start_room.get())
        plan = plan_exploration(start)

        self.visit_order.delete(0, tk.END)
        for index, room in enumerate(plan.visit_order, start=1):
            self.visit_order.insert(tk.END, f"{index}. {room.value}")

        self.path_text.configure(state="normal")
        self.path_text.delete("1.0", tk.END)
        self.path_text.insert("1.0", " -> ".join(room.value for room in plan.path))
        self.path_text.configure(state="disabled")

        self._draw_room_graph()
        self._draw_path(plan.path)
        self.status.set(f"Route planned from {start.value}: {len(plan.path)} flight steps")

    def _draw_path(self, path: tuple[Room, ...]) -> None:
        for index, (room, next_room) in enumerate(zip(path, path[1:]), start=1):
            x1, y1 = ROOM_POSITIONS[room]
            x2, y2 = ROOM_POSITIONS[next_room]
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=6,
                fill="#1f7a8c",
                arrow=tk.LAST,
                arrowshape=(14, 18, 7),
                tags=("path", "path-line"),
            )
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label = self.canvas.create_oval(
                mid_x - 11,
                mid_y - 11,
                mid_x + 11,
                mid_y + 11,
                fill="#f4d35e",
                outline="#a88405",
                tags=("path", "path-label"),
            )
            self.path_labels.append(label)
            self.canvas.create_text(
                mid_x,
                mid_y,
                text=str(index),
                font=("Segoe UI", 8, "bold"),
                tags=("path", "path-label"),
            )

        for room in path:
            self.canvas.itemconfigure(f"room:{room.value}", outline="#1f7a8c", width=3)
        self.canvas.tag_lower("path-line", "room")
        self.canvas.tag_raise("path-label")

    def _toggle_camera(self) -> None:
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        if self.camera_running:
            return
        if Tello is None:
            self.camera_status.set("Camera unavailable: install djitellopy")
            return
        if Image is None or ImageTk is None:
            self.camera_status.set("Camera preview unavailable: install Pillow")
            return

        try:
            self.tello = Tello()
            self.tello.connect()
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read()
            self.camera_running = True
            self.camera_button.configure(text="Stop camera")
            self.camera_status.set("Camera started")
            self._update_camera_frame()
        except Exception as exc:
            self.camera_status.set(f"Camera start failed: {exc}")
            self._cleanup_camera()

    def _stop_camera(self) -> None:
        self._cleanup_camera()
        self.camera_button.configure(text="Start camera")
        self.camera_status.set("Camera stopped")
        self.video_label.configure(image="", text="Camera feed stopped")
        self.video_photo = None

    def _cleanup_camera(self) -> None:
        if self.frame_reader is not None and hasattr(self.frame_reader, "stop"):
            try:
                self.frame_reader.stop()
            except Exception:
                pass
        if self.tello is not None:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            try:
                self.tello.end()
            except Exception:
                pass
        self.frame_reader = None
        self.tello = None
        self.camera_running = False

    def _update_camera_frame(self) -> None:
        if not self.camera_running or self.frame_reader is None:
            return

        frame = getattr(self.frame_reader, "frame", None)
        if frame is not None:
            self._display_video_frame(frame)
        else:
            self.camera_status.set("Waiting for camera feed...")

        self.after(50, self._update_camera_frame)

    def _display_video_frame(self, frame: object) -> None:
        if Image is None or ImageTk is None:
            return

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.BILINEAR)
            self.video_photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=self.video_photo, text="")
        except Exception as exc:
            self.camera_status.set(f"Camera preview error: {exc}")


def main() -> None:
    app = DroneRescueApp()
    app.mainloop()
