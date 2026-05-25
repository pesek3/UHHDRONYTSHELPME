from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Room(str, Enum):
    RAI = "RAI"
    P1 = "P1"
    P2 = "P2"
    PRINTERS = "PRINTERS"
    CORRIDOR = "CORRIDOR"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RoomNode:
    room: Room
    description: str


ROOMS: dict[Room, RoomNode] = {
    Room.RAI: RoomNode(Room.RAI, "RAI classroom"),
    Room.P1: RoomNode(Room.P1, "P1 classroom (next to P2)"),
    Room.P2: RoomNode(Room.P2, "P2 classroom (connected to P1 and printers)"),
    Room.PRINTERS: RoomNode(Room.PRINTERS, "3D printer area"),
    Room.CORRIDOR: RoomNode(Room.CORRIDOR, "main corridor"),
}

# Room topology updated to match the sketch: corridor connects to RAI, P1, and printers,
# while P1 connects onward to P2 and P2 connects to the printer area.
ROOM_GRAPH: dict[Room, tuple[Room, ...]] = {
    Room.CORRIDOR: (Room.RAI, Room.P1, Room.PRINTERS),
    Room.RAI: (Room.CORRIDOR,),
    Room.P1: (Room.CORRIDOR, Room.P2),
    Room.P2: (Room.P1, Room.PRINTERS),
    Room.PRINTERS: (Room.CORRIDOR, Room.P2),
}


def parse_room(value: str) -> Room:
    normalized = value.strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "3D": Room.PRINTERS,
        "3D_PRINTERS": Room.PRINTERS,
        "PRINTER": Room.PRINTERS,
        "TISKARNY": Room.PRINTERS,
        "TISKÁRNY": Room.PRINTERS,
        "CHODBA": Room.CORRIDOR,
    }
    if normalized in aliases:
        return aliases[normalized]
    return Room(normalized)
