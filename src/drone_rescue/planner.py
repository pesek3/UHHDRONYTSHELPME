from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .rooms import ROOM_GRAPH, Room


@dataclass(frozen=True)
class ExplorationPlan:
    start: Room
    visit_order: tuple[Room, ...]
    path: tuple[Room, ...]


def shortest_path(start: Room, goal: Room) -> tuple[Room, ...]:
    if start == goal:
        return (start,)

    queue: deque[tuple[Room, tuple[Room, ...]]] = deque([(start, (start,))])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        for neighbor in ROOM_GRAPH[current]:
            if neighbor in visited:
                continue
            next_path = (*path, neighbor)
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))

    raise ValueError(f"No route from {start} to {goal}")


def nearest_unvisited(current: Room, unvisited: set[Room]) -> tuple[Room, tuple[Room, ...]]:
    best_room: Room | None = None
    best_path: tuple[Room, ...] | None = None

    for room in sorted(unvisited):
        path = shortest_path(current, room)
        if best_path is None or len(path) < len(best_path):
            best_room = room
            best_path = path

    if best_room is None or best_path is None:
        raise ValueError("No unvisited room available")

    return best_room, best_path


def plan_exploration(start: Room, targets: set[Room] | None = None) -> ExplorationPlan:
    remaining = set(targets or ROOM_GRAPH.keys())
    remaining.discard(start)

    current = start
    visit_order = [start]
    full_path = [start]

    while remaining:
        next_room, leg = nearest_unvisited(current, remaining)
        visit_order.append(next_room)
        full_path.extend(leg[1:])
        remaining.remove(next_room)
        current = next_room

    return ExplorationPlan(
        start=start,
        visit_order=tuple(visit_order),
        path=tuple(full_path),
    )
