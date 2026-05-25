from drone_rescue.planner import plan_exploration, shortest_path
from drone_rescue.rooms import Room


def test_shortest_path_from_room_to_room_goes_through_corridor():
    assert shortest_path(Room.RAI, Room.P1) == (Room.RAI, Room.CORRIDOR, Room.P1)


def test_shortest_path_from_p1_to_printers_goes_through_p2():
    assert shortest_path(Room.P1, Room.PRINTERS) == (Room.P1, Room.P2, Room.PRINTERS)


def test_shortest_path_from_rai_to_printers_goes_through_p1_and_p2():
    assert shortest_path(Room.RAI, Room.PRINTERS) == (
        Room.RAI,
        Room.P1,
        Room.P2,
        Room.PRINTERS,
    )


def test_plan_exploration_visits_every_room_once():
    plan = plan_exploration(Room.RAI)
    assert set(plan.visit_order) == set(Room)
    assert plan.visit_order[0] == Room.RAI
