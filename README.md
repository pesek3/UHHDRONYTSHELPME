# Dron Search & Rescue

Prototype for DJI Tello indoor search and rescue.

Goal: the drone explores the RAI, P1, P2, 3D printer area, and nearby corridor, recognizes where it starts, plans a route, captures photos, and estimates how many people are in each room.

## What Is Inside

- State-space search over a room graph.
- Room classification from images.
- People-count regression / detection from images.
- Optional DJI Tello photo collection.
- Simulation mode for development without the drone.

## Project Structure

```text
data/
  raw/                 photos collected from the drone
  training/
    rooms/             room classifier training images, one folder per room
    people_counts/     images plus labels.csv for people-count regression
models/                trained model files
src/drone_rescue/      application code
```

## Install

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Simulation

```powershell
python -m drone_rescue simulate --start RAI
```

or from the project root:

```powershell
python app.py simulate --start RAI
```

## GUI Planner

Open the route planner window:

```powershell
python app.py gui
```

Choose a start room, then click `Simulate Route` to view the visit order and flight path.

## Collect Photos With Tello

Connect to the Tello Wi-Fi first, then run:

```powershell
python -m drone_rescue collect --room RAI --count 20
```

The photos will be saved into `data/raw/RAI/`.

## Train Models

Room classification expects folders like:

```text
data/training/rooms/RAI/*.jpg
data/training/rooms/P1/*.jpg
data/training/rooms/P2/*.jpg
data/training/rooms/PRINTERS/*.jpg
data/training/rooms/CORRIDOR/*.jpg
```

Train the room classifier:

```powershell
python -m drone_rescue train-room
```

People-count regression expects `data/training/people_counts/labels.csv`:

```csv
image,count
rai_001.jpg,3
p1_001.jpg,0
```

Train the people-count regressor:

```powershell
python -m drone_rescue train-people
```

## Run Inference On One Image

```powershell
python -m drone_rescue analyze path\to\image.jpg
```

## Run The Mission Flow

After training the room classifier:

```powershell
python -m drone_rescue mission path\to\start_image.jpg
```

This recognizes the start room and prints the planned exploration path.

## Flight Safety Notes

- Test every command without propellers or with the drone safely grounded first.
- Keep a person ready to press emergency stop.
- Indoor autonomous flight should use slow movement and short distances.
- The included flight code starts as photo collection only; full autonomous movement should be added after reliable detection and route tests.
