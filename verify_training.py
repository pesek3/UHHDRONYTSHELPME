import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from drone_rescue.vision import train_room_classifier
import time

start = time.time()
result = train_room_classifier(Path('data/training/rooms'), Path('models/room_classifier.joblib'))
elapsed = time.time() - start

print(f'✅ Training complete in {elapsed:.1f}s')
print(f'Samples trained: {result["samples"]} (with 3x brightness variations)')
print(f'Rooms: {result["labels"]}')
print(f'Accuracy: {result["accuracy"]:.1%}')
