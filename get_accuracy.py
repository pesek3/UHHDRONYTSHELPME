import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from drone_rescue.vision import train_room_classifier
    import time
    
    start = time.time()
    result = train_room_classifier(Path('data/training/rooms'), Path('models/room_classifier.joblib'))
    elapsed = time.time() - start
    
    # Write to file so we can read it
    with open('training_output.txt', 'w') as f:
        f.write(f'Training complete in {elapsed:.1f}s\n')
        f.write(f'Samples trained: {result["samples"]}\n')
        f.write(f'Rooms: {result["labels"]}\n')
        f.write(f'Accuracy: {result["accuracy"]:.1%}\n')
        f.write(f'Success percentage: {result["accuracy"]*100:.1f}%\n')
    
except Exception as e:
    with open('training_output.txt', 'w') as f:
        f.write(f'Error: {str(e)}\n')
