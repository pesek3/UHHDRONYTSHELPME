from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AnalysisResult:
    room: str | None
    people_count: int
    confidence: float | None
    source: str


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def augment_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness by a factor (0.5 = darker, 1.5 = lighter)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def extract_features(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (160, 120))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    return np.concatenate([hist, edge_density])


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def train_room_classifier(data_dir: Path, model_path: Path) -> dict[str, object]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    # Brightness augmentation factors: darker, original, lighter
    brightness_factors = [0.6, 1.0, 1.4]

    for room_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in iter_images(room_dir):
            image = load_image(image_path)
            room_label = room_dir.name.upper()
            
            # Create augmented versions with different brightness levels
            for brightness_factor in brightness_factors:
                augmented_image = augment_brightness(image, brightness_factor)
                features.append(extract_features(augmented_image))
                labels.append(room_label)

    if len(set(labels)) < 2:
        raise ValueError("Need training images for at least two room folders.")

    x = np.vstack(features)
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if min(np.bincount(np.unique(y, return_inverse=True)[1])) > 1 else None,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    score = float(model.score(x_test, y_test)) if len(y_test) else 1.0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"samples": len(labels), "labels": sorted(set(labels)), "accuracy": score}


def train_people_regressor(data_dir: Path, labels_path: Path, model_path: Path) -> dict[str, object]:
    features: list[np.ndarray] = []
    counts: list[int] = []

    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = data_dir / row["image"]
            features.append(extract_features(load_image(image_path)))
            counts.append(int(row["count"]))

    if len(counts) < 5:
        raise ValueError("Need at least 5 labelled people-count images.")

    x = np.vstack(features)
    y = np.array(counts, dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, predictions)) if len(y_test) else 0.0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"samples": len(counts), "mae": mae}


def detect_people_hog(image: np.ndarray) -> int:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    resized = cv2.resize(image, (640, 480))
    boxes, weights = hog.detectMultiScale(
        resized,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    return int(sum(1 for weight in weights if float(weight) > 0.35) or len(boxes))


def analyze_image(image_path: Path, room_model_path: Path, people_model_path: Path) -> AnalysisResult:
    image = load_image(image_path)
    features = extract_features(image).reshape(1, -1)

    room: str | None = None
    confidence: float | None = None
    if room_model_path.exists():
        room_model = joblib.load(room_model_path)
        room = str(room_model.predict(features)[0])
        if hasattr(room_model, "predict_proba"):
            confidence = float(np.max(room_model.predict_proba(features)))

    if people_model_path.exists():
        people_model = joblib.load(people_model_path)
        people_count = max(0, int(round(float(people_model.predict(features)[0]))))
        source = "regression-model"
    else:
        people_count = detect_people_hog(image)
        source = "opencv-hog-fallback"

    return AnalysisResult(
        room=room,
        people_count=people_count,
        confidence=confidence,
        source=source,
    )
