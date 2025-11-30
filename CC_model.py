import os
from pathlib import Path
import re
import json
import time
import random
import shutil

# Data analysis & visualization
import numpy as np
import matplotlib.pyplot as plt

# Image processing
import cv2
from PIL import Image

# Progress bar
from tqdm import tqdm

# Metrics & statistics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# TorchVision
import torchvision.transforms as T
from torchvision import models
from collections import deque, Counter
from torchvision.datasets import ImageFolder
import torch.optim as optim

# MediaPipe for face detection
import mediapipe as mp

TRAIN_MODEL = False

# --- Directories ---
MODEL_FOLDER = "models"
SAMPLE_FOLDER = "sample"
# --- Dataset paths (relative to this script). Can be overridden with env vars TRAIN_FOLDER / VALID_FOLDER ---
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive" / "FaceShape Dataset"
TRAIN_FOLDER = os.environ.get('TRAIN_FOLDER', str(ARCHIVE_DIR / 'training_set'))
VALID_FOLDER = os.environ.get('VALID_FOLDER', str(ARCHIVE_DIR / 'testing_set'))

# --- Training Hyperparameters ---
SEED = 42
BATCH_SIZE = 32  # Optimized for GPU training
LEARNING_RATE = 0.001
EPOCHS = 50  # Full training for better accuracy
IMG_SIZE = 224
PATIENCE = 5 

# --- System Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 # Set to 0 for compatibility on Windows


# ===================================================================
# SECTION 1: SKIN TONE ANALYZER
# ===================================================================
class SkinToneAnalyzer:
    def analyze_skin_tone(self, image):
        """Analyze skin tone from a pre-cropped face image."""
        try:
            # Isolate Skin Pixels using HSV color space
            face_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            skin_regions = cv2.bitwise_and(image, image, mask=skin_mask)

            # Find Dominant Colors using KMeans
            skin_rgb = cv2.cvtColor(skin_regions, cv2.COLOR_BGR2RGB)
            pixels = skin_rgb.reshape(-1, 3)
            non_black_pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

            if len(non_black_pixels) < 50: # Need enough pixels
                return "Unknown"

            kmeans = KMeans(n_clusters=3, random_state=SEED, n_init='auto')
            kmeans.fit(non_black_pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)

            return self._classify_skin_undertone(dominant_colors)
        except Exception:
            return "Error"

    def _classify_skin_undertone(self, dominant_colors):
        """Classify undertone based on dominant RGB colors."""
        warm_score = 0
        cool_score = 0
        
        for color in dominant_colors:
            r, g, b = color
            # Simple logic: more red/yellow leaning is warm, more blue is cool
            if r > b and g > b:
                warm_score += 1
            if b > r and b > g:
                cool_score += 1
        
        if warm_score > cool_score:
            return 'Warm'
        elif cool_score > warm_score:
            return 'Cool'
        else:
            return 'Neutral'


# ===================================================================
# SECTION 2: FACE SHAPE MODEL TRAINING
# ===================================================================
class SafeImageFolder(ImageFolder):
    """Custom ImageFolder to handle corrupted images."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception:
            # Skip corrupted image by loading the next one
            return self.__getitem__((index + 1) % len(self.samples))

def seed_everything(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed} for reproducibility.")

def train_model():
    """Main function to orchestrate the training and evaluation."""
    seed_everything(SEED)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15), T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = SafeImageFolder(TRAIN_FOLDER, transform=train_transform)
    val_dataset = SafeImageFolder(VALID_FOLDER, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    num_classes = len(train_dataset.classes)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_f1 = 0.0
    early_stop_counter = 0
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} -> Validation F1-score: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'best_f1_model.pth'))
            print(f"✅ New best model saved with F1-score: {best_f1:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print("❌ Early stopping triggered.")
            break
        scheduler.step(val_f1)
    print("\n--- Training Complete ---")


# ===================================================================
# SECTION 3: REAL-TIME ANALYSIS
# ===================================================================
def run_realtime_analysis():
    """Load models and run real-time analysis from webcam."""
    print("--- Starting Real-Time Analysis ---")
    # Validate dataset paths before attempting to load ImageFolder
    if not os.path.isdir(TRAIN_FOLDER) or not os.path.isdir(VALID_FOLDER):
        print("ERROR: Dataset folders not found at expected locations.")
        print(f"  TRAIN_FOLDER: '{TRAIN_FOLDER}'")
        print(f"  VALID_FOLDER: '{VALID_FOLDER}'")
        print("Place the 'FaceShape Dataset' folder under the repository 'archive' directory,")
        print("or set environment variables 'TRAIN_FOLDER' and 'VALID_FOLDER' to the correct paths.")
        return
    # --- 1. Load Face Shape Model ---
    best_model_path = os.path.join(MODEL_FOLDER, 'best_f1_model.pth')
    if not os.path.exists(best_model_path):
        print(f"ERROR: Trained model not found at '{best_model_path}'.")
        print("Please set TRAIN_MODEL = True and run the script to train first.")
        return
        
    # We need the number of classes and class names from the dataset
    temp_dataset = ImageFolder(TRAIN_FOLDER)
    num_classes = len(temp_dataset.classes)
    class_names = temp_dataset.classes
    del temp_dataset

    model = models.resnet18(weights=None) # No need for pre-trained weights here
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- 2. Initialize Analyzers ---
    skin_analyzer = SkinToneAnalyzer()
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

    # --- 3. Define Prediction Transform ---
    # This must match the validation transform from training
    predict_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 4. Start Webcam Loop ---
    print("🎥 Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- 4.1: Temporal smoothing buffers ---
    pred_buffer = deque(maxlen=15)  # recent predicted class indices
    conf_buffer = deque(maxlen=15)  # recent confidences
    stable_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            # Select the largest detected face to stabilize which face we track
            h, w, _ = frame.shape
            def det_to_box_area(det):
                b = det.location_data.relative_bounding_box
                return int(b.width * w) * int(b.height * h)
            detection = max(results.detections, key=det_to_box_area)

            bboxC = detection.location_data.relative_bounding_box
            x, y, cw, ch = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Build a square, margin-expanded crop around the detected box
            side = max(cw, ch)
            margin_factor = 0.2  # 20% margin around the face
            side = int(side * (1.0 + margin_factor))
            cx = x + cw // 2
            cy = y + ch // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w - 1, x1 + side)
            y2 = min(h - 1, y1 + side)
            # Re-adjust top-left if clipped on the right/bottom
            x1 = max(0, x2 - side)
            y1 = max(0, y2 - side)

            # Crop face ROI (BGR)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                # --- Task 1: Face Shape Prediction ---
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                input_tensor = predict_transform(face_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)

                # Push into buffers
                pred_idx_int = predicted_idx.item()
                pred_buffer.append(pred_idx_int)
                conf_buffer.append(confidence.item())

                # Majority vote smoothing
                display_idx = pred_idx_int
                if len(pred_buffer) >= 5:
                    counts = Counter(pred_buffer)
                    mode_idx, mode_count = counts.most_common(1)[0]
                    agree_ratio = mode_count / len(pred_buffer)
                    avg_conf_for_mode = sum(c for p, c in zip(pred_buffer, conf_buffer) if p == mode_idx) / max(1, sum(1 for p in pred_buffer if p == mode_idx))
                    if agree_ratio >= 0.6 and avg_conf_for_mode >= 0.6:
                        display_idx = mode_idx

                # Update stable label only when smoothed decision changes decisively
                if stable_label is None:
                    stable_label = display_idx
                elif display_idx != stable_label:
                    # Require strong agreement to switch
                    counts = Counter(pred_buffer)
                    mode_idx, mode_count = counts.most_common(1)[0]
                    if mode_idx == display_idx and (mode_count >= int(0.7 * len(pred_buffer))):
                        stable_label = display_idx

                shape_label = class_names[stable_label]
                shape_conf = (sum(conf_buffer) / len(conf_buffer) if len(conf_buffer) > 0 else confidence.item()) * 100
                shape_text = f"Shape: {shape_label} ({shape_conf:.1f}%)"

                # --- Task 2: Skin Tone Analysis ---
                tone_label = skin_analyzer.analyze_skin_tone(face_roi)
                tone_text = f"Undertone: {tone_label}"

                # --- Display Results ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, shape_text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, tone_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Analysis (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # This block controls whether to train or run analysis
    if TRAIN_MODEL:
        train_model()
    else:
        run_realtime_analysis()