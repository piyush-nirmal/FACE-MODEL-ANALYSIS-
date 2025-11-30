# ============================================================
# Face Shape Classification + Skin Tone Analyzer (Improved)
# ============================================================

# --- Imports ---
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision import models
from torchvision.datasets import ImageFolder

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())


# ============================================================
# CONFIG
# ============================================================
TRAIN_MODEL = False 

MODEL_FOLDER = "models"
SAMPLE_FOLDER = "sample"

TRAIN_FOLDER = r"D:\AuraSync\Face_Model\archive\FaceShape Dataset\training_set"
VALID_FOLDER = r"D:\AuraSync\Face_Model\archive\FaceShape Dataset\testing_set"

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
IMG_SIZE = 224
PATIENCE = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0  # For Windows compatibility


# ============================================================
# SECTION 1: SKIN TONE ANALYZER
# ============================================================
class SkinToneAnalyzer:
    def analyze_skin_tone(self, image):
        """Analyze skin tone from a cropped face image."""
        try:
            face_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            skin_regions = cv2.bitwise_and(image, image, mask=skin_mask)

            skin_rgb = cv2.cvtColor(skin_regions, cv2.COLOR_BGR2RGB)
            pixels = skin_rgb.reshape(-1, 3)
            non_black_pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

            if len(non_black_pixels) < 50:
                return "Unknown"

            kmeans = KMeans(n_clusters=3, random_state=SEED, n_init="auto")
            kmeans.fit(non_black_pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)

            return self._classify_skin_undertone(dominant_colors)
        except Exception:
            return "Error"

    def _classify_skin_undertone(self, dominant_colors):
        warm_score, cool_score = 0, 0
        for r, g, b in dominant_colors:
            if r > b and g > b:
                warm_score += 1
            if b > r and b > g:
                cool_score += 1
        if warm_score > cool_score:
            return "Warm"
        elif cool_score > warm_score:
            return "Cool"
        else:
            return "Neutral"


# ============================================================
# SECTION 2: DATASET WRAPPER
# ============================================================
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
            return self.__getitem__((index + 1) % len(self.samples))


# ============================================================
# SECTION 3: TRAINING
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed} for reproducibility.")


def train_model():
    seed_everything(SEED)
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # --- Data transforms ---
    train_transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    val_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SafeImageFolder(TRAIN_FOLDER, transform=train_transform)
    val_dataset = SafeImageFolder(VALID_FOLDER, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_dataset.classes)

    # --- Model (ResNet50 + Dropout) ---
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    model = base_model.to(device)

    # --- Loss & Optimizer ---
    class_counts = np.bincount([label for _, label in train_dataset.samples])
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, early_stop_counter = 0.0, 0

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch+1}/{EPOCHS} -> Validation F1-score: {val_f1:.4f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix (Epoch {epoch+1})")
        plt.savefig(os.path.join(MODEL_FOLDER, f"confusion_matrix_epoch{epoch+1}.png"))
        plt.close()

        # --- Save best model ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, "best_f1_model_resnet50.pth"))
            print(f"✅ New best model (ResNet50) saved with F1-score: {best_f1:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print("❌ Early stopping triggered.")
            break

        scheduler.step()

    print("\n--- Training Complete ---")


# ============================================================
# SECTION 4: SINGLE IMAGE PREDICTION
# ============================================================
def load_model():
    """Load best trained ResNet50 model for inference."""
    best_model_path = os.path.join(MODEL_FOLDER, "best_f1_model_resnet50.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best ResNet50 model not found. Train the model first.")

    temp_dataset = ImageFolder(TRAIN_FOLDER)
    class_names = temp_dataset.classes
    del temp_dataset

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, class_names


def predict_face_shape(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face shape inference/training")
    parser.add_argument("--image", type=str, default="", help="Path to an image for inference. If omitted, uses first image in testing set.")
    parser.add_argument("--train", action="store_true", help="Run training instead of inference.")
    args = parser.parse_args()

    if args.train or TRAIN_MODEL:
        train_model()
    else:
        # --- Run a test prediction on a single image ---
        model, class_names = load_model()
        predict_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        test_img = args.image
        if not test_img:
            # find first image in validation folder
            for root, _, files in os.walk(VALID_FOLDER):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        test_img = os.path.join(root, fname)
                        break
                if test_img:
                    break
        if not test_img:
            raise FileNotFoundError("No image found. Provide --image or populate testing_set.")

        shape = predict_face_shape(test_img, model, predict_transform, class_names)
        print(f"Image: {test_img}")
        print(f"Predicted Face Shape: {shape}")
