import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision import models
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score, f1_score

# Reuse configuration and helpers from the main module
from CC_model import (
	IMG_SIZE,
	MODEL_FOLDER,
	TRAIN_FOLDER,
	VALID_FOLDER,
	device,
	SafeImageFolder,
)


def build_val_loader() -> Tuple[DataLoader, int]:
	"""Create the validation dataloader and return it with num_classes."""
	val_transform = T.Compose([
		T.Resize((IMG_SIZE, IMG_SIZE)),
		T.ToTensor(),
		T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

	val_dataset = SafeImageFolder(VALID_FOLDER, transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

	# Determine number of classes from training folder (consistent with training)
	temp_dataset = ImageFolder(TRAIN_FOLDER)
	num_classes = len(temp_dataset.classes)
	return val_loader, num_classes


def load_model(num_classes: int) -> nn.Module:
	"""Load the trained model with the correct classifier head."""
	model_path = os.path.join(MODEL_FOLDER, 'best_f1_model.pth')
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model not found at {model_path}. Train first to generate it.")

	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model = model.to(device)
	model.eval()
	return model


def evaluate() -> Tuple[float, float]:
	"""Compute accuracy and macro F1 on the validation set."""
	val_loader, num_classes = build_val_loader()
	model = load_model(num_classes)

	all_preds = []
	all_labels = []
	with torch.no_grad():
		for images, labels in val_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			preds = outputs.argmax(dim=1)
			all_preds.extend(preds.cpu().tolist())
			all_labels.extend(labels.cpu().tolist())

	acc = accuracy_score(all_labels, all_preds)
	f1 = f1_score(all_labels, all_preds, average='macro')
	return acc, f1


if __name__ == "__main__":
	acc, f1 = evaluate()
	print(f"Accuracy: {acc:.4f}")
	print(f"Macro F1: {f1:.4f}")
