#!/usr/bin/env python3
"""Train a simple CNN on the HuggingFace MNIST parquet dataset."""
from __future__ import annotations

import argparse
import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HFImageDataset(Dataset):
    """Wrap a HuggingFace dataset so it behaves like a PyTorch dataset."""

    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        image = sample["image"].convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple forward
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_parquet_files(data_dir: Path) -> Dict[str, List[str]]:
    data_files: Dict[str, List[str]] = {}
    for split in ("train", "test"):
        matches = sorted(glob(str(data_dir / f"{split}-*.parquet")))
        if not matches:
            raise FileNotFoundError(f"No parquet files found for split '{split}' in {data_dir}")
        data_files[split] = matches
    return data_files


def load_mnist(data_dir: Path):
    data_files = discover_parquet_files(data_dir)
    print(f"Loading MNIST parquet files from {data_files}")
    return load_dataset("parquet", data_files=data_files)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST classifier with PyTorch")
    parser.add_argument("--data-dir", type=Path, default=Path("mnist-dataset/mnist"), help="Directory containing train/test parquet files")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Where to store the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    dataset_dict = load_mnist(args.data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = HFImageDataset(dataset_dict["train"], transform)
    test_dataset = HFImageDataset(dataset_dict["test"], transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Training on {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={acc*100:.2f}%")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "mnist_cnn.pt"
    torch.save({"model_state_dict": model.state_dict(), "epochs": args.epochs}, output_path)
    print(f"Model saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
