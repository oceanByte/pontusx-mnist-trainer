#!/usr/bin/env python3
"""Train a simple CNN on the HuggingFace MNIST parquet dataset."""
from __future__ import annotations

import argparse
import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import tarfile
import tempfile
import zipfile


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


def discover_parquet_files(data_root: Path) -> Dict[str, List[str]]:
    data_files: Dict[str, List[str]] = {}
    for split in ("train", "test"):
        matches = sorted(str(path) for path in data_root.rglob(f"{split}-*.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"No parquet files found for split '{split}' under {data_root}"
            )
        data_files[split] = matches
    return data_files


def load_mnist(data_root: Path):
    data_files = discover_parquet_files(data_root)
    print(f"Loading MNIST parquet files from {data_files}")
    return load_dataset("parquet", data_files=data_files)


def is_supported_archive(path: Path) -> bool:
    return (path.is_file() and (tarfile.is_tarfile(path) or zipfile.is_zipfile(path)))


def extract_archive(archive_path: Path) -> Tuple[Path, tempfile.TemporaryDirectory]:
    temp_dir = tempfile.TemporaryDirectory()
    dest = Path(temp_dir.name)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar:
            tar.extractall(dest)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    else:
        temp_dir.cleanup()
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return dest, temp_dir


def resolve_data_root(data_dir: Path, data_archive: Optional[Path]) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if data_archive:
        print(f"Extracting archive {data_archive}")
        return extract_archive(data_archive)

    data_dir = data_dir.expanduser().resolve()
    if data_dir.is_file() and is_supported_archive(data_dir):
        print(f"Extracting archive {data_dir}")
        return extract_archive(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # If the directory already has parquet files we can use it directly.
    if list(data_dir.rglob("*.parquet")):
        return data_dir, None

    # Otherwise, see if the directory contains an archive we can unpack.
    for candidate in data_dir.iterdir():
        if is_supported_archive(candidate):
            print(f"Extracting archive {candidate}")
            return extract_archive(candidate)

    raise FileNotFoundError(f"No parquet files or supported archives found under {data_dir}")


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data"),
        help="Directory containing train/test parquet files or archives",
    )
    parser.add_argument(
        "--data-archive",
        type=Path,
        default=None,
        help="Optional explicit path to a tar/zip archive containing the parquet files",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Where to store the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    data_root, temp_dir = resolve_data_root(args.data_dir, args.data_archive)
    try:
        dataset_dict = load_mnist(data_root)

        transform = transforms.Compose([
            transforms.ToTensor(),  # converts to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = HFImageDataset(dataset_dict["train"], transform)
        test_dataset = HFImageDataset(dataset_dict["test"], transform)

        num_workers = max(1, (os.cpu_count() or 1) // 2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
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
    finally:
        if temp_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
