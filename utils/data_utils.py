import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from utils.config import DATA_DIR, DOGS_VS_CATS_DIR, IMAGE_SIZE, RANDOM_SEED


def get_transforms(model_type: str):
    """
    Return preprocessing for generative training.

    VAE decoder uses Sigmoid -> pixels in [0, 1].
    DCGAN uses Tanh -> pixels in [-1, 1].
    """
    resize_crop = [
        T.Resize(IMAGE_SIZE + 8),
        T.CenterCrop(IMAGE_SIZE),
    ]

    if model_type == "vae":
        return T.Compose(resize_crop + [T.ToTensor()])

    if model_type == "dcgan":
        return T.Compose(
            resize_crop
            + [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

    raise ValueError(f"Unknown model_type: {model_type}")


class ImageFolderFlat(Dataset):
    """
    Load all images from a folder (searches recursively for .jpg/.jpeg/.png).
    Works when images are not organized in class subfolders.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            p
            for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class CatsAndDogsDataset(Dataset):
    """
    Combined cats + dogs for stage 5.
    Expects Dogs vs Cats layout: train/cat.*.jpg and train/dog.*.jpg
    plus optional extra cat images in DATA_DIR.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, cats_dir: str, dogs_vs_cats_train_dir: str, transform=None):
        self.transform = transform
        self.samples = []  # list of (path, label) — 0=cat, 1=dog

        cats_path = Path(cats_dir)
        if cats_path.exists():
            for p in sorted(cats_path.rglob("*")):
                if p.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(p), 0))

        dvc = Path(dogs_vs_cats_train_dir)
        if dvc.exists():
            for p in sorted(dvc.iterdir()):
                if p.suffix.lower() not in self.EXTENSIONS:
                    continue
                name = p.name.lower()
                if name.startswith("cat"):
                    self.samples.append((str(p), 0))
                elif name.startswith("dog"):
                    self.samples.append((str(p), 1))

        if not self.samples:
            raise FileNotFoundError(
                "No images for cats_and_dogs. Check DATA_DIR and DOGS_VS_CATS_DIR in config.py."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _apply_subset(dataset, subset_ratio: float):
    if subset_ratio >= 1.0:
        return dataset

    rng = np.random.default_rng(RANDOM_SEED)
    n = max(1, int(len(dataset) * subset_ratio))
    indices = rng.choice(len(dataset), size=n, replace=False)
    print(f"Using subset: {n} / {len(dataset)} images ({subset_ratio * 100:.0f}%)")
    return Subset(dataset, indices.tolist())


def build_dataset(config: dict):
    model_type = config["model"]
    dataset_name = config.get("dataset", "cats")
    transform = get_transforms(model_type)

    if dataset_name == "cats":
        dataset = ImageFolderFlat(DATA_DIR, transform=transform)
        return _apply_subset(dataset, config.get("subset_ratio", 1.0))

    if dataset_name == "cats_and_dogs":
        train_dir = os.path.join(DOGS_VS_CATS_DIR, "train")
        dataset = CatsAndDogsDataset(DATA_DIR, train_dir, transform=transform)
        return _apply_subset(dataset, config.get("subset_ratio", 1.0))

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(config: dict, num_workers: int = 2) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
