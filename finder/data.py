from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


TIME_STEPS = 460
BANDS = 6
PATCH_SIZE = 14
YEARS = 10


def _sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def paired_sample_ids(sample_dir: str | Path, mask_dir: str | Path) -> list[str]:
    samples = {path.stem for path in Path(sample_dir).glob("*.tif")}
    masks = {path.stem for path in Path(mask_dir).glob("*.tif")}
    result = sorted(samples & masks, key=_sort_key)
    if not result:
        raise RuntimeError("No paired TIFF samples and masks were found")
    return result


def split_ids(
    sample_ids: Sequence[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train, validation, and test ratios must sum to 1")
    shuffled = np.asarray(list(sample_ids), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return {
        "train": shuffled[:train_end].tolist(),
        "validation": shuffled[train_end:validation_end].tolist(),
        "test": shuffled[validation_end:].tolist(),
    }


def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    with rasterio.open(path) as source:
        raw = source.read().astype(np.float32)
        profile = source.profile.copy()
    expected_layers = BANDS * TIME_STEPS
    if raw.shape[0] != expected_layers or raw.shape[1] < PATCH_SIZE or raw.shape[2] < PATCH_SIZE:
        raise ValueError(
            f"{path} has shape {raw.shape}; expected ({expected_layers}, >=14, >=14)"
        )
    image = raw[:, :PATCH_SIZE, :PATCH_SIZE].reshape(
        BANDS, TIME_STEPS, PATCH_SIZE, PATCH_SIZE
    ).transpose(1, 0, 2, 3)
    return image.astype(np.float32, copy=False), profile


class TrainingDataset(Dataset):
    def __init__(self, sample_dir: str | Path, mask_dir: str | Path, sample_ids: Sequence[str]):
        self.sample_dir = Path(sample_dir)
        self.mask_dir = Path(mask_dir)
        self.sample_ids = list(sample_ids)
        if not self.sample_ids:
            raise RuntimeError("The requested dataset split is empty")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        image, _ = load_image(self.sample_dir / f"{sample_id}.tif")
        mask_path = self.mask_dir / f"{sample_id}.tif"
        with rasterio.open(mask_path) as source:
            mask_image = source.read().astype(np.float32)
        if mask_image.shape != (YEARS, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"{mask_path} has shape {mask_image.shape}; expected (10, 14, 14)")
        mask_tensor = torch.from_numpy(mask_image)
        active = (mask_tensor.sum(dim=0) != 0).long().reshape(PATCH_SIZE * PATCH_SIZE)
        semantic = mask_tensor.long().reshape(YEARS, -1).transpose(1, 0)
        return torch.from_numpy(image.copy()), (active, semantic), sample_id
