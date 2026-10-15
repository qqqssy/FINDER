from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import FinderModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def semantic_loss(masks: torch.Tensor, labels: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    masks = masks.reshape(masks.shape[0] * masks.shape[1], -1)
    labels = labels.clone().reshape(labels.shape[0] * labels.shape[1], -1)
    labels[(masks == 0).expand_as(labels)] = -1
    valid_rows = ((labels == 0) | (labels == 1)).any(dim=1)
    one_mask = labels == 1
    first_one = one_mask.float().argmax(dim=1)
    rows_with_one = one_mask.any(dim=1) & valid_rows

    rows, times = labels.shape
    soft = torch.zeros(rows, times, output.shape[-1], device=output.device)
    hard_mask = labels >= 0
    hard_indexes = labels.clone()
    hard_indexes[~hard_mask] = 0
    soft.scatter_(2, hard_indexes.unsqueeze(-1), 1.0)
    soft[~hard_mask] = 0.0
    change_positions = torch.zeros(rows, times, dtype=torch.bool, device=output.device)
    change_positions[rows_with_one, first_one[rows_with_one]] = True
    soft[change_positions] = 0.5

    logits = output.reshape(rows, times, output.shape[-1])
    token_loss = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1) * hard_mask.float()
    loss = token_loss.sum() / hard_mask.float().sum().clamp(min=1)
    return loss


def build_model(config: dict[str, Any], device: torch.device) -> FinderModel:
    model = FinderModel(config["model"])
    model.to(device)
    return model


def initialize_for_training(model: FinderModel, config: dict[str, Any]) -> None:
    model.configure_prithvi(
        initialization="pretrained",
        freeze_backbone=True,
        pretrained_path=config["paths"]["prithvi_weights"],
    )


def load_checkpoint(model: FinderModel, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}


def predict_years(logits: torch.Tensor, entropy_threshold: float) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=-1)
    predictions = entropy.argmax(dim=-1)
    predictions[entropy.max(dim=-1).values < entropy_threshold] = -1
    return predictions


def set_saliency_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def sliding_baseline(images: np.ndarray, window: int) -> np.ndarray:
    baseline = np.zeros_like(images)
    initial = images[: min(window, images.shape[0])].mean(axis=0)
    for index in range(images.shape[0]):
        if index == 0:
            baseline[index] = initial
        elif index < window:
            baseline[index] = images[:index].mean(axis=0)
        else:
            baseline[index] = images[index - window : index].mean(axis=0)
    return baseline


def integrated_gradients(
    model: nn.Module,
    device: torch.device,
    images: np.ndarray,
    row: int,
    column: int,
    target_year: int,
    steps: int,
    baseline_window: int,
) -> np.ndarray:
    set_saliency_mode(model)
    
    baseline = images.copy() 
    target_end = min((target_year + 1) * 46, images.shape[0])
    pixel_baseline = sliding_baseline(images[:, :, row, column], baseline_window)
    baseline[:target_end, :, row, column] = pixel_baseline[:target_end]
    delta = images - baseline
    
    gradient_sum = torch.zeros((images.shape[0], images.shape[1]), device=device)
    for alpha in np.linspace(0.0, 1.0, steps, dtype=np.float32):
        path = torch.from_numpy(baseline + alpha * delta).float().unsqueeze(0).to(device)
        path.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(path).reshape(1, 14 * 14, 10, 2)
        output[:, row * 14 + column, target_year, 1].sum().backward()
        gradient_sum += path.grad[0, :, :, row, column].detach()
    attribution = torch.from_numpy(delta[:, :, row, column]).float().to(device)
    return (attribution * (gradient_sum / steps)).cpu().numpy()


def threshold_index(curve: np.ndarray, target_year: int, ratio: float) -> int:
    start, end = target_year * 46, min((target_year + 1) * 46, len(curve))
    maximum_index = start + int(np.argmax(curve[start:end]))
    maximum = curve[maximum_index]
    for index in range(maximum_index - 1, -1, -1):
        if curve[index] > maximum:
            return index + 1
        if curve[index] <= maximum * ratio:
            return index
    return maximum_index
