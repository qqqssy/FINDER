from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finder.config import load_config, select_device
from finder.data import TrainingDataset, paired_sample_ids, split_ids
from finder.runtime import build_model, initialize_for_training, semantic_loss, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FINDER with a 60/20/20 split")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    return parser.parse_args()


def valid_label_count(masks: torch.Tensor, semantic: torch.Tensor) -> int:
    return int(((masks != 0).unsqueeze(-1) & (semantic >= 0)).sum().item())


def mean_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    total_valid = 0
    with torch.inference_mode():
        for images, labels, _ in loader:
            masks, semantic = [value.to(device) for value in labels]
            loss = semantic_loss(masks, semantic, model(images.to(device)))
            valid = valid_label_count(masks, semantic)
            total += float(loss) * valid
            total_valid += valid
    if total_valid == 0:
        raise ValueError("Validation split contains no valid labels")
    return total / total_valid


def main() -> None:
    arguments = parse_args()
    config, _ = load_config(arguments.config)
    device = select_device(arguments.device)
    seed = int(config["split"]["seed"])
    set_seed(seed)

    sample_ids = paired_sample_ids(config["paths"]["sample_dir"], config["paths"]["mask_dir"])
    splits = split_ids(
        sample_ids,
        float(config["split"]["train"]),
        float(config["split"]["validation"]),
        float(config["split"]["test"]),
        seed,
    )
    print("Split sizes:", {name: len(values) for name, values in splits.items()})

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        TrainingDataset(config["paths"]["sample_dir"], config["paths"]["mask_dir"], splits["train"]),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        TrainingDataset(
            config["paths"]["sample_dir"], config["paths"]["mask_dir"], splits["validation"]
        ),
        shuffle=False,
        **loader_options,
    )

    model = build_model(config, device)
    initialize_for_training(model, config)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    checkpoint_path = Path(config["paths"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        training_total = 0.0
        training_valid = 0
        for images, labels, _ in train_loader:
            masks, semantic = [value.to(device) for value in labels]
            optimizer.zero_grad(set_to_none=True)
            loss = semantic_loss(masks, semantic, model(images.to(device)))
            loss.backward()
            optimizer.step()
            valid = valid_label_count(masks, semantic)
            training_total += float(loss.detach()) * valid
            training_valid += valid
        if training_valid == 0:
            raise ValueError("Training split contains no valid labels")
        training_loss = training_total / training_valid
        validation_loss = mean_loss(model, validation_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "split_seed": seed,
                },
                checkpoint_path,
            )
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

