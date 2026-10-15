from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from finder.config import load_config, select_device
from finder.data import load_image
from finder.runtime import build_model, load_checkpoint, predict_years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run annual FINDER inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="One processed TIFF; defaults to sample_dir/*.tif")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config, _ = load_config(arguments.config)
    device = select_device(arguments.device)
    model = build_model(config, device)
    load_checkpoint(model, config["paths"]["checkpoint"], device)
    model.eval()

    inputs = [Path(arguments.input)] if arguments.input else sorted(
        Path(config["paths"]["sample_dir"]).glob("*.tif")
    )
    if not inputs:
        raise FileNotFoundError("No processed TIFF inputs were found")
    output_dir = Path(config["paths"]["annual_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        image, profile = load_image(path)
        with torch.inference_mode():
            logits = model(torch.from_numpy(image).unsqueeze(0).to(device)).reshape(1, 14 * 14, 10, 2)
        years = predict_years(logits.cpu(), float(config["inference"]["annual_entropy_threshold"]))
        result = years[0].reshape(14, 14).numpy().astype(np.int16)
        profile.update(driver="GTiff", height=14, width=14, count=1, dtype="int16", compress="lzw")
        output = output_dir / f"{path.stem}_year_index.tif"
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(result, 1)
        print(output)


if __name__ == "__main__":
    main()

