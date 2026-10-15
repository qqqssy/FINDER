from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from finder.config import load_config, select_device
from finder.data import load_image
from finder.runtime import (
    build_model,
    integrated_gradients,
    load_checkpoint,
    predict_years,
    threshold_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IG-TE for one pixel")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True, help="Processed 460x6 TIFF")
    parser.add_argument("--row", type=int, required=True, choices=range(14))
    parser.add_argument("--col", type=int, required=True, choices=range(14))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_dates(path: str | Path) -> list[datetime]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        dates = [datetime.strptime(row["date"], "%Y/%m/%d") for row in csv.DictReader(stream)]
    if len(dates) != 460:
        raise ValueError(f"Expected 460 dates, found {len(dates)}")
    return dates


def main() -> None:
    arguments = parse_args()
    config, _ = load_config(arguments.config)
    device = select_device(arguments.device)
    model = build_model(config, device)
    load_checkpoint(model, config["paths"]["checkpoint"], device)
    image, _ = load_image(arguments.input)

    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(image).unsqueeze(0).to(device)).reshape(14 * 14, 10, 2)
    pixel = arguments.row * 14 + arguments.col
    predicted_year = int(
        predict_years(
            logits.unsqueeze(0).cpu(), float(config["inference"]["annual_entropy_threshold"])
        )[0, pixel]
    )
    if predicted_year < 0:
        print("No change year passed the entropy threshold for this pixel")
        return

    attribution = integrated_gradients(
        model,
        device,
        image,
        arguments.row,
        arguments.col,
        predicted_year,
        int(config["inference"]["ig_steps"]),
        int(config["inference"]["ig_baseline_window"]),
    )
    change_index = threshold_index(
        attribution.sum(axis=1), predicted_year, float(config["inference"]["ig_threshold"])
    )
    dates = load_dates(config["paths"]["date_csv"])
    output_dir = Path(config["paths"]["igte_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{Path(arguments.input).stem}_r{arguments.row}_c{arguments.col}.npz"
    np.savez(
        output,
        attribution=attribution,
        time_series=image[:, :, arguments.row, arguments.col],
        logits=logits[pixel].cpu().numpy(),
        predicted_year_index=predicted_year,
        change_time_index=change_index,
        change_date=dates[change_index].strftime("%Y-%m-%d"),
    )
    model.eval()
    print(f"change_date={dates[change_index]:%Y-%m-%d}")
    print(output)


if __name__ == "__main__":
    main()

