# FINDER

FINDER is a Foundation model-based INterpretation framework for mangrove
DEgradation from Remotely sensed time series. This repository contains the
training and inference implementation used by the paper: P-WIC provides
year-level localization and IG-TE provides fine-grained temporal-spectral
attribution.

## Repository contents

```text
FINDER/
├── config.yaml          # paths, split, model, training, and IG-TE settings
├── train.py             # P-WIC training
├── infer.py             # annual P-WIC inference
├── igte.py              # pixel-level IG-TE inference
├── finder/              # dataset, model, loss, and attribution implementation
└── requirements.txt
```

The repository intentionally does not distribute satellite imagery, processed
TIFF files, annotations, trained checkpoints, or Prithvi weights.

## Data preparation

The study uses NASA HLS Landsat Operational Land Imager Surface Reflectance and
TOA Brightness Daily Global 30 m v2.0 (`NASA/HLS/HLSL30/v002`) observations
from 2014 through 2023. The preprocessing procedure is:

1. Mask clouds and cloud shadows with the HLS `Fmask` quality band.
2. Create 8-day median composites.
3. Treat missing and invalid observations as missing values and apply linear
   interpolation, including edge filling.
4. Apply a Savitzky-Golay filter.
5. Retain six bands and 460 observations for each 14 x 14 pixel patch.

Each processed sample TIFF must contain 2,760 layers (6 bands x 460 time
steps). Each training mask TIFF must have 10 layers (one annual semantic layer
for each year). Files are paired by their stem:

```text
data/
├── tif_processed/
│   ├── 0001.tif
│   └── ...
├── masks/
│   ├── 0001.tif
│   └── ...
└── dates.csv
weights/
└── Prithvi_EO_V1_100M.pt
```

`dates.csv` must contain a `date` column with 460 entries formatted as
`YYYY/MM/DD`. Update the paths in `config.yaml` if a different layout is used.

## Environment

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the official Prithvi-EO-1.0 100M checkpoint and place it at the path
configured by `paths.prithvi_weights`.

## Training

```bash
python train.py --config config.yaml --device cuda
```

## Annual inference

Run inference for every TIFF in `paths.sample_dir`:

```bash
python infer.py --config config.yaml --device cuda
```

Or process one TIFF:

```bash
python infer.py --config config.yaml --input data/tif_processed/0001.tif --device cuda
```

## IG-TE inference

Run IG-TE for one pixel in a processed patch:

```bash
python igte.py --config config.yaml \
  --input data/tif_processed/0001.tif --row 7 --col 7 --device cuda
```

The resulting NPZ file contains the 460 x 6 attribution array, input time
series, annual logits, predicted annual index, fine-grained change
index, and change date.
