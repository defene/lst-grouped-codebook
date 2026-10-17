# EFDiff: Earth Foundation Model-guided Diffusion for LST Super-Resolution

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-2F6DB5?style=for-the-badge&logo=readthedocs&logoColor=white)](#citation) [![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

</div>

> EFDiff equips a diffusion model with an Earth Foundation Model (EFM) conditioning layer that grounds fine-scale thermal reconstruction in globally pretrained geospatial semantics.
>
> Starting from a coarse 7×7 Landsat thermal patch and its co-registered 224×224 HLS reflectance, the system encodes reflectance into Prithvi-EO-2.0 geospatial embeddings, injects them into a UNet denoiser via cross-attention, and iteratively reconstructs the 224×224 Land Surface Temperature (LST) residual.
>
> Two diffusion formulations — **EFDiff-ε** and **EFDiff-x₀** — offer complementary trade-offs between perceptual realism and pixel-level fidelity under an extreme 32× scale gap.

## From Coarse Thermal Observations to Fine-Scale LST

Extreme LST super-resolution is difficult for both deterministic regressors and vanilla diffusion baselines.

**Deterministic regression**
- Minimizes pixel-wise error and collapses toward a blurred conditional mean, losing sharp local contrast and attenuating thermal extremes.

**Region-specific pipelines**
- Fit spectral–thermal relationships to a single geographic domain and transfer poorly across climates, biomes, and land-cover regimes.

**Raw HLS channel concatenation**
- Injects spectral information at the pixel level but lacks the semantic understanding needed to disambiguate sub-pixel mixtures of vegetation, built surfaces, water, and terrain.

EFDiff addresses these issues by using a frozen EFM to convert multispectral reflectance into spatially structured embeddings, which the diffusion denoiser consumes via cross-attention as a stable, semantically rich conditioning signal.

## EFDiff Framework

![EFDiff framework](assets/framework.png)

### Core Components

- **EFM-guided conditioning**. Six-band HLS reflectance is tokenized into 14×14 = 196 patches and encoded by a frozen Prithvi-EO-2.0-300M ViT into geospatial embeddings `Z ∈ ℝ^{196×1024}`.
- **Residual-space prediction**. The task is reformulated as predicting `R = Y − X̃`, where `X̃` is the bicubic-upsampled low-resolution thermal field, with 3σ z-score standardization into `[−1, 1]`.
- **Cross-attention injection**. `Z` is queried at the UNet bottleneck and decoder blocks; the encoder processes only `[X̃, R_t]` so it first learns resolution-invariant thermal features before fusing HR spectral context.
- **EFDiff-ε (noise prediction)**. DDPM with 1000-step variance-preserving schedule and L1 loss; DDIM sampling at inference for faster reverse. Stronger perceptual realism.
- **EFDiff-x₀ (clean-residual prediction)**. ResShift-style residual-shifting transition with 15 steps and L2 loss. Stronger pixel-level fidelity at a fraction of the sampling cost.

## Why EFM Cross-Attention Matters

- **Semantic grounding for diffusion**: pretrained Earth representations capture vegetation boundaries, urban fabric, water bodies, and terrain texture — exactly the structures that govern sub-pixel thermal mixing.
- **Stable conditioning surface**: frozen embeddings avoid catastrophic forgetting and decouple the encoder's parameter count from training cost.
- **Complementary variants**: one shared conditioning design yields two samplers with different perception–distortion trade-offs, selectable per application.
- **Scene-complexity scaling**: the gain over HLS channel concatenation grows with spatial heterogeneity, confirming that semantic priors are most useful where raw spectral inputs fail.

## Experimental Highlights

- Evaluated on a globally diverse benchmark of **242,416 co-registered Landsat thermal–reflectance patches** from **3,094 MGRS tiles** (2,943 training / 151 test) under a unified **32× super-resolution protocol**.
- **EFDiff-x₀** achieves the lowest RMSE (0.29–0.56 °C) and the highest SSIM in every LULC group (Forest / Low Vegetation / Water), surpassing all deterministic baselines including RCAN.
- **EFDiff-ε** achieves the best perceptual scores by a wide margin (FID below 10 on land categories — roughly 4× improvement over the best non-EFM baseline).
- Per-pixel RMSE maps show EFDiff-x₀ largely suppresses the 7×7 checkerboard artifacts that persist in SR3, EDSR, and SRGAN at extreme 32× upsampling.
- Ablation on scene complexity confirms that EFM cross-attention conditioning consistently outperforms HLS channel concatenation, with the advantage growing in spatially heterogeneous scenes.

![Main results comparison](assets/main_results.png)

![Qualitative comparison](assets/results.png)

## Repository Structure

```text
EFDiff/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- .env.example
|-- train.py
|-- infer.py
|-- evaluate.py
|-- lst_dataset.py
|-- configs/
|   |-- config_sr3_prithvi.yaml       # EFDiff-epsilon (noise prediction, 1000 steps)
|   |-- config_resshift_prithvi.yaml  # EFDiff-x0    (clean-residual, 15 steps)
|   |-- config_sr3.yaml               # SR3 baseline (no EFM)
|   `-- config_resshift.yaml          # ResShift baseline (no EFM)
|-- model/
|   |-- unet.py               # UNet with optional cross-attention
|   |-- prithvi_encoder.py    # Frozen Prithvi-EO-2.0 encoder
|   |-- diffusion.py          # GaussianDiffusion (SR3) + ResShiftDiffusion
|   |-- ddpm.py               # Training/inference orchestration, EMA, DDP
|   |-- factory.py            # define_G, weight init
|   `-- base.py               # BaseModel helpers
|-- util/
|   |-- config.py             # Config validation
|   |-- downsampling.py       # Wald's Protocol (PSF + area-average)
|   |-- normalization.py      # Z-score with 3-sigma clipping
|   |-- metrics.py            # PSNR, SSIM, RMSE
|   |-- logger.py             # Config parsing, W&B integration
|   |-- setup.py              # DDP + seed initialization
|   `-- temperature.py        # Tensor <-> Celsius conversions
|-- data_download/            # Google Earth Engine pipeline (Phases 1-4)
`-- scripts/
    `-- compute_residual_stats.py     # One-time residual normalization statistics
```

### Key Entry Points

- `train.py`: DDP-compatible training loop with AMP bfloat16 and EMA 0.9999
- `infer.py`: Inference that saves `.npy` predictions per test patch
- `evaluate.py`: Offline evaluation (RMSE, SSIM, LPIPS, FID) — no GPU required
- `lst_dataset.py`: Dataset with Wald's Protocol synthesis of LR input and optional HLS loading
- `data_download/run_pipeline.py`: End-to-end GEE data pipeline (tile selection → download → finalize)

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+ with CUDA, and
[`terratorch`](https://github.com/IBM/terratorch) for Prithvi-EO-2.0 loading.

### 3. Configure paths and credentials

```bash
cp .env.example .env
```

Then update `.env` with:

- `SCRATCH`: base path for data and experiment outputs
- `WANDB_API_KEY`: optional, for Weights & Biases experiment tracking

### 4. Review runtime settings

Configs use `${SCRATCH}` template syntax, expanded at load time by
`util/logger.py`. Before training, review the active config under `configs/`
and adjust:

- `datasets.train.dataroot`, `datasets.test.dataroot`, `datasets.normalization_stats`
- `train.n_iter`, `train.optimizer.lr`, `datasets.train.batch_size`
- `model.prithvi.model_name` (defaults to `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`)

## Running The Code

### One-time setup

```bash
# Compute residual normalization statistics from training patches
python scripts/compute_residual_stats.py
```

This writes `normalization_stats.json` to the path configured under
`datasets.normalization_stats` in the YAML. Training will fail fast if the
file is missing.

### Training

```bash
# Single GPU — EFDiff-x0 (15-step, faster)
python train.py -c configs/config_resshift_prithvi.yaml

# Single GPU — EFDiff-epsilon (1000-step, richer texture)
python train.py -c configs/config_sr3_prithvi.yaml

# Multi-GPU via torchrun
torchrun --nproc_per_node=4 train.py -c configs/config_resshift_prithvi.yaml -enable_wandb
```

### Inference

```bash
python infer.py -c configs/config_resshift_prithvi.yaml \
    --checkpoint /path/to/I200000_E400 -o results/efdiff_x0
```

DDIM sampling for EFDiff-ε is controlled via `--ddim_steps`; the native
1000-step schedule is used when omitted.

### Evaluation

```bash
python evaluate.py --pred_dir results/efdiff_x0
```

Reports RMSE (°C), SSIM, LPIPS, and FID per land-cover group.

## Data Prerequisites

This repository contains the code, but not the underlying geospatial assets or
trained model weights.

The pipeline expects, per patch under `$SCRATCH/lst-super-resolution/prithvi_data/{train,test}/`:

- `lst.npy`: HR LST `(256, 256)` int16, scale 100 (value / 100 = °C)
- `hls_bands.npy`: HLS reflectance `(6, 256, 256)` int16, scale 10000
- `fmask.npy`: Cloud mask `(256, 256)` uint8

The low-resolution thermal input is synthesized on-the-fly per patch via
Wald's Protocol (Gaussian PSF + area-average + bicubic upsampling), so no
bicubic cache file is required.

Data sources follow the paper's Appendix A:

- Reflectance: NASA HLSL30 v2.0 product (6 bands, 30 m)
- Thermal: Landsat Collection 2 Level-2 `ST_B10` (matched within ±1 day)
- Land cover: Copernicus 100 m (for LULC-stratified tile sampling)
- Ecoregions: RESOLVE 2017

The full Google Earth Engine pipeline is in `data_download/` — entry point
`data_download/run_pipeline.py`.

## Outputs

A typical training run writes:

- `experiments/<name>_<timestamp>/checkpoint/I{iter}_E{epoch}_gen.pth`: model weights
- `experiments/<name>_<timestamp>/checkpoint/I{iter}_E{epoch}_opt.pth`: optimizer state
- `experiments/<name>_<timestamp>/checkpoint/I{iter}_E{epoch}_ema.pth`: EMA weights (used for evaluation)
- `experiments/<name>_<timestamp>/logs/train.log`: training log
- `experiments/<name>_<timestamp>/tb_logger/`: TensorBoard event files

Inference and evaluation write `.npy` prediction maps and a CSV of per-patch
metrics under the specified `-o` directory.

## Citation

```bibtex
@article{chen2026earth,
  title={When Earth Foundation Models Meet Diffusion: An Application to Land Surface Temperature Super-Resolution},
  author={Chen, Yiheng and Ma, Zihui and Jiang, Peishi and Dai, Yilong and Hu, Qikai and Ye, Xinyue and Li, Lingyao and Sousa, Rita and Yu, Runlong},
  journal={arXiv preprint arXiv:2604.16841},
  year={2026}
}
```
