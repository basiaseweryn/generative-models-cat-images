# generative-models-cat-images

Deep Learning project 3 — compare **VAE** and **DCGAN** for cat image generation (WUT Data Science).

## Project layout

```
generative-models-cat-images/
├── experiments.ipynb      # run training & evaluation here
├── utils/
│   ├── config.py          # paths, seeds, EXPERIMENTS dict
│   ├── data_utils.py      # datasets & dataloaders
│   ├── models.py          # VAE + DCGAN architectures
│   └── evaluation_utils.py  # FID, plots, interpolation, checkpoints
├── data/                  # put datasets here (not in git)
├── trained_models/        # saved checkpoints (not in git)
└── outputs/               # figures & grids (not in git)
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download the [Cats Dataset](https://www.kaggle.com/datasets) from Kaggle and place images under `data/cats/` (any folder structure — all `.jpg`/`.png` files are found recursively).
2. For **stage 5**, also download [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) and set `DOGS_VS_CATS_DIR` in `utils/config.py`.
3. On Kaggle, update `DATA_DIR` and `DOGS_VS_CATS_DIR` to the input paths, e.g. `/kaggle/input/...`.

## Running experiments

Open `experiments.ipynb`, set `SCENARIO` to a key from `EXPERIMENTS` in `utils/config.py`, and run the cells.

| Stage | Scenario keys (examples) |
|-------|--------------------------|
| 1 | `stage_1_vae`, `stage_1_dcgan` |
| 2 | `stage_2_vae_lr_1e-4`, `stage_2_dcgan_batch_64`, … |
| 3 | interpolation cell after training |
| 4 | `stage_4_dcgan_n_critic_5`, `stage_4_dcgan_instance_noise` |
| 5 | `stage_5_dcgan_cats_dogs`, `stage_5_vae_cats_dogs` |

## Metrics & rubric helpers

- **FID (2000 samples)** — `compare_fid_scenarios()` saves CSV under `outputs/reports/`
- **Qualitative** — `generate_candidate_grid()` + saved PNGs
- **Interpolation** — pick `IDX_A` / `IDX_B`, save `interpolation_latents.pth`, 10-image grid
- **Mode collapse** — `compare_mode_collapse_scenarios()` flags low diversity
- **Cats vs cats+dogs** — `compare_cats_vs_cats_dogs()` side-by-side FID + grids
