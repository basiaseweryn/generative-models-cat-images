# generative-models-cat-images

Deep Learning project 3 — compare **VAE** and **DCGAN** for cat image generation (WUT Data Science).

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
