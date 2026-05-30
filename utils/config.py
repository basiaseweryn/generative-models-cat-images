import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility (plan section 4)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# Paths
# Local example:
#   DATA_DIR = "./data/cats"
#   DOGS_VS_CATS_DIR = "./data/dogs_vs_cats"
#
# Kaggle example (uncomment when running on Kaggle):
DATA_DIR = "/kaggle/input/cat-dataset"
DOGS_VS_CATS_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"


TRAINED_MODELS_DIR = "./trained_models/"
OUTPUT_DIR = "./outputs/"

#Global hyperparameters

IMAGE_SIZE = 64
LATENT_DIM = 128
RANDOM_SEED = 42
NUM_WORKERS = 2

# FID: use at least 1000–2000 samples for report-quality scores
FID_NUM_SAMPLES = 2000

# Scenario groups (for batch evaluation in the notebook)
STAGE_1_SCENARIOS = ["stage_1_vae", "stage_1_dcgan"]
STAGE_2_SCENARIOS = [
    "stage_2_vae_lr_1e-4",
    "stage_2_vae_latent_64",
    "stage_2_dcgan_lr_1e-4",
    "stage_2_dcgan_batch_64",
]
STAGE_4_SCENARIOS = [
    "stage_1_dcgan",
    "stage_4_dcgan_n_critic_5",
    "stage_4_dcgan_instance_noise",
]

# ImageNet normalization (used only for FID feature extraction)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Experiment definitions
EXPERIMENTS = {
    # Stage 1 — baseline VAE and DCGAN on preprocessed cats
    "stage_1_vae": {
        "stage": 1,
        "model": "vae",
        "dataset": "cats",
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
    },
    "stage_1_dcgan": {
        "stage": 1,
        "model": "dcgan",
        "dataset": "cats",
        "epochs": 50,
        "lr": 2e-4,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
        "n_critic": 1,  # discriminator updates per generator update
    },
    # Stage 2 — hyperparameter search (use smaller subset_ratio to save GPU time)
    "stage_2_vae_lr_1e-4": {
        "stage": 2,
        "model": "vae",
        "dataset": "cats",
        "epochs": 30,
        "lr": 1e-4,
        "batch_size": 64,
        "latent_dim": 128,
        "subset_ratio": 0.25,
    },
    "stage_2_vae_latent_64": {
        "stage": 2,
        "model": "vae",
        "dataset": "cats",
        "epochs": 30,
        "lr": 1e-3,
        "batch_size": 64,
        "latent_dim": 64,
        "subset_ratio": 0.25,
    },
    "stage_2_dcgan_lr_1e-4": {
        "stage": 2,
        "model": "dcgan",
        "dataset": "cats",
        "epochs": 30,
        "lr": 1e-4,
        "batch_size": 64,
        "latent_dim": 128,
        "subset_ratio": 0.25,
        "n_critic": 1,
    },
    "stage_2_dcgan_batch_64": {
        "stage": 2,
        "model": "dcgan",
        "dataset": "cats",
        "epochs": 30,
        "lr": 2e-4,
        "batch_size": 64,
        "latent_dim": 128,
        "subset_ratio": 0.25,
        "n_critic": 1,
    },
    # Stage 4 — mode-collapse mitigation (DCGAN-focused)
    "stage_4_dcgan_n_critic_5": {
        "stage": 4,
        "model": "dcgan",
        "dataset": "cats",
        "epochs": 50,
        "lr": 2e-4,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
        "n_critic": 5,
        "instance_noise_std": 0.0,
    },
    "stage_4_dcgan_instance_noise": {
        "stage": 4,
        "model": "dcgan",
        "dataset": "cats",
        "epochs": 50,
        "lr": 2e-4,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
        "n_critic": 1,
        "instance_noise_std": 0.1,
    },
    # Stage 5 — best model on cats + dogs (pick model name after stage 1–2)
    "stage_5_dcgan_cats_dogs": {
        "stage": 5,
        "model": "dcgan",
        "dataset": "cats_and_dogs",
        "epochs": 50,
        "lr": 2e-4,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
        "n_critic": 1,
    },
    "stage_5_vae_cats_dogs": {
        "stage": 5,
        "model": "vae",
        "dataset": "cats_and_dogs",
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 128,
        "latent_dim": 128,
        "subset_ratio": 1.0,
    },
}
