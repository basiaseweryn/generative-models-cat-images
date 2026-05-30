import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import transforms as T
from torchvision.models import inception_v3
from torchvision.utils import make_grid, save_image

from utils.config import (
    EXPERIMENTS,
    FID_NUM_SAMPLES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUTPUT_DIR,
    TRAINED_MODELS_DIR,
)
from utils.data_utils import get_dataloader
from utils.models import get_model, sample_dcgan, sample_vae, sample_with_latents

# Below this diversity score, generations may indicate mode collapse (tune after first run)
MODE_COLLAPSE_DIVERSITY_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Saving / visualization
# ---------------------------------------------------------------------------
def denormalize_dcgan(images: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] -> [0, 1] for plotting."""
    return (images * 0.5 + 0.5).clamp(0, 1)


def save_sample_grid(
    images: torch.Tensor,
    filepath: str,
    nrow: int = 8,
    normalize: bool = False,
):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    save_image(images, filepath, nrow=nrow, normalize=normalize, padding=2)
    print(f"Saved grid -> {filepath}")


def show_image_grid(images: torch.Tensor, title: str = "", nrow: int = 8):
    grid = make_grid(images.cpu(), nrow=nrow, padding=2)
    if grid.size(0) == 1:
        grid = grid.squeeze(0)
    else:
        grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.numpy(), cmap="gray" if grid.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------
class InceptionFeatureExtractor(nn.Module):
    """Inception v3 up to the last pooling layer (2048-d features)."""

    def __init__(self):
        super().__init__()
        model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        model.fc = nn.Identity()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Inception expects 299x299 and ImageNet normalization
        return self.model(x)


def _images_to_inception_input(images: torch.Tensor) -> torch.Tensor:
    """Convert model outputs ([0,1] or [-1,1]) to Inception input."""
    if images.min() < 0:
        images = denormalize_dcgan(images)
    resize = T.Resize((299, 299), antialias=True)
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return norm(resize(images))


@torch.no_grad()
def get_inception_features(images: torch.Tensor, device: torch.device, batch_size: int = 32):
    extractor = InceptionFeatureExtractor().to(device)
    feats = []
    for i in range(0, images.size(0), batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = _images_to_inception_input(batch)
        feats.append(extractor(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


@torch.no_grad()
def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Lower FID = generated distribution closer to real (plan section 6).
    Pass tensors in [0, 1] for VAE or [-1, 1] for DCGAN — both are handled.
    """
    real_feat = get_inception_features(real_images, device)
    fake_feat = get_inception_features(fake_images, device)

    mu1, sigma1 = real_feat.mean(axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = fake_feat.mean(axis=0), np.cov(fake_feat, rowvar=False)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


@torch.no_grad()
def fid_from_dataloader(
    dataloader,
    generate_fn,
    device: torch.device,
    max_real: int = 2000,
    max_fake: int = 2000,
) -> float:
    """Collect real/fake batches and compute FID."""
    reals, fakes = [], []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        reals.append(batch)
        fakes.append(generate_fn(batch.size(0)))
        if sum(b.size(0) for b in reals) >= max_real:
            break

    real_t = torch.cat(reals, dim=0)[:max_real]
    fake_t = torch.cat(fakes, dim=0)[:max_fake]
    return compute_fid(real_t, fake_t, device)


# ---------------------------------------------------------------------------
# Latent space interpolation (plan stage 3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def interpolate_vae(
    model,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    num_steps: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """Linear interpolation between two latent vectors -> num_steps + 2 images."""
    device = device or z_a.device
    alphas = torch.linspace(0, 1, num_steps + 2, device=device)
    images = []
    for alpha in alphas:
        z = (1 - alpha) * z_a + alpha * z_b
        images.append(model.decode(z.unsqueeze(0)))
    return torch.cat(images, dim=0)


@torch.no_grad()
def interpolate_dcgan(
    generator,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    num_steps: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    device = device or z_a.device
    alphas = torch.linspace(0, 1, num_steps + 2, device=device)
    images = []
    for alpha in alphas:
        z = (1 - alpha) * z_a + alpha * z_b
        images.append(generator(z.unsqueeze(0)))
    return torch.cat(images, dim=0)


def save_interpolation_grid(
    images: torch.Tensor,
    filepath: str,
    is_dcgan: bool = False,
):
    if is_dcgan:
        images = denormalize_dcgan(images)
    save_sample_grid(images, filepath, nrow=images.size(0))


# ---------------------------------------------------------------------------
# Mode collapse helpers (plan stage 4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def feature_diversity_score(features: np.ndarray) -> float:
    """
    Simple diversity proxy: mean pairwise L2 distance in feature space.
    Very low values may indicate mode collapse.
    """
    n = min(500, features.shape[0])
    idx = np.random.choice(features.shape[0], n, replace=False)
    f = features[idx]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(f[i] - f[j]))
    return float(np.mean(dists)) if dists else 0.0


@torch.no_grad()
def evaluate_mode_collapse(
    fake_images: torch.Tensor,
    device: torch.device,
) -> dict:
    feats = get_inception_features(fake_images, device)
    return {
        "feature_diversity": feature_diversity_score(feats),
        "num_samples": fake_images.size(0),
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    epoch: int,
    history: dict,
    config: dict,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("history", {}), ckpt.get("config", {})


def output_path(scenario_name: str, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, scenario_name, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Model loading & scenario-level evaluation
# ---------------------------------------------------------------------------
def load_trained_model(scenario_name: str, device: torch.device):
    """Load checkpoint for a scenario from trained_models/."""
    config = EXPERIMENTS[scenario_name].copy()
    model = get_model(config["model"], config).to(device)
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, scenario_name, "model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train '{scenario_name}' first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt.get("history", {})


def _to_display_images(images: torch.Tensor, config: dict) -> torch.Tensor:
    if config["model"] == "dcgan":
        return denormalize_dcgan(images)
    return images


def make_generate_fn(model, config: dict, device: torch.device):
    """Callable for fid_from_dataloader: generate n fake images in [0, 1]."""

    def generate(n: int) -> torch.Tensor:
        with torch.no_grad():
            if config["model"] == "vae":
                return sample_vae(model, n, device)
            return denormalize_dcgan(
                sample_dcgan(model.generator, n, config["latent_dim"], device)
            )

    return generate


@torch.no_grad()
def evaluate_fid_scenario(
    scenario_name: str,
    device: torch.device,
    max_samples: int = None,
) -> float:
    """Report-quality FID on held-out training distribution."""
    max_samples = max_samples or FID_NUM_SAMPLES
    model, config, _ = load_trained_model(scenario_name, device)
    loader = get_dataloader(config)
    generate_fn = make_generate_fn(model, config, device)
    fid = fid_from_dataloader(
        loader, generate_fn, device, max_real=max_samples, max_fake=max_samples
    )
    print(f"FID [{scenario_name}] ({max_samples} samples): {fid:.2f}")
    return fid


def compare_fid_scenarios(
    scenario_names: list,
    device: torch.device,
    max_samples: int = None,
    save_csv: str = None,
) -> list:
    """Build comparison table for VAE vs DCGAN / hyperparameter runs."""
    max_samples = max_samples or FID_NUM_SAMPLES
    rows = []
    for name in scenario_names:
        cfg = EXPERIMENTS[name]
        fid = evaluate_fid_scenario(name, device, max_samples=max_samples)
        rows.append(
            {
                "scenario": name,
                "model": cfg["model"],
                "stage": cfg.get("stage"),
                "lr": cfg.get("lr"),
                "latent_dim": cfg.get("latent_dim"),
                "batch_size": cfg.get("batch_size"),
                "subset_ratio": cfg.get("subset_ratio"),
                "fid": round(fid, 2),
            }
        )

    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved FID table -> {save_csv}")

    return rows


def print_fid_table(rows: list):
    header = f"{'scenario':<32} {'model':<6} {'lr':<10} {'z':<5} {'bs':<5} {'FID':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scenario']:<32} {r['model']:<6} {r['lr']:<10} "
            f"{r['latent_dim']:<5} {r['batch_size']:<5} {r['fid']:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Interpolation from selected generated images (project guideline)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_candidate_grid(
    model,
    config: dict,
    device: torch.device,
    num_candidates: int = 64,
    seed: int = 42,
):
    """
    Generate a grid of samples; pick two indices and use their latent vectors.
    Returns display-ready images [0,1], raw latents [N, latent_dim], raw images for saving.
    """
    raw_images, latents = sample_with_latents(
        model, config, num_candidates, device, seed=seed
    )
    display = _to_display_images(raw_images, config)
    return display, latents, raw_images


def save_latent_codes(z_a: torch.Tensor, z_b: torch.Tensor, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save({"z_a": z_a.cpu(), "z_b": z_b.cpu()}, filepath)
    print(f"Saved latent codes -> {filepath}")


@torch.no_grad()
def interpolate_between_latents(
    model,
    config: dict,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    device: torch.device,
    num_steps: int = 8,
):
    """
    Linear interpolation between two saved latent vectors.
    Returns 10 images (2 endpoints + 8 intermediate) and display tensor.
    """
    z_a = z_a.squeeze().to(device)
    z_b = z_b.squeeze().to(device)

    if config["model"] == "vae":
        images = interpolate_vae(model, z_a, z_b, num_steps=num_steps, device=device)
        is_dcgan = False
    else:
        images = interpolate_dcgan(
            model.generator, z_a, z_b, num_steps=num_steps, device=device
        )
        is_dcgan = True

    return _to_display_images(images, config), images, is_dcgan


# ---------------------------------------------------------------------------
# Mode collapse: detect & compare mitigations
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mode_collapse_scenario(
    scenario_name: str,
    device: torch.device,
    num_samples: int = 512,
) -> dict:
    model, config, _ = load_trained_model(scenario_name, device)
    with torch.no_grad():
        if config["model"] == "vae":
            fakes = sample_vae(model, num_samples, device)
        else:
            fakes = denormalize_dcgan(
                sample_dcgan(model.generator, num_samples, config["latent_dim"], device)
            )
    metrics = evaluate_mode_collapse(fakes, device)
    metrics["scenario"] = scenario_name
    metrics["likely_mode_collapse"] = (
        metrics["feature_diversity"] < MODE_COLLAPSE_DIVERSITY_THRESHOLD
    )
    return metrics


def compare_mode_collapse_scenarios(scenario_names: list, device: torch.device) -> list:
    results = []
    for name in scenario_names:
        m = evaluate_mode_collapse_scenario(name, device)
        results.append(m)
        flag = "POSSIBLE COLLAPSE" if m["likely_mode_collapse"] else "ok"
        print(
            f"{name}: diversity={m['feature_diversity']:.2f} "
            f"(threshold={MODE_COLLAPSE_DIVERSITY_THRESHOLD}) -> {flag}"
        )
    return results


# ---------------------------------------------------------------------------
# Stage 5: cats-only vs cats+dogs comparison
# ---------------------------------------------------------------------------
@torch.no_grad()
def compare_cats_vs_cats_dogs(
    scenario_cats_only: str,
    scenario_mixed: str,
    device: torch.device,
    num_samples: int = 64,
    save_dir: str = None,
):
    """
    Qualitative + FID comparison between cats-only and mixed training.
    For the report: do samples look like distinct cats/dogs or blends?
    """
    save_dir = save_dir or os.path.join(OUTPUT_DIR, "stage_5_comparison")
    os.makedirs(save_dir, exist_ok=True)

    model_cat, cfg_cat, _ = load_trained_model(scenario_cats_only, device)
    model_mix, cfg_mix, _ = load_trained_model(scenario_mixed, device)

    imgs_cat, _, _ = generate_candidate_grid(
        model_cat, cfg_cat, device, num_candidates=num_samples, seed=42
    )
    imgs_mix, _, _ = generate_candidate_grid(
        model_mix, cfg_mix, device, num_candidates=num_samples, seed=42
    )

    save_sample_grid(imgs_cat, os.path.join(save_dir, "cats_only_model.png"), nrow=8)
    save_sample_grid(imgs_mix, os.path.join(save_dir, "cats_dogs_model.png"), nrow=8)

    fid_cats_only = evaluate_fid_scenario(scenario_cats_only, device)
    fid_mixed = evaluate_fid_scenario(scenario_mixed, device)

    summary = {
        "scenario_cats_only": scenario_cats_only,
        "scenario_mixed": scenario_mixed,
        "fid_cats_only": fid_cats_only,
        "fid_mixed": fid_mixed,
        "save_dir": save_dir,
    }
    print("Comparison summary:", summary)
    return summary, imgs_cat, imgs_mix
