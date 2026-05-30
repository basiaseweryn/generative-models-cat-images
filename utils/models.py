import torch
import torch.nn as nn


# VAE (64x64 RGB)
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 64 -> 32 -> 16 -> 8 -> 4  => 256 * 4 * 4
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """Reconstruction (BCE) + KL divergence."""
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kl) / x.size(0), bce / x.size(0), kl / x.size(0)


# DCGAN (64x64 RGB) — generator output in [-1, 1]
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


class DCGAN(nn.Module):
    """Wrapper holding generator + discriminator."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = DCGANGenerator(latent_dim)
        self.discriminator = DCGANDiscriminator()


def get_model(model_name: str, config: dict) -> nn.Module:
    latent_dim = config.get("latent_dim", 128)
    if model_name == "vae":
        return VAE(latent_dim=latent_dim)
    if model_name == "dcgan":
        return DCGAN(latent_dim=latent_dim)
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def sample_vae(model: VAE, num_images: int, device: torch.device):
    z = torch.randn(num_images, model.latent_dim, device=device)
    return model.decode(z)


@torch.no_grad()
def sample_dcgan(generator: DCGANGenerator, num_images: int, latent_dim: int, device: torch.device):
    z = torch.randn(num_images, latent_dim, device=device)
    return generator(z)


@torch.no_grad()
def sample_with_latents(model, config: dict, num_images: int, device: torch.device, seed: int = None):
    """
    Generate images and return their latent noise vectors (shape: [N, latent_dim]).
    Use two rows of `latents` for interpolation (project guideline).
    """
    if seed is not None:
        torch.manual_seed(seed)

    latent_dim = config["latent_dim"]
    z = torch.randn(num_images, latent_dim, device=device)

    if config["model"] == "vae":
        images = model.decode(z)
    elif config["model"] == "dcgan":
        images = model.generator(z)
    else:
        raise ValueError(config["model"])

    return images, z
