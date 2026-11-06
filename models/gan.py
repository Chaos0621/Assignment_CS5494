"""Label-conditioned GAN for original (non-binarised) MNIST.

The generator and discriminator both embed digit labels and concatenate them
with their inputs, allowing the generator to produce a requested digit.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        label_embed_dim=10,
        image_size=(28, 28),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        in_dim = latent_dim + label_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size[0] * image_size[1]),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_embedding(labels)
        x = torch.cat([z, emb], dim=1)
        return self.net(x).view(z.size(0), 1, *self.image_size)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str = "cpu",
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        else:
            labels = labels.to(device)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self(z, labels).view(num_samples, -1)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes=10,
        label_embed_dim=10,
        image_size=(28, 28),
    ):
        super().__init__()
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        in_dim = image_size[0] * image_size[1] + label_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_embedding(labels)
        x = torch.cat([x.view(x.size(0), -1), emb], dim=1)
        return self.net(x)
