"""Save final generated-sample grids for the three trained models."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.fvsbn import Fvsbn
from models.gan import Generator
from models.nade import nade

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 500
LATENT_DIM = 100
CKPT_DIR = REPO / "scripts" / "outputs"
FIG_DIR = REPO / "docs" / "figures"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_fvsbn():
    model = Fvsbn(INPUT_SIZE).to(_device())
    ckpt = torch.load(CKPT_DIR / "fvsbn_best.pt", map_location=_device())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_nade():
    model = nade(INPUT_SIZE, HIDDEN_SIZE).to(_device())
    ckpt = torch.load(CKPT_DIR / "nade_best.pt", map_location=_device())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_gan():
    model = Generator(latent_dim=LATENT_DIM).to(_device())
    ckpt = torch.load(CKPT_DIR / "gan_gan_epoch20.pt", map_location=_device())
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model


def _save_grid(samples, path, rows, cols, title=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for ax, sample in zip(axes.flat, samples):
        ax.imshow(sample.view(28, 28).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = _device()

    model = _load_fvsbn()
    samples = model.sample(20, device=dev).cpu()
    _save_grid(samples, FIG_DIR / "fvsbn_samples.png", 4, 5, "FVSBN samples")

    model = _load_nade()
    samples = model.sample(20, device=dev).cpu()
    _save_grid(samples, FIG_DIR / "nade_samples.png", 4, 5, "NADE samples")

    model = _load_gan()
    labels = torch.arange(10, device=dev)
    samples = model.sample(10, device=dev, labels=labels).cpu()
    _save_grid(samples, FIG_DIR / "gan_samples.png", 2, 5, "cGAN conditional samples (0-9)")

    print("Saved sample grids to", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
