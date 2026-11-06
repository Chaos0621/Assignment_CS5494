"""Sampling + visualisation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def sample_grid(
    model,
    num_samples: int,
    device: torch.device | str,
    grid: tuple[int, int] = (5, 5),
    image_size: tuple[int, int] = (28, 28),
    title: str | None = None,
    save_path: str | Path | None = None,
):
    samples = model.sample(num_samples, device=device).cpu().view(-1, *image_size).numpy()
    rows, cols = grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < samples.shape[0]:
            ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_loss_curves(
    histories: dict[str, list[dict]],
    save_path: str | Path | None = None,
    metric: str = "test_bpd",
):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, epochs in histories.items():
        xs = [e["epoch"] for e in epochs]
        ax.plot(xs, [e[metric] for e in epochs], label=f"{name} (test)", linewidth=2)
        train_metric = metric.replace("test", "train")
        ax.plot(
            xs,
            [e[train_metric] for e in epochs],
            label=f"{name} (train)",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bits per dim")
    ax.set_title("Negative log-likelihood on binarised MNIST")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_gan_losses(
    histories: dict[str, list[dict]],
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, epochs in histories.items():
        xs = [e["epoch"] for e in epochs]
        ax.plot(xs, [e["d_loss"] for e in epochs], label=f"{name} D loss", linewidth=2)
        ax.plot(xs, [e["g_loss"] for e in epochs], label=f"{name} G loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN discriminator / generator losses")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def save_progress_grid(
    stages: list[torch.Tensor],
    save_path: str | Path,
    image_size: tuple[int, int] = (28, 28),
):
    rows = len(stages)
    cols = stages[0].shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for r, stage in enumerate(stages):
        arr = stage.detach().cpu().numpy()
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(arr[c].reshape(*image_size), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
