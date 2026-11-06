"""Generate an image of a requested MNIST digit using the trained FVSBN.

The FVSBN in this repo is an *unconditional* density model: it has no class
input, so it cannot directly sample a chosen digit. This script works around
that with importance sampling: it draws samples from the FVSBN and keeps the
sample a small classifier is most confident is the requested digit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import data_loaders
from models.fvsbn import Fvsbn
from models.nade import nade
from models.gan import Generator

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 500
Lantent_dim = 100
GAN_CKPT = REPO / "scripts" / "outputs" / "gan_gan_epoch20.pt"
FVSBN_CKPT = REPO / "scripts" / "outputs" / "fvsbn_best.pt"
NADE_CKPT = REPO / "scripts" / "outputs" / "nade_best.pt"
CLASSIFIER_CKPT = REPO / "scripts" / "outputs" / "digit_classifier.pt"
OUT_DIR = REPO / "scripts" / "outputs"

MAX_ATTEMPTS = 2000
BATCH_SIZE = 128
CONFIDENCE = 0.95


class DigitClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_SIZE,
        hidden: int = 512,
        num_classes: int = 10,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_fvsbn(device: torch.device) -> Fvsbn:
    model = Fvsbn(INPUT_SIZE).to(device)
    checkpoint = torch.load(FVSBN_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def _load_nade(device: torch.device) -> Fvsbn:
    model = nade(INPUT_SIZE, HIDDEN_SIZE).to(device)
    checkpoint = torch.load(NADE_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
def _load_gan(device: torch.device) -> Fvsbn:
    model = Generator(latent_dim=Lantent_dim).to(device)
    checkpoint = torch.load(GAN_CKPT, map_location=device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model

def _get_classifier(device: torch.device) -> DigitClassifier:
    model = DigitClassifier().to(device)
    if CLASSIFIER_CKPT.exists():
        model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device))
        model.eval()
        return model

    print("No cached classifier found; training one on binarised MNIST...", flush=True)
    train_loader, _ = data_loaders(
        REPO / "data", batch_size=256, num_workers=0, pin_memory=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    epochs = 3
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        print(
            f"  classifier epoch {epoch}/{epochs} "
            f"loss {total_loss / total:.4f} acc {correct / total:.4f}",
            flush=True,
        )

    model.eval()
    CLASSIFIER_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_CKPT)
    return model


def _generate_digit(
    model: Fvsbn,
    classifier: DigitClassifier,
    digit: int,
    device: torch.device,
):
    best_sample = None
    best_conf = 0.0
    attempts = 0

    while attempts < MAX_ATTEMPTS:
        n = min(BATCH_SIZE, MAX_ATTEMPTS - attempts)
        samples = model.sample(n, device=device)
        with torch.no_grad():
            probs = torch.softmax(classifier(samples), dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        mask = preds == digit
        if mask.any():
            idx = confs[mask].argmax().item()
            conf = confs[mask][idx].item()
            if conf > best_conf:
                best_conf = conf
                best_sample = samples[mask][idx].cpu()
            if best_conf >= CONFIDENCE:
                break

        attempts += n

    if best_sample is None:
        raise RuntimeError(
            f"could not find a sample classified as digit {digit} "
            f"within {MAX_ATTEMPTS} attempts"
        )
    return best_sample, best_conf


def _save_image(sample: torch.Tensor, digit: int) -> Path:
    out_path = OUT_DIR / f"digit_{digit}_.png"
    plt.figure(figsize=(2, 2))
    plt.imshow(sample.view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return out_path


def _save_grid(samples: torch.Tensor, rows: int = 2, cols: int = 5) -> Path:
    out_path = OUT_DIR / "digits_random_.png"
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for ax, sample in zip(axes.flat, samples):
        ax.imshow(sample.view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an MNIST digit image with the trained FVSBN"
    )
    parser.add_argument(
        "digit",
        type=int,
        choices=range(10),
        nargs="?",
        default=None,
        help="digit to generate (0-9)",
    )
    parser.add_argument("--Name", type=str)
    args = parser.parse_args()

    device = _get_device()
    name = (args.Name or "gan").lower()
    if name == "gan":
        model = _load_gan(device)
    elif name == "nade":
        model = _load_nade(device)
    elif name == "fvsbn":
        model = _load_fvsbn(device)
    else:
        raise SystemExit(f"unknown --Name {args.Name!r}")

    if args.digit is None:
        if name == "gan":
            labels = torch.arange(10, device=device)
            samples = model.sample(10, device=device, labels=labels).cpu()
        else:
            samples = model.sample(10, device=device).cpu()
        path = _save_grid(samples)
        print(f"Saved 10 random digits to {path}", flush=True)
        return

    if name == "gan":
        labels = torch.full((1,), args.digit, dtype=torch.long, device=device)
        sample = model.sample(1, device=device, labels=labels).cpu()[0]
        path = _save_image(sample, args.digit)
        print(f"Saved digit {args.digit} image to {path} (conditional GAN)", flush=True)
        return

    classifier = _get_classifier(device)

    sample, confidence = _generate_digit(model, classifier, args.digit, device)
    path = _save_image(sample, args.digit)
    print(
        f"Saved digit {args.digit} image to {path} "
        f"(classifier confidence {confidence:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
