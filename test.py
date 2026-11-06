"""Smoke tests for the model interfaces used by the assignment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from models.fvsbn import Fvsbn
from models.gan import Generator
from models.nade import nade

INPUT_SIZE = 28 * 28


def _binary_tensor(batch: int, size: int) -> torch.Tensor:
    return torch.randint(0, 2, (batch, size)).float()


def test_fvsbn() -> None:
    model = Fvsbn(INPUT_SIZE)
    x = _binary_tensor(4, INPUT_SIZE)
    assert model(x).shape == (4, INPUT_SIZE)
    samples = model.sample(4)
    assert samples.shape == (4, INPUT_SIZE)
    assert torch.all((samples == 0) | (samples == 1))


def test_nade() -> None:
    model = nade(INPUT_SIZE, hidden_size=64)
    x = _binary_tensor(4, INPUT_SIZE)
    assert model(x).shape == (4, INPUT_SIZE)
    samples = model.sample(4)
    assert samples.shape == (4, INPUT_SIZE)
    assert torch.all((samples == 0) | (samples == 1))


def test_conditional_gan() -> None:
    model = Generator(latent_dim=16, image_size=(28, 28))
    labels = torch.tensor([0, 1, 2, 3, 4])
    samples = model.sample(5, labels=labels)
    assert samples.shape == (5, INPUT_SIZE)


if __name__ == "__main__":
    test_fvsbn()
    test_nade()
    test_conditional_gan()
    print("All smoke tests passed.", flush=True)
