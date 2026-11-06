# CS5494 Assignment 1 — Generative Models for MNIST

This repository implements three generative models for handwritten digits:

- `fvsbn` — Fully Visible Sigmoid Belief Network (binarised MNIST)
- `nade` — Neural Autoregressive Distribution Estimator (binarised MNIST)
- `gan` — label-conditioned GAN (original MNIST)

## Requirements

- Python 3.10+
- PyTorch, torchvision, matplotlib, numpy

## Training

From the repository root:

```bash
# FVSBN
python scripts/training.py --model fvsbn --epochs 20 --output-dir outputs

# NADE
python scripts/training.py --model nade --epochs 20 --hidden-size 500 --output-dir outputs

# Conditional GAN
python scripts/training.py --model gan --epochs 20 --g-lr 1e-4 --d-lr 1e-4 --output-dir outputs
```

Training history is written to `outputs/<model>_history.json` and checkpoints to
`outputs/<model>_*.pt`.

## Plots and samples

```bash
# Loss curves
python scripts/draw_results.py --output-dir outputs

# Final sample grids (needs trained checkpoints in scripts/outputs)
python scripts/make_samples.py

# Generate a specific digit with the conditional GAN
python scripts/generate_digit.py 7 --Name gan

# Generate a 2x5 grid of digits 0-9
python scripts/generate_digit.py --Name gan
```

## Training-progress grids

Pass `--progress-every 4` to save a 5-row grid image after every 4 epochs:

```bash
python scripts/training.py --model fvsbn --epochs 20 --output-dir outputs \
  --progress-every 4 --progress-path docs/figures/fvsbn_progress.png
```

## Notes

- `scripts/training.py` defaults to the `mps` device on Apple silicon; use
  `--device cpu` on other platforms.
- The first time `scripts/generate_digit.py` is used for an unconditional model
  (FVSBN/NADE) it trains a small digit classifier to filter samples; the cGAN
  does not need this because it is conditioned on labels.
