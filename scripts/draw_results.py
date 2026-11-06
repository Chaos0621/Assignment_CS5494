"""Merge per-model history JSON files into a single loss-curve figure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import plot_gan_losses, plot_loss_curves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--save-path", default="./docs/figures/loss_curves.png")
    p.add_argument("--gan-save-path", default="./docs/figures/gan_loss_curves.png")
    args = p.parse_args()

    histories = {}
    for name in ["FVSBN", "NADE", "MADE"]:
        path = Path(args.output_dir) / f"{name.lower()}_history.json"
        if not path.exists():
            print(f"[warn] missing {path}", flush=True)
            continue
        data = json.loads(path.read_text())
        histories[name] = data["epochs"]

    if histories:
        save = Path(args.save_path)
        save.parent.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(histories, save_path=save)
        print(f"Saved {save}", flush=True)
    else:
        print("[warn] no likelihood-model histories found", flush=True)

    gan_path = Path(args.output_dir) / "gan_history.json"
    if gan_path.exists():
        data = json.loads(gan_path.read_text())
        save = Path(args.gan_save_path)
        save.parent.mkdir(parents=True, exist_ok=True)
        plot_gan_losses({"GAN": data["epochs"]}, save_path=save)
        print(f"Saved {save}", flush=True)
    else:
        print(f"[warn] missing {gan_path}", flush=True)


if __name__ == "__main__":
    main()
