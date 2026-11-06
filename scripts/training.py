import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np
import torch
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models import fvsbn, nade, gan
from data import IMAGE_SIZE, data_loaders,input_size, data_gan_loaders
from train import train
from train_gan import train_gan

def build_model(name: str, input_size: int, hidden_size:int,latten_size:int, hidden_layers:int):
    name = name.lower()
    if name == "fvsbn":
        return fvsbn.Fvsbn(input_size)
    if name == "nade":
        return nade.nade(input_size, hidden_size)
    if name == "gan":
        return gan.Generator(latent_dim=latten_size)
    return ValueError(f"unknown model {name!r}")

def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["fvsbn", "nade", "gan"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--run-name", default=None,
                   help="checkpoint/history basename; defaults to the model name")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--g-lr", type=float, default=1e-4)
    p.add_argument("--d-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-size", type=int, default=500)
    p.add_argument("--hidden-layers", type=int, default=2, help="MADE only")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps" )
    p.add_argument("--progress-every", type=int, default=0,
                   help="save a row of samples every N epochs (0 disables)")
    p.add_argument("--progress-path", default=None,
                   help="path for the training-progress grid image")
    args = p.parse_args()
    set_seed(args.seed)
    run_name = args.run_name or args.model
    if args.model == "gan":
        train_loader, test_loader = data_gan_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader, test_loader = data_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False
        )
    model = build_model(args.model, input_size, args.hidden_size,args.latent_dim, args.hidden_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}  Input dim: {input_size}", flush=True)
    if args.model == "gan":
        gen = gan.Generator(args.latent_dim)
        disc = gan.Discriminator()
        train_gan(
            generator=gen,
            discriminator=disc,
            device=args.device,
            train_loader=train_loader,
            epochs=args.epochs,
            g_lr=args.g_lr,
            d_lr=args.d_lr,
            output_dir=Path(args.output_dir),
            run_name=run_name,
            latent_dim=args.latent_dim,
            progress_every=args.progress_every,
            progress_path=args.progress_path,
            )
    else:
        train(
            model=model, train_loader=train_loader, test_loader=test_loader,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            device=args.device,
            output_dir=Path(args.output_dir),
            model_name=run_name,
            progress_every=args.progress_every,
            progress_path=args.progress_path,
        )


if __name__ == "__main__":
    main()
