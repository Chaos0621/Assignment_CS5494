import json
import math
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation import save_progress_grid

@dataclass
class GanEpochStates:
    epoch: int
    d_loss: float
    g_loss: float
    epoch_seconds: float


@dataclass
class GanTrainingHistory:
    model: str
    epochs: list[GanEpochStates] = field(default_factory=list)

    def to_json(self):
        return json.dumps({"model": self.model, "epochs": [asdict(e) for e in self.epochs]})


def train_gan(
    generator,
    discriminator,
    device,
    train_loader,
    epochs,
    g_lr,
    d_lr,
    output_dir,
    run_name,
    latent_dim=100,
    log_every=1,
    real_label_smoothing=0.9,
    noise_std=0.1,
    num_classes=10,
    wrong_label_weight=0.5,
    progress_every=0,
    progress_path=None,
    progress_cols=4,
):
    device = torch.device(device)
    generator.to(device)
    discriminator.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = GanTrainingHistory(model=run_name)
    progress_stages = []

    for epoch in range(1 , 1+epochs):
        t0 = time.time()
        d_sum = g_sum = 0.0
        n = 0
        for real, labels in train_loader:
            real = real.to(device)
            labels = labels.to(device)
            real = real + noise_std * torch.randn_like(real)
            b = real.size(0)
            real_label = torch.full((b, 1), real_label_smoothing, device=device)
            fake_label = torch.zeros(b, 1, device=device)
            gen_label = torch.ones(b, 1, device=device)

            x = torch.randn(b, latent_dim, device=device)
            fake = generator(x, labels)
            fake_for_d = fake.detach() + noise_std * torch.randn_like(fake)

            d_loss = criterion(discriminator(real, labels), real_label) + criterion(discriminator(fake_for_d, labels), fake_label)

            # Real image paired with a wrong label should also be treated as fake,
            # otherwise the discriminator can learn to ignore the label entirely.
            wrong_labels = (labels + torch.randint(1, num_classes, (b,), device=device)) % num_classes
            d_loss = d_loss + wrong_label_weight * criterion(discriminator(real, wrong_labels), fake_label)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            g_loss = criterion(discriminator(fake, labels), gen_label)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            d_sum += d_loss.item() * b
            g_sum += g_loss.item() * b
            n += b
        stats = GanEpochStates(
            epoch=epoch,
            d_loss=d_sum / n,
            g_loss=g_sum / n,
            epoch_seconds=time.time() - t0,
        )
        history.epochs.append(stats)
        if progress_every and epoch % progress_every == 0:
            with torch.no_grad():
                labels = torch.arange(progress_cols, device=device)
                stage = generator.sample(progress_cols, device=device, labels=labels).cpu()
            progress_stages.append(stage)

        if epoch % log_every == 0 or epoch == epochs:
            print(f"[{run_name}] epoch {epoch}/{epochs} "
                  f"d_loss {stats.d_loss:.4f} g_loss {stats.g_loss:.4f} "
                  f"({stats.epoch_seconds:.1f}s)", flush=True)

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(
                {"generator": generator.state_dict(),
                 "discriminator": discriminator.state_dict(),
                 "epoch": epoch},
                output_dir / f"{run_name}_gan_epoch{epoch}.pt",
            )

        (output_dir / f"{run_name}_history.json").write_text(history.to_json())

    if progress_stages and progress_path is not None:
        save_progress_grid(progress_stages, Path(progress_path))

    return history




