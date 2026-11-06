import json
import math
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation import save_progress_grid

LN2 = math.log(2.0)
@dataclass
class EpochStates:
    epoch:int
    train_nll_per_pixel:float
    test_nll_per_pixel:float
    train_bpd: float
    test_bpd: float
    epoch_seconds: float

@dataclass
class TrainingHistory:
    model: str
    epochs: list[EpochStates] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"model": self.model, "epochs": [asdict(e) for e in self.epochs]}
        )

def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optim=None) -> float:
    is_train = optim is not None
    model.train(is_train)
    total_nll, total_pixels = 0.0, 0
    for data, _ in loader:
        data = data.view(data.size(0), -1).to(device, non_blocking=True)
        log_probs = model(data)
        nll = -log_probs.sum()
        if is_train:
            optim.zero_grad(set_to_none=True)
            (nll / data.size(0)).backward()
            optim.step()
        total_nll += nll.item()
        total_pixels += data.numel()
    return total_nll / total_pixels

def train(
        model:nn.Module,
        device: torch.device,
        train_loader:DataLoader,
        test_loader:DataLoader,
        epochs:int,
        lr:float,
        output_dir:Path,
        model_name:str,
        weight_decay: float = 0.0,
        log_every: int = 1,
        progress_every: int = 0,
        progress_path=None,
        progress_cols: int = 4,
) -> TrainingHistory:
    device = torch.device(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainingHistory(model=model_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_test = float("inf")
    progress_stages = []
    for epoch in range(1, 1+epochs):
        t0 = time.time()
        train_nll = _run_epoch(model, train_loader,device,optim)
        with torch.no_grad():
            test_nll = _run_epoch(model, test_loader,device, None)
        stats = EpochStates(
            epoch=epoch,
            train_nll_per_pixel=train_nll,
            test_nll_per_pixel=test_nll,
            train_bpd= train_nll / LN2,
            test_bpd= test_nll / LN2,
            epoch_seconds=time.time() - t0
        )
        history.epochs.append(stats)
        if progress_every and epoch % progress_every == 0:
            with torch.no_grad():
                stage = model.sample(progress_cols, device=device).cpu()
            progress_stages.append(stage)
        if epoch % log_every == 0 or epoch == epochs:
            print(
                f"[{model_name}] epoch {epoch:03d}/{epochs} "
                f"train {stats.train_bpd:.4f} bpd  test {stats.test_bpd:.4f} bpd  "
                f"({stats.epoch_seconds:.1f}s)",
                flush=True,
            )

        if test_nll < best_test:
            best_test = test_nll
            torch.save(
                {"model_state_dict": model.state_dict(), "stats": asdict(stats)},
                output_dir / f"{model_name}_best.pt",
            )

        (output_dir / f"{model_name}_history.json").write_text(history.to_json())

    if progress_stages and progress_path is not None:
        save_progress_grid(progress_stages, Path(progress_path))

    torch.save(
        {"model_state_dict": model.state_dict(), "final_epoch": epochs},
        output_dir / f"{model_name}_final.pt",
    )
    return history
