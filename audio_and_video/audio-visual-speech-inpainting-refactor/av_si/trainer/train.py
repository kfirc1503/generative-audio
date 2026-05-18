"""Training loop for StackedBLSTM speech inpainter (paper §3.3)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.blstm import StackedBLSTM


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 8
    max_epochs: int = 50
    early_stop_patience: int = 5
    ctc_weight: float = 0.001
    grad_clip: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/blstm"


def _move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _loss_step(model, batch, ctc_weight: float):
    out = model(batch["audio_in"], batch["mask"], batch.get("video_in"))
    rec = nn.functional.l1_loss(out["y_hat"], batch["target_norm"])

    total = rec
    parts = {"l1": rec.item()}
    if "phone_logits" in out and "phones" in batch:
        # CTC needs (T, B, C) log-probs + lengths.
        log_probs = nn.functional.log_softmax(out["phone_logits"], dim=-1).transpose(0, 1)
        T, B = log_probs.shape[:2]
        input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
        target_lengths = torch.tensor([len(p) for p in batch["phones"]], dtype=torch.long, device=log_probs.device)
        targets = torch.cat(batch["phones"]).to(log_probs.device)
        ctc = nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)
        total = rec + ctc_weight * ctc
        parts["ctc"] = ctc.item()
    return total, parts


def train(model: StackedBLSTM, train_ds, val_ds, cfg: TrainConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    stale = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_l1 = 0.0
        for batch in train_dl:
            batch = _move(batch, cfg.device)
            loss, parts = _loss_step(model, batch, cfg.ctc_weight)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_l1 += parts["l1"]
        train_l1 /= max(1, len(train_dl))

        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = _move(batch, cfg.device)
                _, parts = _loss_step(model, batch, cfg.ctc_weight)
                val_l1 += parts["l1"]
        val_l1 /= max(1, len(val_dl))

        history.append({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})
        print(f"epoch {epoch:3d}  train_l1={train_l1:.4f}  val_l1={val_l1:.4f}")

        if val_l1 < best_val:
            best_val = val_l1
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_l1": val_l1}, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"early stop at epoch {epoch} (no val improvement for {stale} epochs)")
                break
    return history
