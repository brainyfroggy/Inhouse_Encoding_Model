from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics, write_report


@dataclass
class NNConfig:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int = 80
    patience: int = 10


def configs() -> list[NNConfig]:
    return [
        NNConfig("mlp_256_do0.1_lr1e-3", (256,), 0.1, 1e-3, 1e-4, 256),
        NNConfig("mlp_512_do0.1_lr1e-3", (512,), 0.1, 1e-3, 1e-4, 256),
        NNConfig("mlp_1024_512_do0.2_lr1e-3", (1024, 512), 0.2, 1e-3, 1e-4, 256),
        NNConfig("mlp_1024_512_do0.4_lr3e-4", (1024, 512), 0.4, 3e-4, 1e-4, 256),
        NNConfig("mlp_2048_512_do0.3_lr3e-4", (2048, 512), 0.3, 3e-4, 3e-4, 128),
        NNConfig("mlp_512_256_do0.3_lr1e-4", (512, 256), 0.3, 1e-4, 1e-3, 256),
    ]


class ResponseDataset(Dataset):
    def __init__(self, x, y_valence, y_arousal, indices, mean, std):
        self.x = x
        self.y_valence = y_valence
        self.y_arousal = y_arousal
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = (np.asarray(self.x[i], dtype=np.float32) - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_valence[i], dtype=torch.long),
            torch.tensor(self.y_arousal[i], dtype=torch.long),
        )


class MultiTaskMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = n_features
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.valence_head = nn.Linear(prev, 3)
        self.arousal_head = nn.Linear(prev, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.valence_head(z), self.arousal_head(z)


def load_dataset(dataset_dir: Path):
    x = np.load(dataset_dir / "X_float32.npy", mmap_mode="r")
    yv = np.load(dataset_dir / "y_valence.npy")
    ya = np.load(dataset_dir / "y_arousal.npy")
    sample = pd.read_csv(dataset_dir / "sample_index.csv")
    idx = {
        split: sample.index[sample["split"] == split].to_numpy()
        for split in ["train", "val", "test"]
    }
    return x, yv, ya, idx


def compute_standardizer(x, train_idx):
    x_train = np.asarray(x[train_idx], dtype=np.float64)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(y, train_idx, device):
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, loss_valence, loss_arousal):
    model.eval()
    rows = []
    total_loss = 0.0
    n = 0
    yv_true, yv_pred, ya_true, ya_pred = [], [], [], []
    for xb, yv, ya in loader:
        xb = xb.to(device)
        yv = yv.to(device)
        ya = ya.to(device)
        out_v, out_a = model(xb)
        loss = loss_valence(out_v, yv) + loss_arousal(out_a, ya)
        total_loss += float(loss.item()) * len(xb)
        n += len(xb)
        yv_true.append(yv.cpu().numpy())
        ya_true.append(ya.cpu().numpy())
        yv_pred.append(out_v.argmax(dim=1).cpu().numpy())
        ya_pred.append(out_a.argmax(dim=1).cpu().numpy())
    yv_true = np.concatenate(yv_true)
    ya_true = np.concatenate(ya_true)
    yv_pred = np.concatenate(yv_pred)
    ya_pred = np.concatenate(ya_pred)
    return {
        "loss": total_loss / max(n, 1),
        "valence_accuracy": float((yv_true == yv_pred).mean()),
        "arousal_accuracy": float((ya_true == ya_pred).mean()),
        "valence_balanced_accuracy": float(balanced_accuracy_score(yv_true, yv_pred)),
        "arousal_balanced_accuracy": float(balanced_accuracy_score(ya_true, ya_pred)),
        "valence_macro_f1": float(f1_score(yv_true, yv_pred, average="macro", zero_division=0)),
        "arousal_macro_f1": float(f1_score(ya_true, ya_pred, average="macro", zero_division=0)),
        "valence_true": yv_true,
        "valence_pred": yv_pred,
        "arousal_true": ya_true,
        "arousal_pred": ya_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x, yv, ya, idx = load_dataset(args.dataset_dir)
    mean, std = compute_standardizer(x, idx["train"])
    np.save(args.out_dir / "train_feature_mean.npy", mean)
    np.save(args.out_dir / "train_feature_std.npy", std)

    train_ds = ResponseDataset(x, yv, ya, idx["train"], mean, std)
    val_ds = ResponseDataset(x, yv, ya, idx["val"], mean, std)
    test_ds = ResponseDataset(x, yv, ya, idx["test"], mean, std)

    rows = []
    final_rows = []
    for cfg in configs():
        print(f"Training {cfg.name}", flush=True)
        cfg_dir = args.out_dir / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        model = MultiTaskMLP(x.shape[1], cfg.hidden, cfg.dropout).to(device)
        loss_v = nn.CrossEntropyLoss(weight=class_weights(yv, idx["train"], device))
        loss_a = nn.CrossEntropyLoss(weight=class_weights(ya, idx["train"], device))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

        best_val = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start = time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            n_train = 0
            for xb, yvb, yab in train_loader:
                xb = xb.to(device)
                yvb = yvb.to(device)
                yab = yab.to(device)
                opt.zero_grad(set_to_none=True)
                out_v, out_a = model(xb)
                loss = loss_v(out_v, yvb) + loss_a(out_a, yab)
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * len(xb)
                n_train += len(xb)

            train_eval = evaluate(model, train_loader, device, loss_v, loss_a)
            val_eval = evaluate(model, val_loader, device, loss_v, loss_a)
            scheduler.step(val_eval["loss"])
            row = {
                "model": cfg.name,
                "epoch": epoch,
                "train_optim_loss": train_loss / max(n_train, 1),
                **{f"train_{k}": v for k, v in train_eval.items() if not k.endswith(("true", "pred"))},
                **{f"val_{k}": v for k, v in val_eval.items() if not k.endswith(("true", "pred"))},
                "lr": opt.param_groups[0]["lr"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out_dir / "nn_epoch_metrics.csv", index=False)
            print(row, flush=True)

            if val_eval["loss"] < best_val:
                best_val = val_eval["loss"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), cfg_dir / "best_model.pt")
            else:
                bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

        model.load_state_dict(torch.load(cfg_dir / "best_model.pt", map_location=device))
        elapsed = time.time() - start
        for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            ev = evaluate(model, loader, device, loss_v, loss_a)
            for target in ["valence", "arousal"]:
                m = classification_metrics(ev[f"{target}_true"], ev[f"{target}_pred"])
                m.update(
                    {
                        "model": cfg.name,
                        "target": target,
                        "split": split,
                        "best_epoch": best_epoch,
                        "fit_seconds": elapsed,
                    }
                )
                final_rows.append(m)
                write_report(
                    cfg_dir / f"{split}_{target}_report.json",
                    ev[f"{target}_true"],
                    ev[f"{target}_pred"],
                )
        pd.DataFrame(final_rows).to_csv(args.out_dir / "nn_final_metrics.csv", index=False)

    print(pd.DataFrame(final_rows).sort_values(["target", "split", "macro_f1"], ascending=[True, True, False]))


if __name__ == "__main__":
    main()
