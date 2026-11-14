"""Dilated temporal CNN sequence-to-sequence model for ventilator pressures.

Run ``python3 sequence_model.py`` for breath-grouped validation.
Run ``python3 sequence_model.py --predict --output sequence_submission.csv``
to fit on all training breaths and write a submission file.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEQ_LEN = 80


def prepare_sequences(frame: pd.DataFrame, with_target: bool = False):
    """Sort rows into breaths and return features, row mappings, and metadata."""
    frame = frame.copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int32)
    frame = frame.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    groups = frame.groupby("breath_id", sort=False)
    frame["step"] = groups.cumcount().astype(np.float32) / (SEQ_LEN - 1)
    frame["dt"] = groups["time_step"].diff().fillna(0)
    frame["u_in_cumsum"] = groups["u_in"].cumsum()
    frame["_u_in_dt"] = frame["u_in"] * frame["dt"]
    frame["u_in_volume"] = frame.groupby("breath_id", sort=False)["_u_in_dt"].cumsum()
    frame["u_in_delta"] = groups["u_in"].diff().fillna(0)
    frame["R_over_C"] = frame["R"] / frame["C"]

    sizes = groups.size().to_numpy()
    if not np.all(sizes == SEQ_LEN):
        raise ValueError(f"Expected {SEQ_LEN} rows per breath; got sizes {np.unique(sizes)}")
    n_breaths = len(sizes)
    u_in = frame["u_in"].to_numpy(dtype=np.float32).reshape(n_breaths, SEQ_LEN)
    u_out_seq = frame["u_out"].to_numpy(dtype=np.float32).reshape(n_breaths, SEQ_LEN)

    def past(values: np.ndarray, lag: int) -> np.ndarray:
        result = np.zeros_like(values)
        result[:, lag:] = values[:, :-lag]
        return result

    def future(values: np.ndarray, lead: int) -> np.ndarray:
        result = np.zeros_like(values)
        result[:, :-lead] = values[:, lead:]
        return result

    feature_map = {
        name: frame[name].to_numpy(dtype=np.float32).reshape(n_breaths, SEQ_LEN)
        for name in (
            "time_step", "step", "R", "C", "R_over_C", "dt",
            "u_in_cumsum", "u_in_volume", "u_in_delta",
        )
    }
    feature_map["u_in"] = u_in
    feature_map["u_out"] = u_out_seq
    feature_map["u_in_R"] = u_in * feature_map["R"]
    feature_map["u_in_C"] = u_in * feature_map["C"]
    feature_map["u_in_RC"] = u_in * feature_map["R"] * feature_map["C"]
    for signal, values, offsets in (
        ("u_in", u_in, (1, 2, 3, 5, 10, 20)),
        ("u_out", u_out_seq, (1, 2, 3, 5, 10)),
    ):
        for offset in offsets:
            feature_map[f"{signal}_lag{offset}"] = past(values, offset)
            feature_map[f"{signal}_lead{offset}"] = future(values, offset)
    feature_map["u_in_delta_forward"] = future(u_in, 1) - u_in

    cumulative = np.cumsum(u_in, axis=1)
    reverse_cumulative = np.flip(np.cumsum(np.flip(u_in, axis=1), axis=1), axis=1)
    for window in (3, 5, 10, 20):
        prior_cumulative = past(cumulative, window)
        feature_map[f"u_in_past_sum{window}"] = cumulative - prior_cumulative
        feature_map[f"u_in_past_mean{window}"] = feature_map[f"u_in_past_sum{window}"] / window
        future_start = future(reverse_cumulative, 1)
        future_end = future(reverse_cumulative, window + 1) if window + 1 < SEQ_LEN else np.zeros_like(u_in)
        feature_map[f"u_in_future_sum{window}"] = future_start - future_end
        feature_map[f"u_in_future_mean{window}"] = feature_map[f"u_in_future_sum{window}"] / window

    X = np.stack(list(feature_map.values()), axis=-1).astype(np.float32, copy=False)
    row_order = frame["_row_order"].to_numpy().reshape(n_breaths, SEQ_LEN)
    breath_ids = frame.groupby("breath_id", sort=False)["breath_id"].first().to_numpy()
    u_out = frame["u_out"].to_numpy(dtype=np.int8).reshape(n_breaths, SEQ_LEN)
    y = None
    if with_target:
        y = frame["pressure"].to_numpy(dtype=np.float32).reshape(n_breaths, SEQ_LEN)
    return X, y, u_out, breath_ids, row_order


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.dropout(self.layers(x)))


class BreathTCN(nn.Module):
    """Dilated residual temporal CNN with receptive field spanning each breath."""

    def __init__(self, input_size: int, channels: int = 128):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            TemporalBlock(channels, dilation, dropout=0.10)
            for dilation in (1, 2, 4, 8, 16, 32)
        ])
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(channels, 1, kernel_size=1),
        )
        self.skip = nn.Conv1d(input_size, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        sequence = self.blocks(self.input_projection(x))
        return (self.head(sequence) + self.skip(x)).squeeze(1)


def predict_in_batches(model, features: torch.Tensor, indices: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long, device=features.device)
            outputs.append(model(features[idx]).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    u_out: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray | None,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int = 8,
    requested_device: str = "auto",
):
    """Train with scored-row SmoothL1 loss; return model and best epoch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4) - 1)))
    print(f"Training device: {device}", flush=True)

    feature_mean = features[train_idx].reshape(-1, features.shape[-1]).mean(axis=0)
    feature_std = features[train_idx].reshape(-1, features.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    target_rows = u_out[train_idx] == 0
    target_values = targets[train_idx][target_rows]
    target_mean = float(target_values.mean())
    target_std = float(target_values.std())
    if target_std < 1e-6:
        target_std = 1.0

    X = torch.from_numpy(((features - feature_mean) / feature_std).astype(np.float32)).to(device)
    Y = torch.from_numpy(((targets - target_mean) / target_std).astype(np.float32)).to(device)
    mask = torch.from_numpy(u_out == 0).to(device)
    model = BreathTCN(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    last_epoch = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        order = rng.permutation(train_idx)
        total_loss = 0.0
        scored_count = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            x_batch = X[idx]
            y_batch = Y[idx]
            mask_batch = mask[idx]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = F.l1_loss(prediction[mask_batch], y_batch[mask_batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * int(mask_batch.sum())
            scored_count += int(mask_batch.sum())

        if valid_idx is None:
            print(f"epoch {epoch:03d} train scored MAE={total_loss / scored_count:.5f}", flush=True)
            continue

        val_pred = predict_in_batches(model, X, valid_idx, batch_size)
        val_mask = u_out[valid_idx] == 0
        val_mae = float(np.abs(val_pred[val_mask] * target_std + target_mean - targets[valid_idx][val_mask]).mean())
        scheduler.step(val_mae)
        print(
            f"epoch {epoch:03d} train scored MAE={total_loss / scored_count:.5f} "
            f"validation inspiratory MAE={val_mae:.6f}",
            flush=True,
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if valid_idx is not None and best_state is not None:
        model.load_state_dict(best_state)
    reported_epoch = best_epoch if valid_idx is not None else last_epoch
    return model, reported_epoch, best_mae, feature_mean, feature_std, target_mean, target_std


def breath_group_split(breath_ids: np.ndarray, valid_size: float, seed: int):
    indices = np.arange(len(breath_ids))
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    return next(splitter.split(indices, groups=breath_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="fit all training breaths and write a submission")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "sequence_submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv", dtype={"R": "int8", "C": "int8", "u_out": "int8"})
    X_train, y_train, u_out_train, breath_ids, train_rows = prepare_sequences(train, with_target=True)
    del train

    if args.predict:
        test = pd.read_csv(DATA / "test.csv", dtype={"R": "int8", "C": "int8", "u_out": "int8"})
        X_test, _, _, _, test_rows = prepare_sequences(test, with_target=False)
        test_ids = test["id"].to_numpy()
        del test
        train_idx = np.arange(len(X_train))
        fit_idx, valid_idx = breath_group_split(breath_ids, args.valid_size, args.seed)
        print(
            f"Selecting epochs on {len(fit_idx):,} fit breaths and "
            f"{len(valid_idx):,} validation breaths",
            flush=True,
        )
        selection_model, selected_epochs, selection_mae, *_ = train_model(
            X_train, y_train, u_out_train, fit_idx, valid_idx,
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            patience=args.patience, requested_device=args.device,
        )
        print(f"Selected epoch {selected_epochs} (validation MAE {selection_mae:.6f})", flush=True)
        del selection_model
        gc.collect()
        model, _, _, mean, std, target_mean, target_std = train_model(
            X_train, y_train, u_out_train, train_idx, None,
            epochs=selected_epochs, batch_size=args.batch_size, seed=args.seed,
            patience=args.patience, requested_device=args.device,
        )
        normalized_test = torch.from_numpy(((X_test - mean) / std).astype(np.float32)).to(
            next(model.parameters()).device
        )
        test_idx = np.arange(len(X_test))
        pred_scaled = predict_in_batches(model, normalized_test, test_idx, args.batch_size)
        pred_sequence = pred_scaled * target_std + target_mean
        pred_rows = np.empty(len(test_ids), dtype=np.float32)
        pred_rows[test_rows.reshape(-1)] = pred_sequence.reshape(-1)
        submission = pd.DataFrame({"id": test_ids, "pressure": pred_rows})
        submission.to_csv(args.output, index=False)
        print(f"Wrote {args.output} ({len(submission):,} predictions; {epochs_used} epochs)")
        return

    train_idx, valid_idx = breath_group_split(breath_ids, args.valid_size, args.seed)
    print(f"Training on {len(train_idx):,} breaths; validating on {len(valid_idx):,} breaths", flush=True)
    _, best_epoch, best_mae, *_ = train_model(
        X_train, y_train, u_out_train, train_idx, valid_idx,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        patience=args.patience, requested_device=args.device,
    )
    print(f"Breath-grouped sequence-model scored MAE: {best_mae:.6f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
