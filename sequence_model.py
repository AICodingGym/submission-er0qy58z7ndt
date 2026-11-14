Sequence model · PY
"""Bidirectional GRU for ventilator pressure. One run = one seed.
 
Validation mode (default): trains on 85% of breaths (split fixed by --split-seed),
prints masked MAE each epoch, saves the epoch curve, raw validation predictions
and raw test predictions for ensembling/blending (see blend.py).
--full: trains on all breaths (no validation), saves test predictions.
 
Ensemble example (same validation breaths, different model seeds):
    for s in 42 43 44; do python3 sequence_model.py --seed $s --split-seed 42; done
"""
import argparse, copy, time, math
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
 
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEQ = 80
CONT = [
    "time_step", "step", "dt", "u_in", "u_out", "R", "C", "R_over_C",
    "u_in_cumsum", "u_in_volume", "u_in_cummax",
    "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_lead1", "u_in_lead2",
    "u_in_diff1", "u_in_diff2", "u_out_change",
]
R_VALS, C_VALS = [5, 20, 50], [10, 20, 50]
 
 
def build(frame, with_target):
    f = frame.copy()
    f["_row"] = np.arange(len(f))
    f = f.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    g = f.groupby("breath_id", sort=False)
    f["step"] = g.cumcount() / (SEQ - 1)
    f["dt"] = g["time_step"].diff().fillna(0)
    f["u_in_cumsum"] = g["u_in"].cumsum()
    f["_uin_dt"] = f["u_in"] * f["dt"]
    f["u_in_volume"] = f.groupby("breath_id", sort=False)["_uin_dt"].cumsum()
    f["u_in_cummax"] = g["u_in"].cummax()
    for k in (1, 2, 3):
        f[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
    for k in (1, 2):
        f[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
    f["u_in_diff1"] = f["u_in"] - f["u_in_lag1"]
    f["u_in_diff2"] = f["u_in"] - f["u_in_lag2"]
    f["u_out_change"] = g["u_out"].diff().fillna(0)
    f["R_over_C"] = f["R"] / f["C"]
    cols = list(CONT)
    for r in R_VALS:
        f[f"R_{r}"] = (f["R"] == r).astype(np.float32); cols.append(f"R_{r}")
    for c in C_VALS:
        f[f"C_{c}"] = (f["C"] == c).astype(np.float32); cols.append(f"C_{c}")
    for r in R_VALS:
        for c in C_VALS:
            f[f"RC_{r}_{c}"] = ((f["R"] == r) & (f["C"] == c)).astype(np.float32)
            cols.append(f"RC_{r}_{c}")
 
    sizes = f.groupby("breath_id", sort=False).size().to_numpy()
    assert np.all(sizes == SEQ), "every breath must have 80 rows"
    n = len(sizes)
    X = f[cols].to_numpy(np.float32).reshape(n, SEQ, -1)
    u_out = f["u_out"].to_numpy(np.int8).reshape(n, SEQ)
    rows = f["_row"].to_numpy().reshape(n, SEQ)
    bids = f["breath_id"].to_numpy().reshape(n, SEQ)[:, 0]
    y = f["pressure"].to_numpy(np.float32).reshape(n, SEQ) if with_target else None
    return X, y, u_out, bids, rows
 
 
class Net(nn.Module):
    def __init__(self, n_in, hid, layers, dropout):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, hid), nn.LayerNorm(hid), nn.GELU())
        self.gru = nn.GRU(hid, hid, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
 
    def forward(self, x):
        out, _ = self.gru(self.embed(x))
        return self.head(out).squeeze(-1)
 
 
@torch.no_grad()
def predict(model, X, bs):
    model.eval()
    return torch.cat([model(X[i:i + bs]).float().cpu() for i in range(0, len(X), bs)]).numpy()
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="model/initialisation seed")
    p.add_argument("--split-seed", type=int, default=42,
                   help="validation split seed; keep 42 so every seed and LightGBM share breaths")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--exp-weight", type=float, default=0.0,
                   help="loss weight on expiratory (unscored) steps; 0 = scored steps only")
    p.add_argument("--full", action="store_true", help="train on all breaths, no validation")
    a = p.parse_args()
 
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
 
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X, y, u_out, bids, rows = build(train, True)
    Xt, _, _, _, trows = build(test, False)
    n = len(X)
 
    if a.full:
        tr, va = np.arange(n), None
    else:
        tr, va = next(GroupShuffleSplit(1, test_size=a.valid_size, random_state=a.split_seed)
                      .split(np.arange(n), groups=bids))
        print(f"train breaths {len(tr):,}  validation breaths {len(va):,}", flush=True)
 
    mean = X[tr].reshape(-1, X.shape[-1]).mean(0)
    std = X[tr].reshape(-1, X.shape[-1]).std(0); std[std < 1e-6] = 1.0
    scored = u_out[tr] == 0
    ymean, ystd = float(y[tr][scored].mean()), float(y[tr][scored].std())
    ylo, yhi = float(y[tr].min()), float(y[tr].max())
 
    Xn = torch.from_numpy((X - mean) / std).float()
    Yn = torch.from_numpy((y - ymean) / ystd).float()
    Mk = torch.from_numpy(u_out == 0)
    Xtn = torch.from_numpy((Xt - mean) / std).float().to(device)
    Xtr, Ytr, Mtr = Xn[tr].to(device), Yn[tr].to(device), Mk[tr].to(device)
    Xva = Xn[va].to(device) if va is not None else None
 
    model = Net(X.shape[-1], a.hidden, a.layers, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(len(tr) / a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps)
 
    best, best_ep, best_state, curve = float("inf"), 0, None, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train()
        perm = torch.randperm(len(tr)).to(device)
        for i in range(0, len(tr), a.batch_size):
            idx = perm[i:i + a.batch_size]
            pred = model(Xtr[idx])
            w = Mtr[idx].float() + a.exp_weight * (~Mtr[idx]).float()
            loss = ((pred - Ytr[idx]).abs() * w).sum() / w.sum()      # plain L1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        msg = f"epoch {ep:03d}  {time.time() - t0:5.0f}s  train L1={float(loss) * ystd:.4f}"
        if va is not None:
            pv = predict(model, Xva, 1024) * ystd + ymean
            m = u_out[va] == 0
            mae = float(np.abs(pv[m] - y[va][m]).mean())
            curve.append((ep, mae))
            msg += f"  val MAE={mae:.5f}"
            if mae < best:
                best, best_ep, best_state = mae, ep, copy.deepcopy(model.state_dict())
        print(msg, flush=True)
 
    tag = f"seed{a.seed}"
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best validation MAE: {best:.5f} at epoch {best_ep}/{a.epochs}")
        if best_ep >= a.epochs - 2:
            print("NOTE: best epoch is at the end of training -> try more epochs (e.g. 60-80).")
        pd.DataFrame(curve, columns=["epoch", "val_mae"]).to_csv(ROOT / f"seq_curve_{tag}.csv", index=False)
 
    if va is not None:
        pv = predict(model, Xva, 1024) * ystd + ymean          # RAW (unclipped) predictions
        m = u_out[va] == 0
        raw = float(np.abs(pv[m] - y[va][m]).mean())
        clipped = float(np.abs(np.clip(pv, ylo, yhi)[m] - y[va][m]).mean())
        print(f"validation MAE raw={raw:.5f}  clipped to [{ylo:.2f}, {yhi:.2f}]={clipped:.5f}")
        np.save(ROOT / f"seq_val_{tag}.npy", pv.reshape(-1))
        np.save(ROOT / "seq_val_rows.npy", rows[va].reshape(-1))
    pt = predict(model, Xtn, 1024) * ystd + ymean              # RAW; blend.py decides on clipping
    out = np.empty(len(test), dtype=np.float32)
    out[trows.reshape(-1)] = pt.reshape(-1)
    np.save(ROOT / f"seq_test{'_full' if a.full else ''}_{tag}.npy", out)
    sub = ROOT / f"submission_seq_{tag}.csv"
    pd.DataFrame({"id": test["id"].to_numpy(), "pressure": out}).to_csv(sub, index=False)
    print("wrote", sub)
 
 
if __name__ == "__main__":
    main()