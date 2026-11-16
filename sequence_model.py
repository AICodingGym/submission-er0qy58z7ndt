"""Bidirectional conv+GRU for ventilator pressure (v3). One run = one seed.

Main changes vs v2 (diagnosed from the training-log arithmetic, see chat):
  * batch 128 + long schedules -> ~10-20x more optimizer steps than v2 (which got ~4.5k).
  * --minutes N : cosine LR schedule driven by wall-clock time, so a run always finishes
    and uses the whole budget.  (--epochs N still works if --minutes is 0.)
  * EMA of weights (evaluated and saved), conv front-end, residual skip into the head.
  * breath-level and physics features (u_in*R, volume/C, relative position in inspiration...).

Validation mode: trains on 85% of breaths (split fixed by --split-seed), saves raw EMA
predictions for blend.py.  --full trains on all breaths.  --limit-breaths N is a smoke test.

Examples:
    python3 sequence_model.py --limit-breaths 600 --epochs 2          # smoke test
    python3 sequence_model.py --seed 42 --minutes 60                  # one real run
"""
import argparse, copy, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEQ = 80
CONT = [
    "time_step", "step", "dt", "u_in", "u_out", "R", "C", "R_over_C",
    "u_in_cumsum", "u_in_volume", "u_in_cummax",
    "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_lead1", "u_in_lead2",
    "u_in_diff1", "u_in_diff2", "u_out_change",
    # physics-motivated products and breath-level context
    "u_in_R", "vol_over_C", "cum_over_C", "insp_pos", "b_uin_sum", "b_uin_max", "b_insp_len",
]
R_VALS, C_VALS = [5, 20, 50], [10, 20, 50]


def build(frame, with_target):
    f = frame.copy()
    f["_row"] = np.arange(len(f))
    f = f.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    g = f.groupby("breath_id", sort=False)
    f["_pos"] = g.cumcount()
    f["step"] = f["_pos"] / (SEQ - 1)
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
    f["u_in_R"] = f["u_in"] * f["R"]
    f["vol_over_C"] = f["u_in_volume"] / f["C"]
    f["cum_over_C"] = f["u_in_cumsum"] / f["C"]
    f["_ins"] = (f["u_out"] == 0).astype(np.float32)
    f["_ui"] = f["u_in"] * f["_ins"]
    g = f.groupby("breath_id", sort=False)
    f["b_uin_sum"] = g["_ui"].transform("sum")
    f["b_uin_max"] = g["u_in"].transform("max")
    f["b_insp_len"] = g["_ins"].transform("sum")
    f["insp_pos"] = f["_pos"] / f["b_insp_len"].clip(lower=1)
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
    # raw physical inputs for the hybrid model (all known at inference time): valve opening, dt, R, C
    P = np.stack([f["u_in"] / 100.0, f["dt"], f["R"] / 50.0, f["C"] / 50.0], -1).astype(np.float32).reshape(n, SEQ, 4)
    return X, y, u_out, bids, rows, P


class Net(nn.Module):
    """arch='gru'   : conv + BiGRU, pressure read straight from the head.
    arch='hybrid': pressure = physics term + learned residual, with the physics term built from
        flow_t   = gain_t * u_in_t              (gain_t learned: valve opening is only a flow proxy)
        volume_t = cumsum(flow_t * dt_t)         (explicit integration over the breath)
        phys_t   = a_t * R * flow_t + b_t * volume_t / C + c_t     (a_t, b_t > 0; c_t free)
    All inputs are control signals known at inference, so there is NO feedback of predicted
    pressure/state and therefore no train/test mismatch (unlike the earlier pressure-lag attempt)."""

    def __init__(self, n_in, hid, layers, dropout, arch="hybrid"):
        super().__init__()
        self.arch = arch
        self.embed = nn.Sequential(nn.Linear(n_in, hid), nn.LayerNorm(hid), nn.GELU())
        self.conv = nn.Conv1d(hid, hid, kernel_size=5, padding=2)   # local shape of the control signal
        self.norm = nn.LayerNorm(hid)
        self.gru = nn.GRU(hid, hid, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(3 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
        if arch == "hybrid":
            self.phys = nn.Linear(3 * hid, 4)

    def forward(self, x, p):
        h = self.embed(x)
        h = self.norm(h + F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2))
        out, _ = self.gru(h)
        z = torch.cat([out, h], dim=-1)
        res = self.head(z).squeeze(-1)
        if self.arch != "hybrid":
            return res
        q = self.phys(z)
        gain = 2.0 * torch.sigmoid(q[..., 0])
        a, b, c = F.softplus(q[..., 1]), F.softplus(q[..., 2]), q[..., 3]
        u_in, dt, R, C = p.unbind(-1)
        flow = gain * u_in
        volume = torch.cumsum(flow * dt, dim=1)
        return a * R * flow + b * volume / C + c + res


@torch.no_grad()
def predict(model, X, P, bs=1024):
    model.eval()
    return torch.cat([model(X[i:i + bs], P[i:i + bs]).float().cpu() for i in range(0, len(X), bs)]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="model/initialisation seed")
    p.add_argument("--split-seed", type=int, default=42,
                   help="validation split seed; keep 42 so every seed and LightGBM share breaths")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--minutes", type=float, default=0.0,
                   help="wall-clock training budget; schedule follows time. 0 = use --epochs")
    p.add_argument("--epochs", type=int, default=100, help="used only when --minutes is 0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    p.add_argument("--exp-weight", type=float, default=0.0,
                   help="loss weight on expiratory (unscored) steps; 0 = scored steps only")
    p.add_argument("--arch", choices=["hybrid", "gru"], default="hybrid",
                   help="hybrid = physics-structured output + residual; gru = plain head")
    p.add_argument("--full", action="store_true", help="train on all breaths, no validation")
    p.add_argument("--limit-breaths", type=int, default=0, help="smoke test on the first N breaths")
    a = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    nrows = a.limit_breaths * SEQ if a.limit_breaths else None
    train = pd.read_csv(DATA / "train.csv", nrows=nrows)
    test = pd.read_csv(DATA / "test.csv", nrows=nrows)
    X, y, u_out, bids, rows, P = build(train, True)
    Xt, _, _, _, trows, Pt = build(test, False)
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

    Xn = torch.from_numpy((X - mean) / std).float()
    Yn = torch.from_numpy((y - ymean) / ystd).float()
    Mk = torch.from_numpy(u_out == 0)
    Xtn = torch.from_numpy((Xt - mean) / std).float().to(device)
    Xtr, Ytr, Mtr = Xn[tr].to(device), Yn[tr].to(device), Mk[tr].to(device)
    Xva = Xn[va].to(device) if va is not None else None
    Ptr = torch.from_numpy(P[tr]).to(device)
    Pva = torch.from_numpy(P[va]).to(device) if va is not None else None
    Ptn = torch.from_numpy(Pt).to(device)

    model = Net(X.shape[-1], a.hidden, a.layers, a.dropout, a.arch).to(device)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    nb = len(tr)
    steps_per_epoch = math.ceil(nb / a.batch_size)
    total_steps = steps_per_epoch * a.epochs
    budget = a.minutes * 60.0
    t_start = time.time()
    print(f"{steps_per_epoch} steps/epoch; "
          + (f"time budget {a.minutes:.0f} min" if budget else f"{total_steps:,} total steps"), flush=True)

    def progress(step):
        return (time.time() - t_start) / budget if budget else step / total_steps

    def lr_at(pr, warm=0.03):
        if pr < warm:
            return a.lr * max(pr / warm, 0.02)
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (pr - warm) / (1 - warm))))

    def ema_update(step):
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)

    eval_model = ema if a.ema > 0 else model
    best, best_ep, best_state, curve = float("inf"), 0, None, []
    step, ep, done, run_loss = 0, 0, False, float("nan")
    while not done:
        ep += 1; t0 = time.time(); s0 = step
        model.train()
        perm = torch.randperm(nb, device=device)
        for i in range(0, nb, a.batch_size):
            pr = progress(step)
            if pr >= 1.0:
                done = True; break
            for gr in opt.param_groups:
                gr["lr"] = lr_at(pr)
            idx = perm[i:i + a.batch_size]
            pred = model(Xtr[idx], Ptr[idx])
            w = Mtr[idx].float() + a.exp_weight * (~Mtr[idx]).float()
            loss = ((pred - Ytr[idx]).abs() * w).sum() / w.sum()          # plain L1 on scored steps
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if a.ema > 0:
                ema_update(step)
            step += 1
            if step == 1 or step % 50 == 0:
                run_loss = float(loss) * ystd                              # also syncs the GPU queue
        if not budget and ep >= a.epochs:
            done = True
        msg = (f"epoch {ep:03d}  {time.time() - t0:5.0f}s  {max(step - s0, 1) / max(time.time() - t0, 1e-9):5.1f} it/s  "
               f"lr={lr_at(min(progress(step), 1.0)):.2e}  train L1~{run_loss:.4f}")
        if va is not None and step > s0:
            pv = predict(eval_model, Xva, Pva) * ystd + ymean
            m = u_out[va] == 0
            mae = float(np.abs(pv[m] - y[va][m]).mean())
            curve.append((ep, step, mae))
            msg += f"  val MAE={mae:.5f}"
            if mae < best:
                best, best_ep, best_state = mae, ep, copy.deepcopy(eval_model.state_dict())
        print(msg, flush=True)

    tag = f"seed{a.seed}"
    if best_state is not None:
        eval_model.load_state_dict(best_state)
        print(f"best validation MAE: {best:.5f} at epoch {best_ep}/{ep}  ({step:,} optimizer steps)")
        if best_ep >= ep - 1:
            print("NOTE: best epoch is the last one -> a longer budget (--minutes) should still help.")
        pd.DataFrame(curve, columns=["epoch", "step", "val_mae"]).to_csv(ROOT / f"seq_curve_{tag}.csv", index=False)

    if va is not None:
        pv = predict(eval_model, Xva, Pva) * ystd + ymean              # RAW predictions; blend.py decides on clipping
        np.save(ROOT / f"seq_val_{tag}.npy", pv.reshape(-1))
        np.save(ROOT / "seq_val_rows.npy", rows[va].reshape(-1))
    pt = predict(eval_model, Xtn, Ptn) * ystd + ymean
    out = np.empty(len(test), dtype=np.float32)
    out[trows.reshape(-1)] = pt.reshape(-1)
    np.save(ROOT / f"seq_test{'_full' if a.full else ''}_{tag}.npy", out)
    sub = ROOT / f"submission_seq_{tag}.csv"
    pd.DataFrame({"id": test["id"].to_numpy(), "pressure": out}).to_csv(sub, index=False)
    print("wrote", sub)


if __name__ == "__main__":
    main()
