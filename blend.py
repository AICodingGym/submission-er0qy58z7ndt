"""Seed-ensemble the GRU, blend with LightGBM, test clipping, report error breakdowns.

Needs (all in the project folder):
  seq_val_seed*.npy, seq_test_seed*.npy, seq_val_rows.npy   from sequence_model.py
  lgb_val_rows.npy, lgb_val_pred.npy, lgb_test_pred.npy      from train_model.py (optional)
Everything is chosen on VALIDATION MAE (u_out == 0 rows), not on the public score.
"""
import glob
import re

import numpy as np
import pandas as pd

tr = pd.read_csv("data/train.csv", usecols=["breath_id", "R", "C", "pressure", "u_out"])
tr["step"] = tr.groupby("breath_id").cumcount()
y, u = tr["pressure"].to_numpy(), tr["u_out"].to_numpy()
lo, hi = float(y.min()), float(y.max())

rows = np.load("seq_val_rows.npy")
scored = np.sort(rows[u[rows] == 0])
truth = y[scored]
mae = lambda p: float(np.abs(p - truth).mean())
on_scored = lambda arr, idx: pd.Series(arr, index=idx).loc[scored].to_numpy()

# ---- 1. seed ensemble -------------------------------------------------------
files = sorted(glob.glob("seq_val_seed*.npy"))
seeds = [re.search(r"seed(\d+)", f).group(1) for f in files]
assert seeds, "no seq_val_seed*.npy files found; run sequence_model.py first"
preds = [on_scored(np.load(f), rows) for f in files]
for s, p in zip(seeds, preds):
    print(f"seed {s}: {mae(p):.5f}")
# MAE is minimised by a median-type predictor, so with >=3 seeds test mean vs median on validation.
aggs = {"mean": lambda a: np.mean(a, axis=0)}
if len(seeds) >= 3:
    aggs["median"] = lambda a: np.median(a, axis=0)
agg_scores = {k: mae(f(preds)) for k, f in aggs.items()}
agg = min(agg_scores, key=agg_scores.get)
print("GRU ensemble (" + str(len(seeds)) + " seeds): " + "  ".join(f"{k}={v:.5f}" for k, v in agg_scores.items()) + f"  -> using {agg}")
seq_val = aggs[agg](preds)

# Test predictions: prefer models trained on ALL breaths (sequence_model.py --full) when available;
# the blend weight / clipping are still chosen on the validation models above.
full_files = [f"seq_test_full_seed{s}.npy" for s in seeds]
import os
if all(os.path.exists(f) for f in full_files):
    test_stack = [np.load(f) for f in full_files]
    print("test predictions: using FULL-data models")
else:
    test_stack = [np.load(f"seq_test_seed{s}.npy") for s in seeds]
    print("test predictions: using 85%-data validation models (run --full per seed for slightly better test results)")
seq_test = aggs[agg](test_stack)

# ---- 2. LightGBM blend ------------------------------------------------------
lgb_val = lgb_test = None
try:
    lgb_rows = np.load("lgb_val_rows.npy")
    assert set(lgb_rows) == set(scored), "LightGBM and GRU validation breaths differ (check split seeds)"
    lgb_val = on_scored(np.load("lgb_val_pred.npy"), lgb_rows)
    lgb_test = np.load("lgb_test_pred.npy")
    print(f"LightGBM: {mae(lgb_val):.5f}")
except FileNotFoundError:
    print("no LightGBM files found; using GRU ensemble only")

weights = np.arange(0, 1.01, 0.1) if lgb_val is not None else [1.0]
results = []
for clip in (False, True):
    for w in weights:
        v = w * seq_val + (1 - w) * lgb_val if lgb_val is not None else seq_val
        results.append((mae(np.clip(v, lo, hi) if clip else v), float(w), clip))
if lgb_val is not None:
    print("\nweight on GRU -> validation MAE (unclipped | clipped)")
    for w in weights:
        a = next(r[0] for r in results if r[1] == float(w) and not r[2])
        b = next(r[0] for r in results if r[1] == float(w) and r[2])
        print(f"  {w:.1f}   {a:.5f} | {b:.5f}")
best_mae, best_w, best_clip = min(results)
print(f"\nchosen: GRU weight={best_w:.1f}, clip={best_clip}, validation MAE={best_mae:.5f}")

# ---- 3. error breakdown of the chosen blend --------------------------------
v = best_w * seq_val + (1 - best_w) * lgb_val if lgb_val is not None else seq_val
v = np.clip(v, lo, hi) if best_clip else v
err = pd.DataFrame({"R": tr["R"].to_numpy()[scored], "C": tr["C"].to_numpy()[scored],
                    "step": tr["step"].to_numpy()[scored], "abs_err": np.abs(v - truth)})
print("\nMAE by (R, C):")
print(err.groupby(["R", "C"])["abs_err"].agg(["mean", "count"]).round(4).to_string())
print("\nMAE by step (bins of 5):")
print(err.groupby(err["step"] // 5 * 5)["abs_err"].mean().round(4).to_string())

# ---- 4. write the submission -----------------------------------------------
final = best_w * seq_test + (1 - best_w) * lgb_test if lgb_val is not None else seq_test
if best_clip:
    final = np.clip(final, lo, hi)
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
assert len(ids) == len(final) and np.isfinite(final).all()
pd.DataFrame({"id": ids, "pressure": final}).to_csv("submission_blend.csv", index=False)
print("\nwrote submission_blend.csv  (submit only if validation MAE beats your current best)")