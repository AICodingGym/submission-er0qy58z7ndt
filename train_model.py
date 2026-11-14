"""Breath-aware LightGBM baseline for ventilator pressure prediction.

Run ``python3 train_model.py`` to train and score a grouped validation split.
Add ``--predict`` to fit on all training breaths and write predictions.csv.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build row and within-breath control-history features."""
    frame["_row_order"] = np.arange(len(frame), dtype=np.int32)
    frame = frame.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby("breath_id", sort=False)
    frame["step"] = grouped.cumcount().astype("int8")
    frame["dt"] = grouped["time_step"].diff().fillna(0).astype("float32")

    for col in ("u_in", "u_out"):
        lags = (1, 2, 3, 5, 10, 15, 20) if col == "u_in" else (1, 2, 3, 5, 10)
        for lag in lags:
            frame[f"{col}_lag{lag}"] = grouped[col].shift(lag).fillna(0).astype("float32")

    for window in (3, 5, 10, 15, 20):
        lagged = [grouped["u_in"].shift(i).fillna(0) for i in range(window)]
        frame[f"u_in_sum{window}"] = sum(lagged).astype("float32")
        frame[f"u_in_mean{window}"] = (frame[f"u_in_sum{window}"] / window).astype("float32")

        future = [grouped["u_in"].shift(-i).fillna(0) for i in range(1, window + 1)]
        frame[f"u_in_future_sum{window}"] = sum(future).astype("float32")
        frame[f"u_in_future_mean{window}"] = (frame[f"u_in_future_sum{window}"] / window).astype("float32")

    frame["u_in_dt"] = (frame["u_in"] * frame["dt"]).astype("float32")
    frame["u_in_cumsum"] = grouped["u_in"].cumsum().astype("float32")
    frame["u_in_volume"] = frame.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype("float32")
    frame["u_in_cummax"] = grouped["u_in"].cummax().astype("float32")
    frame["u_in_delta"] = (frame["u_in"] - frame["u_in_lag1"]).astype("float32")
    frame["RC"] = (frame["R"] * frame["C"]).astype("int16")
    frame["R_over_C"] = (frame["R"] / frame["C"]).astype("float32")
    frame["u_in_R"] = (frame["u_in"] * frame["R"]).astype("float32")
    frame["u_in_C"] = (frame["u_in"] * frame["C"]).astype("float32")

    feature_cols = [
        "R", "C", "RC", "R_over_C", "time_step", "step", "dt", "u_in", "u_out",
        "u_in_R", "u_in_C", "u_in_cumsum", "u_in_volume", "u_in_cummax", "u_in_delta",
        *[f"u_in_lag{i}" for i in (1, 2, 3, 5, 10)],
        *[f"u_in_lag{i}" for i in (15, 20)],
        *[f"u_in_lead{i}" for i in (1, 2, 3, 5, 10)],
        *[f"u_out_lag{i}" for i in (1, 2, 3, 5, 10)],
        *[f"u_out_lead{i}" for i in (1, 2, 3, 5, 10)],
        *[f"u_in_{stat}{w}" for w in (3, 5, 10, 15, 20) for stat in ("sum", "mean")],
        *[f"u_in_future_{stat}{w}" for w in (3, 5, 10, 20) for stat in ("sum", "mean")],
    ]
    for col in ("u_in", "u_out"):
        for lead in (1, 2, 3, 5, 10):
            frame[f"{col}_lead{lead}"] = grouped[col].shift(-lead).fillna(0).astype("float32")
    result = frame[["_row_order", *feature_cols]].sort_values("_row_order", kind="stable")
    return result[feature_cols].astype("float32", copy=False).reset_index(drop=True)


def scored_mae(y_true: np.ndarray, y_pred: np.ndarray, u_out: np.ndarray) -> float:
    inspiratory = u_out == 0
    return float(np.mean(np.abs(y_true[inspiratory] - y_pred[inspiratory])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="fit all data and write predictions.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv", dtype={"R": "int8", "C": "int8", "u_out": "int8"})
    test = pd.read_csv(DATA / "test.csv", dtype={"R": "int8", "C": "int8", "u_out": "int8"})
    target = train["pressure"].to_numpy(dtype=np.float32)
    train_u_out = train["u_out"].to_numpy(dtype=np.int8)
    all_rows = pd.concat([train, test], ignore_index=True)
    print(f"Building features for {len(all_rows):,} time steps...", flush=True)
    features = make_features(all_rows)
    del all_rows, train
    gc.collect()

    n_train = len(target)
    x_train = features.iloc[:n_train]
    x_test = features.iloc[n_train:]
    del features
    gc.collect()

    params = {
        "objective": "regression_l1",
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": args.seed,
    }

    if args.predict:
        model = lgb.LGBMRegressor(n_estimators=5000, **params)
        scored_idx = np.flatnonzero(train_u_out == 0)
        model.fit(x_train.iloc[scored_idx], target[scored_idx])
        pred = model.predict(x_test, num_threads=-1)
        submission = pd.DataFrame({"id": np.arange(1, len(pred) + 1), "pressure": pred})
        submission.to_csv(args.output, index=False)
        print(f"Wrote {args.output} ({len(submission):,} predictions)")
        return

    metadata = pd.read_csv(DATA / "train.csv", usecols=["breath_id", "u_out"])
    groups = metadata["breath_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.valid_size, random_state=args.seed)
    fit_idx, valid_idx = next(splitter.split(x_train, target, groups))
    fit_idx = fit_idx[train_u_out[fit_idx] == 0]
    valid_idx = valid_idx[train_u_out[valid_idx] == 0]
    fit_set = lgb.Dataset(x_train.iloc[fit_idx], label=target[fit_idx])

    def scored_metric(predictions: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        truth = dataset.get_label()
        mae = float(np.mean(np.abs(truth - predictions)))
        return "inspiratory_mae", mae, False

    valid_set = lgb.Dataset(x_train.iloc[valid_idx], label=target[valid_idx], reference=fit_set)
    callback = lgb.early_stopping(150, first_metric_only=True, verbose=True)
    round_count = 5000
    print(f"Training up to {round_count} LightGBM rounds on {len(fit_idx):,} rows...", flush=True)
    model = lgb.train(
        params,
        fit_set,
        num_boost_round=round_count,
        valid_sets=[valid_set],
        valid_names=["validation"],
        feval=scored_metric,
        callbacks=[callback, lgb.log_evaluation(250)],
    )
    pred = model.predict(x_train.iloc[valid_idx], num_threads=-1)
    mae = scored_mae(target[valid_idx], pred, train_u_out[valid_idx])
    np.save(ROOT / "lgb_val_rows.npy", valid_idx)
    np.save(ROOT / "lgb_val_pred.npy", pred)
    np.save(ROOT / "lgb_test_pred.npy", model.predict(x_test, num_threads=-1))
    print(f"Breath-grouped validation scored MAE (u_out == 0): {mae:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    print("Top features:")
    print(pd.Series(model.feature_importance(importance_type="gain"), index=x_train.columns)
          .sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
