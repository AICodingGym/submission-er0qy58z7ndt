"""
UW-Madison GI Tract segmentation: 2.5D U-Net, case-grouped validation, RLE submission.
Run from the data folder:  python gi_segmentation.py
"""
import os, re, glob, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from contextlib import nullcontext
import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------------------ CONFIG ------------------------------
ROOT = "data/uw-madison-gi-tract-image-segmentation"                                # folder containing train/ test/ and the csvs
OUT = "submission.csv"
ENCODER, ENCODER_WEIGHTS = "resnet34", "imagenet"   # full run: "efficientnet-b4"
IMG_H, IMG_W = 320, 384
STRIDE = 2                                   # 2.5D: slices i-STRIDE, i, i+STRIDE
EPOCHS, BS, LR = 1, 16, 3e-4                 # full run: EPOCHS = 25
FOLD, N_FOLDS = 0, 5
THRESH = 0.5
MIN_AREA = 50                                # drop predicted blobs smaller than this (pixels)
SUBSET_CASES = 5                             # smoke test on 5 train cases; set None for full data
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ["large_bowel", "small_bowel", "stomach"]
# --------------------------------------------------------------------
AMP = DEVICE == "cuda"
def amp():
    return torch.autocast("cuda") if AMP else nullcontext()


# ---------------------------- RLE helpers ----------------------------
def rle_decode(rle, shape):
    m = np.zeros(shape[0] * shape[1], np.uint8)
    if isinstance(rle, str) and rle:
        s = np.array(rle.split(), int)
        for st, ln in zip(s[0::2] - 1, s[1::2]):
            m[st:st + ln] = 1
    return m.reshape(shape)           # row-major, 1-indexed (Kaggle convention)


def rle_encode(mask):
    p = np.concatenate([[0], mask.flatten(), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(map(str, r))


# ---------------------------- data index -----------------------------
def build_index(split):
    paths = glob.glob(os.path.join(ROOT, split, "*", "*", "scans", "*.png"))
    rows = []
    for p in paths:
        case_day = p.split(os.sep)[-3]                       # e.g. case123_day20
        sl, w, h = re.search(r"slice_(\d+)_(\d+)_(\d+)_", os.path.basename(p)).groups()
        case, day = case_day.split("_")
        rows.append((f"{case_day}_slice_{sl}", case, day, int(sl), int(w), int(h), p))
    df = pd.DataFrame(rows, columns=["id", "case", "day", "slice", "w", "h", "path"])
    return df.sort_values(["case", "day", "slice"]).reset_index(drop=True)


def attach_neighbors(df):
    """Row positions of the slice STRIDE before/after within the same case/day (clamped)."""
    df = df.reset_index(drop=True).copy()
    prev = np.zeros(len(df), int); nxt = np.zeros(len(df), int)
    for _, g in df.groupby(["case", "day"], sort=False):
        ix = g.index.values; n = len(ix)
        for k in range(n):
            prev[ix[k]] = ix[max(k - STRIDE, 0)]
            nxt[ix[k]] = ix[min(k + STRIDE, n - 1)]
    df["prev"], df["next"] = prev, nxt
    return df


def load_img(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.clip(im / (np.percentile(im, 99.5) + 1e-6), 0, 1)


class DS(Dataset):
    def __init__(self, df, rles=None, train=False):
        self.df, self.rles, self.train = df, rles, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ims = [load_img(self.df.iloc[j]["path"]) for j in (r["prev"], i, r["next"])]
        x = np.stack([cv2.resize(a, (IMG_W, IMG_H)) for a in ims], 0)
        if self.rles is None:
            return torch.from_numpy(x)
        y = np.stack([cv2.resize(rle_decode(self.rles.get((r["id"], c)), (r["h"], r["w"])),
                                 (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) for c in CLASSES], 0)
        if self.train:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), np.random.uniform(-10, 10),
                                        np.random.uniform(0.9, 1.1))
            x = np.stack([cv2.warpAffine(np.ascontiguousarray(c), M, (IMG_W, IMG_H)) for c in x], 0)
            y = np.stack([cv2.warpAffine(np.ascontiguousarray(c), M, (IMG_W, IMG_H),
                                         flags=cv2.INTER_NEAREST) for c in y], 0)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).float()


# ------------------------------ model --------------------------------
def make_model():
    return smp.Unet(ENCODER, encoder_weights=ENCODER_WEIGHTS, in_channels=3,
                    classes=3, activation=None).to(DEVICE)


def dice_loss(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((2, 3)) + y.sum((2, 3)) + eps)).mean()


def criterion(logits, y):
    return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, y) + 0.5 * dice_loss(logits, y)


@torch.no_grad()
def predict_batches(models, loader):
    """Yield averaged probabilities (B x 3 x H x W, numpy) one batch at a time."""
    for m in models:
        m.eval()
    for b in loader:
        x = (b[0] if isinstance(b, (list, tuple)) else b).to(DEVICE)
        acc = 0
        for m in models:
            with amp():
                acc = acc + torch.sigmoid(m(x)).float() + torch.sigmoid(m(x.flip(3))).float().flip(3)
        yield (acc / (2 * len(models))).cpu().numpy()


def postprocess(prob, size):
    """prob: H x W for one class -> binary mask at original (w, h)."""
    prob = cv2.resize(prob, size, interpolation=cv2.INTER_LINEAR)
    m = (prob > THRESH).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
            m[lab == k] = 0
    return m


def dice_score(a, b):
    if a.sum() == 0 and b.sum() == 0:
        return 1.0   # NOTE: competition text says 0 when both empty; verify which the grader uses
    return 2 * (a & b).sum() / (a.sum() + b.sum())


# --------------------------- train + validate ------------------------
def main_train():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    idx = build_index("train")
    idx = idx[idx["id"].isin(set(tr["id"]))]
    if SUBSET_CASES:
        keep = sorted(idx["case"].unique())[:SUBSET_CASES]
        idx = idx[idx["case"].isin(keep)]
    idx = attach_neighbors(idx)
    print(f"{len(idx)} slices, {idx['case'].nunique()} cases, device={DEVICE}")

    rles = {(i, c): s for i, c, s in zip(tr["id"], tr["class"], tr["segmentation"])
            if isinstance(s, str)}
    # sanity: RLE round-trip on one real mask
    k = next(iter(rles)); r0 = idx[idx["id"] == k[0]]
    if len(r0):
        shape = (int(r0.iloc[0]["h"]), int(r0.iloc[0]["w"])); m = rle_decode(rles[k], shape)
        assert rle_decode(rle_encode(m), shape).sum() == m.sum()

    has = np.array([any((i, c) in rles for c in CLASSES) for i in idx["id"]])
    n_splits = min(N_FOLDS, idx["case"].nunique())
    folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    tri, vai = next(f for n, f in enumerate(folds.split(idx, has, idx["case"])) if n == FOLD % n_splits)

    pin = DEVICE == "cuda"
    ltr = DataLoader(Subset(DS(idx, rles, True), tri), BS, shuffle=True, num_workers=NUM_WORKERS,
                     drop_last=True, pin_memory=pin)
    lva = DataLoader(Subset(DS(idx, rles, False), vai), BS, num_workers=NUM_WORKERS, pin_memory=pin)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(ltr))
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    best = -1
    for ep in range(EPOCHS):
        model.train()
        for step, (x, y) in enumerate(ltr):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with amp():
                loss = criterion(model(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if step % 20 == 0:
                print(f"epoch {ep} step {step}/{len(ltr)} loss {loss.item():.4f}", flush=True)
        # validation (streamed, per batch)
        sc, pos = [], 0
        for probs in predict_batches([model], DataLoader(Subset(DS(idx), vai), BS, num_workers=NUM_WORKERS)):
            for n in range(len(probs)):
                r = idx.iloc[vai[pos]]; pos += 1
                for c, cls in enumerate(CLASSES):
                    pm = postprocess(probs[n, c], (int(r["w"]), int(r["h"]))).astype(bool)
                    gt = rle_decode(rles.get((r["id"], cls)), (int(r["h"]), int(r["w"]))).astype(bool)
                    sc.append(dice_score(pm, gt))
        d = float(np.mean(sc)); print(f"epoch {ep} val dice {d:.4f}", flush=True)
        if d > best:
            best = d; torch.save(model.state_dict(), f"best_fold{FOLD}.pt")
    print("best val dice", best)


# ------------------------------ inference ----------------------------
def main_predict(weights=None):
    weights = weights or (f"best_fold{FOLD}.pt",)
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    idx = build_index("test")
    idx = idx[idx["id"].isin(set(test["id"]))]
    idx = attach_neighbors(idx)
    models = []
    for w in weights:
        m = make_model(); m.load_state_dict(torch.load(w, map_location=DEVICE)); models.append(m)
    loader = DataLoader(DS(idx), BS, num_workers=NUM_WORKERS)
    rows, pos = {}, 0
    for probs in predict_batches(models, loader):
        for n in range(len(probs)):
            r = idx.iloc[pos]; pos += 1
            for c, cls in enumerate(CLASSES):
                rows[(r["id"], cls)] = rle_encode(postprocess(probs[n, c], (int(r["w"]), int(r["h"]))))
        if pos % (BS * 20) == 0:
            print(f"predicted {pos}/{len(idx)}", flush=True)
    sub = test[["id", "class"]].copy()
    sub["predicted"] = [rows.get((i, c), "") for i, c in zip(sub["id"], sub["class"])]
    sub.to_csv(OUT, index=False)
    print(sub.shape, f"{(sub['predicted'] != '').mean():.3f} non-empty ->", OUT)


if __name__ == "__main__":
    main_train()
    main_predict()