"""
UW-Madison GI Tract segmentation: everything in ONE file.
Run from the repo root (the folder containing data/):
  python gi_segmentation.py diagnose   1) check RLE ordering + labels -> writes diag_overlay.png
  python gi_segmentation.py train      2) train, tune per-class post-processing, write pred_overlay.png
  python gi_segmentation.py predict    3) predict test set -> submission.csv
  python gi_segmentation.py            train + predict
Workflow: diagnose -> set ORDER -> overfit test (OVERFIT_N) -> real run -> submit.
"""
import os, re, sys, glob, json, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from contextlib import nullcontext
from scipy.spatial.distance import directed_hausdorff
import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------------------ CONFIG ------------------------------
ROOT = "data/uw-madison-gi-tract-image-segmentation"   # folder with train/ test/ and the csvs
ORDER = "row"          # "row" or "col": set after looking at diag_overlay.png
OUT = "submission.csv"
ENCODER, ENCODER_WEIGHTS = "resnet34", "imagenet"      # stronger: "efficientnet-b4"
IMG_H, IMG_W = 320, 384
STRIDE = 2             # 2.5D: slices i-STRIDE, i, i+STRIDE
EPOCHS, BS, LR = 8, 16, 3e-4
FOLD, N_FOLDS = 0, 5
SUBSET_CASES = 20      # None = all 76 cases (needs a real GPU)
OVERFIT_N = 0          # >0: overfit test, e.g. 32 with EPOCHS=30. Expect val dice > 0.8
VAL_MAX = 600          # slices scored each epoch
TUNE_DAYS = 6          # validation case-days used for per-class tuning
NUM_WORKERS = 2
CLASSES = ["large_bowel", "small_bowel", "stomach"]
DICE_EMPTY_BOTH = 1.0  # description says 0 when both empty; grader behaviour unconfirmed
THR_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]
AREA_GRID = [0, 30, 100, 300]     # min blob area, in model-resolution pixels
DEFAULT_PP = {c: {"thr": 0.5, "min_area": 50} for c in CLASSES}
PP_FILE = "postproc.json"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# --------------------------------------------------------------------
AMP = DEVICE == "cuda"
def amp():
    return torch.autocast("cuda") if AMP else nullcontext()



# ---------------------------- RLE helpers ----------------------------
def rle_decode(rle, shape, order=None):
    """shape = (h, w). order 'row' = C order, 'col' = Fortran order (top-to-bottom first)."""
    order = order or ORDER
    m = np.zeros(shape[0] * shape[1], np.uint8)
    if isinstance(rle, str) and rle:
        s = np.array(rle.split(), int)
        for st, ln in zip(s[0::2] - 1, s[1::2]):
            m[st:st + ln] = 1
    return m.reshape(shape, order="C" if order == "row" else "F")


def rle_encode(mask, order=None):
    order = order or ORDER
    flat = mask.flatten(order="C" if order == "row" else "F")
    p = np.concatenate([[0], flat, [0]])
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


# ---------------------------- post-processing ------------------------
def remove_small(m, min_area):
    if min_area <= 0 or not m.any():
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            m[lab == k] = 0
    return m


def postprocess(prob, size, thr, min_area):
    """prob: H x W (model res) -> binary mask at original (w, h). min_area is in model-res pixels."""
    w, h = size
    prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = (prob > thr).astype(np.uint8)
    return remove_small(m, min_area * (w * h) / (IMG_W * IMG_H))


# ------------------------- local metric (approximate) -----------------
def dice_slices(pred, gt):
    """pred, gt: N x H x W bool -> mean per-slice Dice."""
    inter = (pred & gt).sum((1, 2)); den = pred.sum((1, 2)) + gt.sum((1, 2))
    d = np.where(den == 0, DICE_EMPTY_BOTH, 2 * inter / np.maximum(den, 1))
    return float(d.mean())


def _surface_pts(vol, cap=20000):
    H, W = vol.shape[1:]
    k = np.ones((3, 3), np.uint8)
    nz = [z for z in range(len(vol)) if vol[z].any()]
    pts = []
    for z in nz:
        m = vol[z].astype(np.uint8)
        edge = m if z in (nz[0], nz[-1]) else m & (1 - cv2.erode(m, k))
        ys, xs = np.nonzero(edge)
        pts.append(np.stack([xs / W, ys / H, np.full(len(xs), z / max(H, W))], 1))
    pts = np.concatenate(pts)
    if len(pts) > cap:
        pts = pts[np.random.RandomState(0).choice(len(pts), cap, replace=False)]
    return pts


def hd_score(pred_vol, gt_vol):
    """APPROXIMATION of the competition's 3D Hausdorff term: 1 - normalised HD in [0,1]."""
    pe, ge = pred_vol.any(), gt_vol.any()
    if not pe and not ge:
        return 1.0
    if pe != ge:
        return 0.0
    a, b = _surface_pts(pred_vol), _surface_pts(gt_vol)
    hd = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
    return float(1 - min(hd / np.sqrt(3), 1))


def score_class(pred, gt, groups, use_hd=True):
    d = dice_slices(pred, gt)
    if use_hd:
        hs = [hd_score(pred[groups == g], gt[groups == g]) for g in np.unique(groups)]
        h = float(np.mean(hs)); s = 0.4 * d + 0.6 * h
    else:
        h, s = float("nan"), d
    return dict(dice=d, hd=h, score=s, pred_frac=float((pred.any((1, 2))).mean()),
                gt_frac=float((gt.any((1, 2))).mean()))


def tune(probs, gts, groups, use_hd=True):
    """probs: N x 3 x H x W float; gts: N x 3 x H x W uint8; groups: N ints (case-day id).
    Picks threshold and min-area separately per class. Returns (pp dict, report dict)."""
    pp, report = {}, {}
    for c, cls in enumerate(CLASSES):
        gt = gts[:, c].astype(bool); best = None
        for thr in THR_GRID:
            base = (probs[:, c].astype(np.float32) > thr).astype(np.uint8)
            for ma in AREA_GRID:
                pred = (np.stack([remove_small(b.copy(), ma) for b in base]) if ma > 0 else base).astype(bool)
                s = score_class(pred, gt, groups, use_hd)
                if best is None or s["score"] > best[0]["score"]:
                    best = (s, thr, ma)
        pp[cls] = {"thr": float(best[1]), "min_area": int(best[2])}; report[cls] = best[0]
    return pp, report


# ------------------------------- overlays ----------------------------
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]     # BGR: large=red, small=green, stomach=blue

def overlay(img, masks, alpha=0.5):
    """img float HxW in [0,1]; masks 3xHxW -> uint8 BGR image."""
    base = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    for c in range(3):
        m = masks[c].astype(bool)
        base[m] = (1 - alpha) * base[m] + alpha * np.array(COLORS[c], np.float32)
    return base.astype(np.uint8)


# ============================ torch part ============================
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
        y = np.stack([cv2.resize(rle_decode(self.rles.get((r["id"], c)), (int(r["h"]), int(r["w"]))),
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
    for m in models:
        m.eval()
    for b in loader:
        x = (b[0] if isinstance(b, (list, tuple)) else b).to(DEVICE)
        acc = 0
        for m in models:
            with amp():
                acc = acc + torch.sigmoid(m(x)).float() + torch.sigmoid(m(x.flip(3))).float().flip(3)
        yield (acc / (2 * len(models))).cpu().numpy()


def load_pp():
    try:
        return json.load(open(PP_FILE))
    except Exception:
        return DEFAULT_PP


def gt_lowres(r, rles):
    return np.stack([cv2.resize(rle_decode(rles.get((r["id"], c)), (int(r["h"]), int(r["w"]))),
                                (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) for c in CLASSES], 0)


# ------------------- tuning + visual check on validation ---------------
def tune_and_report(model, idx, rles, vai):
    sub = idx.iloc[vai]
    days = sub[["case", "day"]].drop_duplicates().head(TUNE_DAYS)
    keep = sub.set_index(["case", "day"]).index.isin(list(map(tuple, days.values)))
    sel = np.asarray(vai)[keep]
    cd = (idx.iloc[sel]["case"] + idx.iloc[sel]["day"]).values
    groups = pd.factorize(cd)[0]
    probs = np.concatenate([p.astype(np.float16) for p in
                            predict_batches([model], DataLoader(Subset(DS(idx), sel), BS, num_workers=NUM_WORKERS))])
    gts = np.stack([gt_lowres(idx.iloc[p], rles) for p in sel])
    print(f"\nTuning per-class threshold / min-area on {len(sel)} val slices ({len(days)} case-days)...")
    pp, rep = tune(probs, gts, groups, use_hd=True)
    print("class         thr  min_area   dice    hd*   score*   pred_nonempty  gt_nonempty")
    for c in CLASSES:
        s = rep[c]
        print(f"{c:12s} {pp[c]['thr']:.2f} {pp[c]['min_area']:8d}  {s['dice']:.3f}  {s['hd']:.3f}  {s['score']:.3f}"
              f"      {s['pred_frac']:.2f}          {s['gt_frac']:.2f}")
    print(f"mean approx score* = {np.mean([rep[c]['score'] for c in CLASSES]):.4f}   (*hd is an approximation)")
    print("If pred_nonempty is far below gt_nonempty, the model is predicting (almost) nothing.")
    json.dump(pp, open(PP_FILE, "w"), indent=1)
    # visual check: raw | ground truth | prediction
    pos = [k for k in range(len(sel)) if gts[k].any()]
    pos = pos[::max(1, len(pos) // 6)][:6]
    rows = []
    for k in pos:
        img = DS(idx)[int(sel[k])][1].numpy()
        pm = np.stack([remove_small((probs[k, c].astype(np.float32) > pp[CLASSES[c]]["thr"]).astype(np.uint8),
                                    pp[CLASSES[c]]["min_area"]) for c in range(3)])
        rows.append(np.hstack([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                               overlay(img, gts[k]), overlay(img, pm)]))
    if rows:
        cv2.imwrite("pred_overlay.png", np.vstack(rows))
        print("Wrote pred_overlay.png  (raw | ground truth | prediction)")


# --------------------------- train + validate ------------------------
def main_train():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    idx = build_index("train")
    idx = idx[idx["id"].isin(set(tr["id"]))]
    if SUBSET_CASES:
        idx = idx[idx["case"].isin(sorted(idx["case"].unique())[:SUBSET_CASES])]
    idx = attach_neighbors(idx)
    print(f"{len(idx)} slices, {idx['case'].nunique()} cases, device={DEVICE}, ORDER={ORDER}")
    rles = {(i, c): s for i, c, s in zip(tr["id"], tr["class"], tr["segmentation"]) if isinstance(s, str)}
    has = np.array([any((i, c) in rles for c in CLASSES) for i in idx["id"]])

    if OVERFIT_N:
        rng = np.random.RandomState(0)
        tri = rng.permutation(np.where(has)[0])[:OVERFIT_N]; vai = tri
        print(f"OVERFIT TEST on {len(tri)} slices (no augmentation). Dice should climb above 0.8.")
    else:
        n_splits = min(N_FOLDS, idx["case"].nunique())
        folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
        tri, vai = next(f for n, f in enumerate(folds.split(idx, has, idx["case"])) if n == FOLD % n_splits)

    pin = DEVICE == "cuda"
    ltr = DataLoader(Subset(DS(idx, rles, train=not OVERFIT_N), tri), min(BS, len(tri)), shuffle=True,
                     num_workers=NUM_WORKERS, drop_last=not OVERFIT_N, pin_memory=pin)
    vsel = np.asarray(vai)[::max(1, len(vai) // VAL_MAX)]
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(ltr))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
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
        sc, pos = [], 0
        for probs in predict_batches([model], DataLoader(Subset(DS(idx), vsel), BS, num_workers=NUM_WORKERS)):
            for n in range(len(probs)):
                r = idx.iloc[vsel[pos]]; pos += 1
                for c, cls in enumerate(CLASSES):
                    pm = postprocess(probs[n, c], (int(r["w"]), int(r["h"])), **DEFAULT_PP[cls]).astype(bool)
                    gt = rle_decode(rles.get((r["id"], cls)), (int(r["h"]), int(r["w"]))).astype(bool)
                    den = pm.sum() + gt.sum()
                    sc.append(DICE_EMPTY_BOTH if den == 0 else 2 * (pm & gt).sum() / den)
        d = float(np.mean(sc)); print(f"epoch {ep} val dice {d:.4f}", flush=True)
        if d > best:
            best = d; torch.save(model.state_dict(), f"best_fold{FOLD}.pt")
    print("best val dice", best)
    model.load_state_dict(torch.load(f"best_fold{FOLD}.pt", map_location=DEVICE))
    tune_and_report(model, idx, rles, vai)


# ------------------------------ inference ----------------------------
def main_predict(weights=None):
    weights = weights or (f"best_fold{FOLD}.pt",)
    pp = load_pp()
    print("post-processing:", pp)
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    idx = build_index("test")
    idx = attach_neighbors(idx[idx["id"].isin(set(test["id"]))])
    models = []
    for w in weights:
        m = make_model(); m.load_state_dict(torch.load(w, map_location=DEVICE)); models.append(m)
    rows, pos = {}, 0
    for probs in predict_batches(models, DataLoader(DS(idx), BS, num_workers=NUM_WORKERS)):
        for n in range(len(probs)):
            r = idx.iloc[pos]; pos += 1
            for c, cls in enumerate(CLASSES):
                rows[(r["id"], cls)] = rle_encode(postprocess(probs[n, c], (int(r["w"]), int(r["h"])), **pp[cls]))
        if pos % (BS * 20) == 0:
            print(f"predicted {pos}/{len(idx)}", flush=True)
    sub = test[["id", "class"]].copy()
    sub["predicted"] = [rows.get((i, c), "") for i, c in zip(sub["id"], sub["class"])]
    sub.to_csv(OUT, index=False)
    print(sub.shape, "->", OUT)
    print("non-empty fraction per class in your predictions (train data: large 0.37, small 0.29, stomach 0.22):")
    print((sub.assign(ne=sub["predicted"] != "").groupby("class")["ne"].mean()).round(3).to_string())


# ============================ diagnose ============================
def main_diagnose():

    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    tr["px"] = tr["segmentation"].fillna("").apply(lambda s: sum(map(int, s.split()[1::2])) if s else 0)
    print("per-class stats on train.csv (all slices):")
    print(tr.groupby("class").agg(nonempty_frac=("px", lambda x: (x > 0).mean()),
                                  mean_px_when_present=("px", lambda x: x[x > 0].mean())))

    idx = build_index("train").set_index("id")
    rl = {(i, c): s for i, c, s in zip(tr["id"], tr["class"], tr["segmentation"]) if isinstance(s, str)}
    full = [i for i in idx.index if all((i, c) in rl for c in CLASSES)]
    empty = [i for i in idx.index if not any((i, c) in rl for c in CLASSES)]
    rng = np.random.RandomState(1)
    pick = list(rng.choice(full, 6, replace=False)) + list(rng.choice(empty, 2, replace=False))

    rows = []
    for i in pick:
        r = idx.loc[i]; h, w = int(r["h"]), int(r["w"])
        img = load_img(r["path"])
        panels = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
        for order in ("row", "col"):
            m = np.stack([rle_decode(rl.get((i, c)), (h, w), order) for c in CLASSES])
            panels.append(overlay(img, m))
        panels = [cv2.resize(p, (360, 360)) for p in panels]
        cv2.putText(panels[0], f"{i} ({w}x{h})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(np.hstack(panels))
    cv2.imwrite("diag_overlay.png", np.vstack(rows))
    print("\nWrote diag_overlay.png  (columns: raw | row-major | col-major;  red=large, green=small, blue=stomach)")
    print("Last two rows are slices with NO labels: the overlays there should be blank.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "diagnose":
        main_diagnose()
    else:
        if mode in ("all", "train"):
            main_train()
        if mode in ("all", "predict"):
            main_predict()