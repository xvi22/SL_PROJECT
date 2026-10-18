r"""
train_baseline.py - a first LSTM baseline, trained on TRAIN and checked on VALIDATION only.

A starting point for the LSTM lead, not the final model. The test set is never loaded
into the training loop or evaluated here; it is kept for the final Q1/Q4 evaluation.

Needs PyTorch (CPU is fine). Install once, with the venv active:
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Run from C:\SL_PROJECT after preprocess.py and split.py:
    python scripts\week3_training\train_baseline.py

Outputs (in runs\baseline_<date-time>\):
    model_best.pt     weights with the best validation accuracy
    history.csv       loss and accuracy per epoch
    curves.png        training vs validation curves (shows overfitting)
    summary.txt       settings and results, ready to paste into notes
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------- settings (the LSTM lead tunes these on VALIDATION) ----------------
DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 32
MAX_EPOCHS = 200
PATIENCE = 30        # stop if validation top-1 has not improved for this many epochs
SEED = 42
STANDARDISE = True   # scale every feature to mean 0, std 1 using TRAINING clips only.
                     # Needed because pose depth (z) values reach -8 shoulder widths
                     # while most values are within +-3. The live pipeline must apply
                     # the same numbers (saved as norm_stats.npz).


class SignLSTM(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=LAYERS, batch_first=True,
                            dropout=DROPOUT if LAYERS > 1 else 0.0)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, n_classes)

    def forward(self, x):                 # x: (batch, frames, features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))   # last time step -> word scores


def topk_correct(logits, y, k):
    k = min(k, logits.shape[1])
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).sum().item()


def run_epoch(model, X, y, loss_fn, opt=None):
    """One pass over (X, y). Trains if opt is given. Returns loss, top-1, top-5."""
    training = opt is not None
    model.train(training)
    idx = torch.randperm(len(X)) if training else torch.arange(len(X))
    tot_loss = c1 = c5 = 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(X), BATCH):
            b = idx[i:i + BATCH]
            logits = model(X[b])
            loss = loss_fn(logits, y[b])
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item() * len(b)
            c1 += topk_correct(logits, y[b], 1)
            c5 += topk_correct(logits, y[b], 5)
    n = len(X)
    return tot_loss / n, c1 / n, c5 / n


def time_inference(model, n_features, frames, repeats=200):
    """Average time to classify ONE sequence on this CPU (for Q4)."""
    model.eval()
    x = torch.randn(1, frames, n_features)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_all = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y.npy"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), dtype={"video_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    words = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))["word"].tolist()
    which = meta["video_id"].map(dict(zip(split["video_id"], split["split"])))
    if which.isna().any():
        raise SystemExit("Some clips in meta.csv are missing from split_v1.csv; re-run split.py.")

    tr, va = (which == "train").values, (which == "val").values
    n_test = int((which == "test").sum())
    Xtr_np, Xva_np = X_all[tr], X_all[va]
    del X_all  # the test clips are not used anywhere below
    if STANDARDISE:
        mu = Xtr_np.mean(axis=(0, 1))
        sd = Xtr_np.std(axis=(0, 1)) + 1e-6
        mu[-2:], sd[-2:] = 0.0, 1.0          # leave the two "hand present" flags as 0/1
        Xtr_np, Xva_np = (Xtr_np - mu) / sd, (Xva_np - mu) / sd
    Xtr, ytr = torch.tensor(Xtr_np, dtype=torch.float32), torch.tensor(y_all[tr])
    Xva, yva = torch.tensor(Xva_np, dtype=torch.float32), torch.tensor(y_all[va])

    n_classes, frames, n_feat = len(words), Xtr.shape[1], Xtr.shape[2]
    model = SignLSTM(n_feat, n_classes)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    stamp = time.strftime("%Y%m%d_%H%M")
    out_dir = os.path.join("runs", f"baseline_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    if STANDARDISE:
        np.savez(os.path.join(out_dir, "norm_stats.npz"), mean=mu, std=sd)
    print(f"Train {len(Xtr)} clips | validation {len(Xva)} clips | test {n_test} clips (held out, not used)")
    print(f"Input {frames} frames x {n_feat} features | {n_classes} words | {n_params:,} parameters\n")

    history, best, best_epoch, wait = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        tl, t1, t5 = run_epoch(model, Xtr, ytr, loss_fn, opt)
        vl, v1, v5 = run_epoch(model, Xva, yva, loss_fn)
        history.append({"epoch": epoch, "train_loss": tl, "train_top1": t1, "train_top5": t5,
                        "val_loss": vl, "val_top1": v1, "val_top5": v5})
        if v1 > best:
            best, best_epoch, wait = v1, epoch, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
        else:
            wait += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train loss {tl:.2f} top1 {t1:.1%} | "
                  f"val loss {vl:.2f} top1 {v1:.1%} top5 {v5:.1%}")
        if wait >= PATIENCE:
            print(f"Stopped early: no validation improvement for {PATIENCE} epochs.")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    best_row = hist.loc[hist["epoch"] == best_epoch].iloc[0]

    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt")))
    ms = time_inference(model, n_feat, frames)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="validation")
    ax[0].set_title("Loss")
    ax[1].plot(hist["epoch"], hist["train_top1"], label="train top-1")
    ax[1].plot(hist["epoch"], hist["val_top1"], label="validation top-1")
    ax[1].plot(hist["epoch"], hist["val_top5"], label="validation top-5", linestyle="--")
    ax[1].axvline(best_epoch, color="grey", lw=0.8)
    ax[1].set_title("Accuracy (grey line = best epoch)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves.png"), dpi=90)
    plt.close(fig)

    summary = [
        "========== BASELINE SUMMARY (train/validation only) ==========",
        f"Data: {DATA_DIR}, split {SPLIT_CSV}",
        f"Clips: train {len(Xtr)}, validation {len(Xva)}, test {n_test} (not used)",
        f"Features standardised with training-set mean/std: {STANDARDISE}",
        f"Model: LSTM {LAYERS} layers x {HIDDEN} units, dropout {DROPOUT}, {n_params:,} parameters",
        f"Training: Adam lr {LR}, weight decay {WEIGHT_DECAY}, batch {BATCH}, "
        f"{len(hist)} epochs ({time.time() - t_start:.0f} s)",
        f"Best epoch: {best_epoch}",
        f"  train top-1 {best_row['train_top1']:.1%}",
        f"  validation top-1 {best_row['val_top1']:.1%}, top-5 {best_row['val_top5']:.1%}",
        f"Gap train - validation (top-1): {best_row['train_top1'] - best_row['val_top1']:.1%} "
        "(large gap = overfitting)",
        f"Model processing time: {ms:.1f} ms per sequence on this CPU (for Q4)",
        f"Saved in {out_dir}",
    ]
    text = "\n".join(summary)
    print("\n" + text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n\nSettings: " + json.dumps(
            {"HIDDEN": HIDDEN, "LAYERS": LAYERS, "DROPOUT": DROPOUT, "LR": LR,
             "WEIGHT_DECAY": WEIGHT_DECAY, "BATCH": BATCH, "MAX_EPOCHS": MAX_EPOCHS,
             "PATIENCE": PATIENCE, "SEED": SEED,
             "STANDARDISE": STANDARDISE}))


if __name__ == "__main__":
    main()