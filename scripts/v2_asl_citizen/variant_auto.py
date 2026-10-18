r"""
v2 step 7: AUTOMATIC variant check - which ASL Citizen gloss is the same sign as WLASL?

Two independent opinions, both comparing ONLY with WLASL TRAINING clips
(validation and test clips are never used for this decision):

  A. Nearest neighbours (all 100 words):
     Each ASL Citizen clip is compared with every WLASL training clip of all 100 words,
     using the HAND part of the model's E7 features (wrist position, handshape measured from
     the wrist, and their movement). Body/arm points are left out here: they barely differ
     between signs and would only add noise. We look at its 5 most similar WLASL clips and count how many are the
     clip's own word. A gloss that is the same sign as WLASL scores clearly higher.
  B. The v1 model (the 80 v1 words only):
     The 3 v1 E7 models (runs\tuning_v4) look at each ASL Citizen clip. We count how often
     the right word is their first guess (top-1) and in their first five (top-5).
     The v1 models never saw ASL Citizen and only learned from WLASL training clips.

For reference, "exact" single matches (e.g. BOOK, HELP) show what scores a CORRECT gloss
usually gets.

Decision per word with several candidate glosses (variants, near matches):
  - keep the best gloss if it scores like a correct match;
  - another gloss is also kept if it scores almost as well (WLASL sometimes mixes versions),
    but marked "check by eye";
  - if the two opinions disagree, or nothing scores like a correct match -> "unclear".
  W.H.A.T is fingerspelled -> always no.

Uses data\processed_v2 (preprocess_v2.py) and data\split_wlasl100.csv (split_new20.py).

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\variant_auto.py > report_outputs\V2_10_variant_auto.txt

Outputs:
  data\asl_citizen\variant_auto.csv     scores and suggestion per word + gloss
  data\asl_citizen\variant_choice.csv   blank 'keep' cells filled with the suggestion
                                        (your own answers are never changed)
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week3_training"))
from tune_v4 import features  # noqa: E402  (same E7 features as the model)

P2 = os.path.join("data", "processed_v2")
SPLIT = os.path.join("data", "split_wlasl100.csv")
V1_LABELS = os.path.join("data", "processed_v1", "labels.csv")
V1_RUNS = [os.path.join("runs", "tuning_v4", f"E7_seed{s}") for s in (42, 7, 123)]
BASE = os.path.join("data", "asl_citizen")
CHOICE = os.path.join(BASE, "variant_choice.csv")
OUT = os.path.join(BASE, "variant_auto.csv")
E7_FEATS = {"drop_pose_z": True, "hand_relative": True, "velocity": True}
K = 5
# E7 feature layout (346): pose 46 | left wrist 3 + shape 60 | right wrist 3 + shape 60 | movement of all 172 | 2 flags
HAND_COLS = np.r_[46:172, 172 + 46:344]
ALSO_KEEP = 0.75        # a second gloss is also kept if it scores >= 75% of the best one
NEVER = {"W.H.A.T": "fingerspelled"}


def knn_votes(Xq, Xref, yref, own, k=K, batch=64):
    """For each query clip: share of its k nearest WLASL training clips that are its own word,
    out of what is possible (a word with only 3 training clips can get at most 3 of 5 votes,
    so it is scored out of 3). Distance = mean over the 64 frames of the feature distance."""
    n_own = pd.Series(yref).value_counts()
    possible = np.minimum(k, pd.Series(own).map(n_own).fillna(0).to_numpy())
    possible = np.where(possible > 0, possible, 1)
    out = np.zeros(len(Xq))
    nearest_word = np.zeros(len(Xq), dtype=int)
    Xref_flat = Xref.reshape(len(Xref), -1)
    T = Xref.shape[1]
    ref_sq = (Xref ** 2).sum(-1)                                     # (R, T)
    for i in tqdm(range(0, len(Xq), batch), desc="A: nearest neighbours", unit="batch", file=sys.stderr):
        q = Xq[i:i + batch]                                          # (B, T, F)
        # squared distance per frame: |q|^2 + |r|^2 - 2 q.r, then sqrt, then mean over frames
        cross = np.einsum("btf,rtf->brt", q, Xref, optimize=True)
        d2 = (q ** 2).sum(-1)[:, None, :] + ref_sq[None] - 2 * cross
        d = np.sqrt(np.clip(d2, 0, None)).mean(-1)                   # (B, R)
        nn_idx = np.argsort(d, axis=1)[:, :k]
        nn_y = yref[nn_idx]
        out[i:i + batch] = (nn_y == own[i:i + batch, None]).sum(1) / possible[i:i + batch]
        nearest_word[i:i + batch] = nn_y[:, 0]
    return out, nearest_word


def model_probs(X_q):
    """Average softmax of the 3 v1 E7 models (same code as the v1 ensemble: tune_v4.val_probs)."""
    from tune_v4 import val_probs
    n_cls = len(pd.read_csv(V1_LABELS))
    probs = []
    for d in tqdm(V1_RUNS, desc="B: v1 models", unit="model", file=sys.stderr):
        probs.append(val_probs(d, X_q, np.ones(len(X_q), bool), n_cls))
    return np.mean(probs, axis=0)


def main():
    X = np.load(os.path.join(P2, "X.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    split = pd.read_csv(SPLIT, dtype={"video_id": str})
    split_of = dict(zip(split["video_id"], split["split"]))

    wl = meta["source"].isin(["wlasl_v1", "wlasl_new20"])
    meta["wl_split"] = np.where(wl, meta["clip_id"].map(split_of), "")
    ref = (wl & (meta["wl_split"] == "train")).to_numpy()
    asc = ((meta["source"] == "asl_citizen") & ~(meta["longest_gap"] > 10)).to_numpy()
    print(f"WLASL training clips used as reference: {ref.sum()} (validation/test NOT used)")
    print(f"ASL Citizen clips checked: {asc.sum()} (usable, no long gaps)")
    missing_split = (wl & meta["wl_split"].isna()).sum()
    if missing_split:
        print(f"WARNING: {missing_split} WLASL clips have no split entry")

    # ---- A. nearest neighbours on E7 features -----------------------------------
    # No standardising here: the hand features already have natural units (wrist in shoulder
    # widths, handshape in hand sizes), and standardising would blow up tiny noisy columns.
    F = features(X, E7_FEATS).astype(np.float32)
    y = meta["label"].to_numpy()
    Fh = np.ascontiguousarray(F[..., HAND_COLS])
    knn, nearest = knn_votes(Fh[asc], Fh[ref], y[ref], y[asc])
    res = meta.loc[asc, ["clip_id", "word", "gloss", "signer"]].copy()
    res["knn"] = knn
    res["nearest_word"] = labels.set_index("label").loc[nearest, "word"].to_numpy()

    # ---- B. v1 model (80 v1 words) -----------------------------------------------
    v1_words = pd.read_csv(V1_LABELS)
    v1_label = dict(zip(v1_words["word"], v1_words["label"]))
    in_v1 = res["word"].isin(v1_label).to_numpy()
    res["m_top1"], res["m_top5"] = np.nan, np.nan
    try:
        P = model_probs(X[asc][in_v1])
        own = res.loc[in_v1, "word"].map(v1_label).to_numpy()
        top5 = np.argsort(-P, axis=1)[:, :5]
        res.loc[in_v1, "m_top1"] = (top5[:, 0] == own).astype(float)
        res.loc[in_v1, "m_top5"] = (top5 == own[:, None]).any(1).astype(float)
        have_model = True
    except Exception as e:
        print(f"\nWARNING: v1 model check skipped ({type(e).__name__}: {e})")
        have_model = False

    # ---- per gloss ---------------------------------------------------------------
    g = res.groupby(["word", "gloss"]).agg(clips=("clip_id", "count"), knn=("knn", "mean"),
                                           m_top1=("m_top1", "mean"), m_top5=("m_top5", "mean")).reset_index()
    choice = pd.read_csv(CHOICE, dtype=str).fillna("")
    mtype = dict(zip(zip(choice.word, choice.gloss), choice.match_type))
    g["match_type"] = [mtype.get((w, gl), "") for w, gl in zip(g.word, g.gloss)]
    exact = g[g.match_type == "exact"]
    ref_knn = exact.knn.quantile([0.1, 0.25, 0.5])
    ref_m5 = exact.m_top5.quantile([0.1, 0.25, 0.5]) if have_model else None
    print("\n========== REFERENCE: CORRECT (exact) MATCHES ==========")
    print(f"A nearest-neighbour score: 10th pct {ref_knn[0.1]:.2f} / 25th {ref_knn[0.25]:.2f} / "
          f"median {ref_knn[0.5]:.2f}   (random guessing would be about {1 / len(labels):.2f})")
    if have_model:
        print(f"B v1 model top-5:          10th pct {ref_m5[0.1]:.2f} / 25th {ref_m5[0.25]:.2f} / "
              f"median {ref_m5[0.5]:.2f}   | top-1 median {exact.m_top1.median():.2f}")

    low_knn = ref_knn[0.1]
    low_m5 = ref_m5[0.1] if have_model else None

    def primary(r):          # the main score: model top-5 for v1 words, nearest neighbours otherwise
        return r.m_top5 if (have_model and not np.isnan(r.m_top5)) else r.knn

    def low_bar(r):
        return low_m5 if (have_model and not np.isnan(r.m_top5)) else low_knn

    g["primary"] = g.apply(primary, axis=1)
    g["suggest"], g["reason"] = "", ""
    for w, sub in g.groupby("word"):
        if len(sub) == 1 and sub.match_type.iat[0] == "exact":
            i = sub.index[0]
            low = sub.primary.iat[0] < low_bar(sub.iloc[0])
            g.loc[i, ["suggest", "reason"]] = ["yes", "exact match, LOW score - check by eye" if low
                                               else "exact match, scores normally"]
            continue
        cand = sub[~sub.gloss.isin(NEVER)]
        for i in sub.index.difference(cand.index):
            g.loc[i, ["suggest", "reason"]] = ["no", NEVER[g.gloss[i]]]
        if cand.empty:
            continue
        best_i = cand.primary.idxmax()
        best = cand.primary.max()
        knn_best = cand.knn.idxmax()
        agree = (not have_model) or np.isnan(cand.m_top5).all() or knn_best == best_i
        ties = (cand.primary == best).sum() > 1
        good = best > 0 and not ties and best >= low_bar(cand.loc[best_i])
        for i in cand.index:
            r = g.loc[i]
            if ties or best <= 0:
                g.loc[i, ["suggest", "reason"]] = ["unclear", "no gloss scores better than the others"]
            elif i == best_i:
                if good and agree:
                    g.loc[i, ["suggest", "reason"]] = ["yes", "best match"]
                elif not good:
                    g.loc[i, ["suggest", "reason"]] = ["unclear", "best, but scores below correct matches"]
                else:
                    g.loc[i, ["suggest", "reason"]] = ["unclear", f"opinions disagree (A prefers {g.gloss[knn_best]})"]
            elif (r.primary >= ALSO_KEEP * best and r.primary >= low_bar(r)) or \
                    (r.knn >= ALSO_KEEP * cand.knn.max() and r.knn >= low_knn):
                g.loc[i, ["suggest", "reason"]] = ["unclear", "almost as good as best - both may be right"]
            else:
                g.loc[i, ["suggest", "reason"]] = ["no", "clearly lower than best"]
    for c in ["knn", "m_top1", "m_top5", "primary"]:
        g[c] = g[c].round(2)
    g.to_csv(OUT, index=False)

    # ---- fill the choice table (never overwrite the user's own answers) -----------
    filled = 0
    for i, r in choice.iterrows():
        m = g[(g.word == r.word) & (g.gloss == r.gloss)]
        if m.empty:
            continue
        s, why = m.suggest.iat[0], m.reason.iat[0]
        if r.match_type == "exact":
            if "LOW" in why and "auto" not in r.note:
                choice.loc[i, "note"] = (r.note + " " if r.note else "") + "auto: low score - check by eye"
            continue
        if r.keep.strip() == "":
            choice.loc[i, "keep"] = s if s in ("yes", "no") else ""
            choice.loc[i, "note"] = f"auto: {why}"
            filled += 1
    choice.to_csv(CHOICE, index=False)

    # ---- report --------------------------------------------------------------------
    multi = g[g.match_type != "exact"]
    pd.set_option("display.width", 200)
    print("\n========== WORDS WITH SEVERAL CANDIDATE GLOSSES ==========")
    cols = ["word", "gloss", "clips", "knn"] + (["m_top1", "m_top5"] if have_model else []) + ["suggest", "reason"]
    print(multi[cols].to_string(index=False))
    low_exact = g[(g.match_type == "exact") & g.reason.str.contains("LOW")]
    print(f"\nExact matches with a LOW score (double meanings? check by eye): {len(low_exact)}")
    if len(low_exact):
        print(low_exact[cols].to_string(index=False))
    print("\n========== SUMMARY ==========")
    print(multi.suggest.value_counts().to_string())
    unclear_words = sorted(multi[multi.suggest == "unclear"].word.unique())
    print(f"Words to check by eye: {len(unclear_words) + len(low_exact)}  "
          f"({', '.join(unclear_words + low_exact.word.tolist())})")
    print(f"variant_choice.csv: {filled} blank rows filled; your own answers unchanged")
    print(f"Saved {OUT} and {CHOICE}")


if __name__ == "__main__":
    main()