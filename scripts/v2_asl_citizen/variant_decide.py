r"""
v2 step 7b: final variant decision, using a clear rule based on what CORRECT matches score.

Why: variant_auto.py only compared versions with each other, so it marked 30 words "unclear".
This script uses the reference numbers from the exact matches (e.g. BOOK, HELP) instead:

  80 v1 words (opinion B, the v1 model - the stronger opinion):
    - best version (highest top-5, ties -> higher top-1):
        keep if top-5 >= 25th percentile of correct matches, otherwise CHECK BY EYE
    - any other version: keep if top-5 >= median AND top-1 >= median of correct matches,
        otherwise no
  20 new words (only opinion A, nearest neighbours - weaker, so stricter):
    - best version: keep if score >= median of correct matches, otherwise CHECK BY EYE
    - other versions: keep if score >= median AND >= 75% of the best, otherwise no
  near matches (different sign name): handled like versions when the v1 model can judge them
    (e.g. short -> SHORTPERSON / SHORTS / SHORTDISTANCE); for new words (e.g. forget ->
    FORGETFUL) only the weak opinion exists, so they are always CHECK BY EYE
  W.H.A.T: no (fingerspelled)
  exact matches: keep, but CHECK BY EYE if far below normal
    (under half the 10th percentile of correct matches)

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\variant_decide.py > report_outputs\V2_11_variant_decide.txt

Output:
  data\asl_citizen\variant_final.csv   word, gloss, keep (yes / no / check), rule, scores
  -> after the eye check, change each "check" to yes or no in this file.
     This file (not variant_choice.csv) is what the training set will be built from.
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = os.path.join("data", "asl_citizen")
AUTO = os.path.join(BASE, "variant_auto.csv")
OUT = os.path.join(BASE, "variant_final.csv")
ALSO_KEEP = 0.75
NEVER = {"W.H.A.T"}


def main():
    g = pd.read_csv(AUTO)
    if os.path.exists(OUT):
        sys.exit(f"{OUT} already exists (it may hold your eye-check answers). "
                 "Rename or delete it first if you really want to remake it.")
    ex = g[g.match_type == "exact"]
    ex_v1, ex_new = ex[ex.m_top5.notna()], ex[ex.m_top5.isna()]
    ref = {"t5_p10": ex_v1.m_top5.quantile(0.10), "t5_p25": ex_v1.m_top5.quantile(0.25),
           "t5_med": ex_v1.m_top5.median(), "t1_med": ex_v1.m_top1.median(),
           "knn_p10": ex.knn.quantile(0.10), "knn_med": ex.knn.median()}
    print("Reference (scores of CORRECT exact matches):")
    print(f"  v1 model top-5: 10th pct {ref['t5_p10']:.2f} | 25th pct {ref['t5_p25']:.2f} | "
          f"median {ref['t5_med']:.2f};  top-1 median {ref['t1_med']:.2f}")
    print(f"  nearest neighbours: 10th pct {ref['knn_p10']:.2f} | median {ref['knn_med']:.2f}")

    g["keep"], g["rule"] = "", ""
    for w, sub in tqdm(g.groupby("word"), desc="Deciding", unit="word", file=sys.stderr):
        v1 = sub.m_top5.notna().any()
        # exact single match
        if len(sub) == 1 and sub.match_type.iat[0] == "exact":
            i, r = sub.index[0], sub.iloc[0]
            far_low = (r.m_top5 < ref["t5_p10"] / 2) if v1 else (r.knn < ref["knn_p10"] / 2)
            g.loc[i, ["keep", "rule"]] = (["check", "exact match, far below normal"] if far_low
                                          else ["yes", "exact match"])
            continue
        for i in sub.index:
            if g.gloss[i] in NEVER:
                g.loc[i, ["keep", "rule"]] = ["no", "fingerspelled"]
        cand = sub[~sub.gloss.isin(NEVER)]
        if (cand.match_type == "near (check)").all() and not v1:
            for i in cand.index:
                g.loc[i, ["keep", "rule"]] = ["check", "near match (different sign name)"]
            continue
        if v1:
            order = cand.sort_values(["m_top5", "m_top1"], ascending=False)
            best_i = order.index[0]
            b = g.loc[best_i]
            g.loc[best_i, ["keep", "rule"]] = (["yes", "best version, model recognises it"]
                                               if b.m_top5 >= ref["t5_p25"]
                                               else ["check", "best version, but model score low"])
            for i in order.index[1:]:
                r = g.loc[i]
                ok = r.m_top5 >= ref["t5_med"] and r.m_top1 >= ref["t1_med"]
                g.loc[i, ["keep", "rule"]] = (["yes", "also scores like a correct match"] if ok
                                              else ["no", "model score below a correct match"])
        else:
            order = cand.sort_values("knn", ascending=False)
            best_i = order.index[0]
            best = g.knn[best_i]
            g.loc[best_i, ["keep", "rule"]] = (["yes", "best version (nearest neighbours)"]
                                               if best >= ref["knn_med"]
                                               else ["check", "best version, but weak score (new word)"])
            for i in order.index[1:]:
                r = g.loc[i]
                ok = r.knn >= ref["knn_med"] and r.knn >= ALSO_KEEP * best
                g.loc[i, ["keep", "rule"]] = (["yes", "also close (nearest neighbours)"] if ok
                                              else ["no", "lower than best (nearest neighbours)"])

    cols = ["word", "gloss", "match_type", "clips", "knn", "m_top1", "m_top5", "keep", "rule"]
    g[cols].to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    multi = g[g.match_type != "exact"]
    print("\n========== WORDS WITH SEVERAL VERSIONS ==========")
    print(multi[cols[:2] + cols[3:]].to_string(index=False))
    check = g[g.keep == "check"]
    print("\n========== CHECK BY EYE ==========")
    print(check[cols[:2] + cols[3:]].to_string(index=False))
    print("\n========== SUMMARY ==========")
    kept = g[g.keep == "yes"]
    print(f"Glosses: yes {len(kept)} | no {(g.keep == 'no').sum()} | check {len(check)}")
    print(f"Words with 2+ versions kept: {(kept.groupby('word').size() > 1).sum()} "
          f"({', '.join(kept.groupby('word').size().loc[lambda s: s > 1].index)})")
    print(f"ASL Citizen clips in kept glosses: {int(kept.clips.sum())} "
          f"(+ up to {int(check.clips.sum())} if all 'check' rows become yes)")
    no_data = sorted(set(g.word) - set(kept.word) - set(check.word))
    print(f"Words with NO ASL Citizen data at the moment: {no_data if no_data else 'none'}")
    print(f"\nSaved {OUT}  -> change each 'check' to yes or no after looking at the images")


if __name__ == "__main__":
    main()