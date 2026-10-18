r"""
check_project_layout.py - check that the project folder is tidy before uploading the code.

What it does (nothing is moved or deleted - it only reports):
  1. A folder overview (2 levels deep): number of files and size per folder.
  2. Every .py script and where it is; flags scripts outside scripts\, duplicate names,
     download copies like "x (1).py", and the known wrong file extract_wlasl_new.py.
  3. Expected folders that are missing.
  4. Files over 100 MB (GitHub refuses those) and the total size per folder type
     (code vs data), to help decide what goes to GitHub and what to Google Drive.
  5. Optional clean-up candidates (quick-test folders, __pycache__).
  Full list of every file -> project_inventory.csv (in the main folder).

Run from C:\SL_PROJECT :
  python -u scripts\tools\check_project_layout.py > report_outputs\V2_22_project_layout.txt
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(".")
EXPECTED = ["scripts/week1_word_selection", "scripts/week2_extraction", "scripts/week3_training",
            "scripts/week5_live", "scripts/v2_asl_citizen", "data/week1", "data/v3", "data/wlasl_new",
            "data/asl_citizen", "data/processed_v1", "data/processed_v2", "runs/tuning_v4", "runs/tuning_v5",
            "runs/tuning_v8", "runs/final_test", "runs/final_test_v2", "runs/final_test_best", "report_outputs",
            "models", "wlasl-processed"]
SKIP = {"venv", ".git", "node_modules"}
KNOWN_WRONG = {"scripts/v2_asl_citizen/extract_wlasl_new.py"}


def mb(b):
    return f"{b / 1024**2:,.1f} MB" if b < 1024**3 else f"{b / 1024**3:,.2f} GB"


def main():
    dirs = []
    for d, sub, _ in os.walk(ROOT):
        sub[:] = [s for s in sub if s not in SKIP]
        dirs.append(d)
    rows = []
    for d in tqdm(dirs, desc="Scanning folders", unit="folder", file=sys.stderr):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                rows.append({"path": rel, "folder": os.path.dirname(rel) or ".", "name": f,
                             "ext": os.path.splitext(f)[1].lower(), "bytes": os.path.getsize(p)})
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(ROOT, "project_inventory.csv"), index=False)

    print("=" * 70)
    print(f"1. FOLDER OVERVIEW ({ROOT}, venv skipped)")
    print("=" * 70)
    inv["top"] = inv["path"].str.split("/").str[0].where(inv["path"].str.contains("/"), ".")
    inv["top2"] = inv["path"].apply(lambda p: "/".join(p.split("/")[:2]) if p.count("/") >= 2 else os.path.dirname(p) or ".")
    for top, g in inv.groupby("top"):
        print(f"{top + '/':28s} {len(g):7,d} files  {mb(g['bytes'].sum()):>10s}")
        if top != "." and g["top2"].nunique() > 1:
            for t2, g2 in g.groupby("top2"):
                if t2 != top:
                    print(f"   {t2 + '/':40s} {len(g2):7,d} files  {mb(g2['bytes'].sum()):>10s}")
    root_files = inv[inv["folder"] == "."]
    print(f"\nFiles directly in the main folder ({len(root_files)}):")
    for n in sorted(root_files["name"]):
        print(f"   {n}")

    print("\n" + "=" * 70)
    print("2. PYTHON SCRIPTS")
    print("=" * 70)
    py = inv[inv["ext"] == ".py"].sort_values("path")
    for folder, g in py.groupby("folder"):
        print(f"{folder}/: {', '.join(g['name'])}")
    problems = []
    for p in py["path"]:
        if not p.startswith("scripts/"):
            problems.append(f"outside scripts/: {p}")
        if re.search(r"\(\d+\)\.py$", p) or " - Copy" in p:
            problems.append(f"download/copy duplicate: {p}")
        if p in KNOWN_WRONG:
            problems.append(f"known wrong file (holds old code) - delete: {p}")
    dup = py[py.duplicated("name", keep=False) & py["path"].str.startswith("scripts/")]
    for n, g in dup.groupby("name"):
        problems.append(f"same name in several folders: {n} -> {', '.join(g['path'])}")
    print("\nScript problems:" + ("" if problems else " none"))
    for pr in problems:
        print(f"   - {pr}")

    print("\n" + "=" * 70)
    print("3. EXPECTED FOLDERS")
    print("=" * 70)
    missing = [e for e in EXPECTED if not os.path.isdir(os.path.join(ROOT, e))]
    print("All present" if not missing else "Missing: " + ", ".join(missing))

    print("\n" + "=" * 70)
    print("4. SIZE (for choosing GitHub vs Google Drive)")
    print("=" * 70)
    big = inv[inv["bytes"] > 100 * 1024**2].sort_values("bytes", ascending=False)
    print(f"Files over 100 MB (GitHub refuses these): {len(big)}")
    for r in big.head(15).itertuples():
        print(f"   {mb(r.bytes):>10s}  {r.path}")
    kinds = defaultdict(int)
    for r in inv.itertuples():
        if r.ext in (".py", ".md", ".json") and not r.path.startswith(("data/", "runs/", "wlasl-processed/")):
            kinds["code + docs"] += r.bytes
        elif r.path.startswith("report_outputs/"):
            kinds["report_outputs"] += r.bytes
        elif r.path.startswith("runs/"):
            kinds["runs (models, results)"] += r.bytes
        elif r.path.startswith(("data/", "wlasl-processed/", "models/")):
            kinds["data, videos, MediaPipe models"] += r.bytes
        else:
            kinds["other"] += r.bytes
    for k, v in kinds.items():
        print(f"   {k:32s} {mb(v):>10s}")
    pt = inv[inv["ext"] == ".pt"]
    print(f"Trained model files (.pt): {len(pt)}, total {mb(pt['bytes'].sum())}")

    print("\n" + "=" * 70)
    print("5. OPTIONAL CLEAN-UP CANDIDATES (only if you want; nothing is deleted)")
    print("=" * 70)
    quick = sorted({p.split("/")[1] for p in inv["path"] if p.startswith("runs/") and "quicktest" in p})
    print("Quick-test folders (results mean nothing): " + (", ".join(f"runs/{q}" for q in quick) or "none"))
    cache = sorted({f for f in inv["folder"] if f.endswith("__pycache__")})
    print(f"__pycache__ folders: {len(cache)} (safe to ignore; never upload them)")
    print(f"\nSaved project_inventory.csv ({len(inv):,} files)")


if __name__ == "__main__":
    main()