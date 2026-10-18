"""
v2 step 2: feasibility check for ASL Citizen (no full download).

What it does, in plain words:
  1. Shows how much free disk space the laptop has (the full zip is 42.8 GB).
  2. Opens the ASL Citizen zip ON MICROSOFT'S SERVER and reads only its
     table of contents and its small metadata CSV files (a few MB).
     The 42.8 GB of videos are NOT downloaded.
  3. Builds the full WLASL100 word list (first 100 glosses of WLASL_v0.3.json),
     marks which are in our v1 80 words, and counts WLASL clips we actually have.
  4. Matches all 100 words to ASL Citizen's sign names ("glosses"): exact match
     first, then near-matches to check by hand, and counts clips and signers.

Run from C:\\SL_PROJECT\\ :
  pip install remotezip
  python -u scripts\\v2_asl_citizen\\check_asl_citizen.py > report_outputs\\V2_01_feasibility.txt

Optional: give a local path instead of the URL (e.g. if you already have the zip):
  python -u scripts\\v2_asl_citizen\\check_asl_citizen.py D:\\ASL_Citizen.zip

Outputs:
  data\\asl_citizen\\meta\\        the metadata CSVs copied out of the zip
  data\\asl_citizen\\zip_contents.csv   every file in the zip with its size
  data\\asl_citizen\\word_match.csv     WLASL100 words -> WLASL clips + ASL Citizen glosses, clips, signers
"""
import os
import re
import sys
import shutil
import difflib
import zipfile

import json

import pandas as pd

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
WORD_LIST = os.path.join("data", "week1", "word_list.csv")      # our v1 80 words
WLASL_JSON = os.path.join("wlasl-processed", "WLASL_v0.3.json")
WLASL_VIDEOS = os.path.join("wlasl-processed", "videos")
OUT_DIR = os.path.join("data", "asl_citizen")
META_DIR = os.path.join(OUT_DIR, "meta")
FULL_ZIP_GB = 42.8


def gb(n_bytes):
    return n_bytes / 1024**3


def open_zip(source):
    """Open a local zip, or the remote one via HTTP range requests."""
    if os.path.exists(source):
        print(f"Opening local zip: {source}")
        return zipfile.ZipFile(source)
    try:
        from remotezip import RemoteZip
    except ImportError:
        sys.exit("remotezip is not installed. Run:  pip install remotezip")
    print("Opening the zip on Microsoft's server (reading the table of contents only)...")
    try:
        return RemoteZip(source)
    except Exception as e:
        print("\nCOULD NOT READ THE ZIP REMOTELY.")
        print(f"  Reason: {type(e).__name__}: {e}")
        print("  This usually means the server does not allow partial downloads.")
        print("  Fallback: download the full zip, then run this script with its path.")
        sys.exit(1)


def norm(g):
    """Simplify a sign name for matching: lower case, letters only, no variant numbers.
    e.g. 'THANKSGIVING' -> 'thanksgiving', 'RIGHT_2' -> 'right', 'thank-you' -> 'thankyou'."""
    g = str(g).strip().lower()
    g = re.sub(r"[\s_\-\.]*\d+$", "", g)   # trailing variant number
    return re.sub(r"[^a-z]", "", g)


def load_words():
    """WLASL100 = the first 100 glosses in WLASL_v0.3.json (sorted by number of videos).
    Returns a table: word, in_v1 (one of our 80), wlasl_clips_listed, wlasl_clips_available."""
    if not os.path.exists(WLASL_JSON):
        sys.exit(f"Cannot find {WLASL_JSON}. Tell Claude where WLASL_v0.3.json is.")
    with open(WLASL_JSON) as f:
        entries = json.load(f)[:100]
    have = set()
    if os.path.isdir(WLASL_VIDEOS):
        have = {os.path.splitext(n)[0] for n in os.listdir(WLASL_VIDEOS)}
    else:
        print(f"WARNING: {WLASL_VIDEOS} not found, WLASL available counts will be 0")
    v1 = pd.read_csv(WORD_LIST)
    col = next((c for c in v1.columns if c.lower() in ("word", "gloss")), v1.columns[0])
    v1_words = set(v1[col].astype(str).str.strip().str.lower())
    rows = []
    for e in entries:
        ids = [str(i["video_id"]) for i in e["instances"]]
        avail = [i for i in e["instances"] if str(i["video_id"]) in have]
        rows.append({"word": e["gloss"].strip().lower(),
                     "in_v1": e["gloss"].strip().lower() in v1_words,
                     "wlasl_clips_listed": len(ids),
                     "wlasl_clips_available": len(avail),
                     "wlasl_signers_available": len({i.get("signer_id") for i in avail})})
    words = pd.DataFrame(rows)
    print(f"WLASL100 words: {len(words)}  (in our v1 list: {words.in_v1.sum()}, "
          f"new: {(~words.in_v1).sum()})")
    missing = v1_words - set(words.word)
    if missing:
        print(f"WARNING: v1 words not found in WLASL100: {sorted(missing)}")
    return words


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else URL

    # ---- 1. Disk space ------------------------------------------------------
    print("=" * 70)
    print("1. DISK SPACE")
    print("=" * 70)
    du = shutil.disk_usage(os.path.abspath("."))
    print(f"Drive of {os.path.abspath('.')}: {gb(du.free):.1f} GB free of {gb(du.total):.1f} GB")
    print(f"Full ASL Citizen zip: {FULL_ZIP_GB} GB "
          f"({'fits' if gb(du.free) > FULL_ZIP_GB + 10 else 'DOES NOT fit with 10 GB spare'})")

    # ---- 2. Zip contents ----------------------------------------------------
    print("\n" + "=" * 70)
    print("2. WHAT IS INSIDE THE ZIP")
    print("=" * 70)
    z = open_zip(source)
    infos = [i for i in z.infolist() if not i.is_dir()]
    os.makedirs(META_DIR, exist_ok=True)
    contents = pd.DataFrame({"name": [i.filename for i in infos],
                             "size_bytes": [i.file_size for i in infos]})
    contents["ext"] = contents["name"].str.extract(r"\.([^./\\]+)$")[0].str.lower()
    contents.to_csv(os.path.join(OUT_DIR, "zip_contents.csv"), index=False)
    print(f"Files in zip: {len(contents)}  (total {gb(contents.size_bytes.sum()):.1f} GB unpacked)")
    print(contents.groupby("ext")["size_bytes"].agg(["count", "sum"])
          .assign(GB=lambda d: (d["sum"] / 1024**3).round(2))
          .drop(columns="sum").to_string())

    # ---- copy out the metadata CSVs -----------------------------------------
    csv_names = [n for n in contents["name"] if n.lower().endswith(".csv")]
    print(f"\nMetadata CSVs found: {len(csv_names)}")
    tables = {}
    for n in csv_names:
        local = os.path.join(META_DIR, os.path.basename(n))
        with open(local, "wb") as f:
            f.write(z.read(n))
        df = pd.read_csv(local)
        tables[os.path.basename(n)] = df
        print(f"  {n}  ->  {local}   rows={len(df)}  columns={list(df.columns)}")

    # Clip tables = CSVs with a gloss column and a video file column
    def find_col(df, keys):
        return next((c for c in df.columns if any(k in c.lower() for k in keys)), None)

    clip_parts = []
    for name, df in tables.items():
        gcol, vcol = find_col(df, ["gloss"]), find_col(df, ["video", "file"])
        pcol = find_col(df, ["participant", "user", "signer"])
        if gcol and vcol:
            part = pd.DataFrame({"video": df[vcol], "gloss": df[gcol].astype(str).str.strip(),
                                 "signer": df[pcol] if pcol else None,
                                 "asl_split": os.path.splitext(name)[0]})
            clip_parts.append(part)
    if not clip_parts:
        sys.exit("No CSV with gloss + video columns found. Paste the column list above back to Claude.")
    clips = pd.concat(clip_parts, ignore_index=True)
    print(f"\nAll ASL Citizen clips: {len(clips)}, glosses: {clips.gloss.nunique()}, "
          f"signers: {clips.signer.nunique()}")

    # Size of each video (to estimate our download)
    size_by_base = dict(zip(contents["name"].map(os.path.basename), contents["size_bytes"]))
    clips["size_bytes"] = clips["video"].astype(str).map(os.path.basename).map(size_by_base)

    # ---- 3. Match our words -------------------------------------------------
    print("\n" + "=" * 70)
    print("3. MATCHING THE WLASL100 WORDS TO ASL CITIZEN GLOSSES")
    print("=" * 70)
    word_table = load_words()
    words = word_table["word"].tolist()
    glosses = sorted(clips.gloss.unique())
    by_norm = {}
    for g in glosses:
        by_norm.setdefault(norm(g), []).append(g)

    rows = []
    for w in words:
        exact = [g for g in glosses if g.lower() == w]
        same_after_norm = by_norm.get(norm(w), [])
        if exact and len(same_after_norm) == 1:
            mtype, matched = "exact", exact
        elif same_after_norm:
            # e.g. several numbered variants (RIGHT1, RIGHT2) -> must check by eye
            mtype, matched = ("exact+variants" if exact else "variant"), same_after_norm
        else:
            close = difflib.get_close_matches(w, glosses, n=3, cutoff=0.75)
            close += [g for g in glosses if w in norm(g).split() or norm(g).startswith(w)][:3]
            close = list(dict.fromkeys(close))
            mtype, matched = ("near (check)" if close else "none"), close
        sub = clips[clips.gloss.isin(matched)] if mtype in ("exact", "exact+variants", "variant") else clips.iloc[0:0]
        rows.append({
            "word": w, "match_type": mtype, "asl_citizen_glosses": "; ".join(matched),
            "clips": len(sub), "signers": sub.signer.nunique() if len(sub) else 0,
            "size_MB": round(sub.size_bytes.sum() / 1024**2, 1) if len(sub) else 0.0,
        })
    match = word_table.merge(pd.DataFrame(rows), on="word")
    match = match.rename(columns={"clips": "asc_clips", "signers": "asc_signers"})
    match.to_csv(os.path.join(OUT_DIR, "word_match.csv"), index=False)

    pd.set_option("display.width", 200)
    print(match.to_string(index=False))

    # ---- 4. Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. SUMMARY")
    print("=" * 70)
    for label, part in [("v1 words", match[match.in_v1]), ("NEW words", match[~match.in_v1])]:
        ok = part[part.match_type == "exact"]
        print(f"\n{label}: {len(part)} words")
        print("  match types: " + ", ".join(f"{k} {v}" for k, v in part.match_type.value_counts().items()))
        print(f"  WLASL clips available per word: min {part.wlasl_clips_available.min()} / "
              f"median {part.wlasl_clips_available.median():.0f}")
        if len(ok):
            print(f"  clean exact matches: {len(ok)}; ASL Citizen clips total {ok.asc_clips.sum()}, "
                  f"per word min {ok.asc_clips.min()} / median {ok.asc_clips.median():.0f}; "
                  f"signers per word min {ok.asc_signers.min()} / median {ok.asc_signers.median():.0f}; "
                  f"download {ok.size_MB.sum() / 1024:.2f} GB")

    print("\nNEW words in detail (WLASL clips available -> likely val/test clips):")
    new = match[~match.in_v1]
    print(new[["word", "wlasl_clips_available", "wlasl_signers_available", "match_type",
               "asl_citizen_glosses", "asc_clips", "asc_signers"]].to_string(index=False))

    need_check = match[match.match_type != "exact"]
    print(f"\nWords needing a manual gloss check: {len(need_check)}")
    for _, r in need_check.iterrows():
        tag = "" if r.in_v1 else " [new]"
        print(f"  {r.word:15s} {r.match_type:15s} {r.asl_citizen_glosses}{tag}")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'word_match.csv')}")


if __name__ == "__main__":
    main()