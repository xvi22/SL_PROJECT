"""
v2 step 3: download ONLY the ASL Citizen videos we need (not the 42.8 GB zip).

What it does, in plain words:
  1. Reads data\\asl_citizen\\word_match.csv (made by check_asl_citizen.py).
  2. Collects every candidate gloss for our 100 words, INCLUDING all numbered
     variants (e.g. DRINK1 and DRINK2) and near matches, because we still
     have to check by eye which variant is the same sign as in WLASL.
  3. Pulls just those videos out of the zip on Microsoft's server.
     Resumable: videos already downloaded (with the right size) are skipped.

Run from C:\\SL_PROJECT\\ :
  python -u scripts\\v2_asl_citizen\\download_asl_citizen.py > report_outputs\\V2_02_download.txt

Optional: a local zip path instead of the URL:
  python -u scripts\\v2_asl_citizen\\download_asl_citizen.py D:\\ASL_Citizen.zip

Outputs:
  data\\asl_citizen\\videos\\           the downloaded .mp4 files
  data\\asl_citizen\\candidates.csv     one row per video: word, gloss, signer, ASL Citizen split, path
"""
import os
import sys
import zipfile

import pandas as pd
from tqdm import tqdm

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
BASE = os.path.join("data", "asl_citizen")
META = os.path.join(BASE, "meta")
VIDEO_DIR = os.path.join(BASE, "videos")
MATCH = os.path.join(BASE, "word_match.csv")


def open_zip(source):
    if os.path.exists(source):
        return zipfile.ZipFile(source)
    from remotezip import RemoteZip
    return RemoteZip(source)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else URL

    # ---- 1. Which glosses? ---------------------------------------------------
    match = pd.read_csv(MATCH)
    pairs = []
    for _, r in match.iterrows():
        if isinstance(r.asl_citizen_glosses, str) and r.asl_citizen_glosses.strip():
            for g in r.asl_citizen_glosses.split(";"):
                pairs.append((r.word, g.strip(), r.match_type))
    gloss_map = pd.DataFrame(pairs, columns=["word", "gloss", "match_type"])
    print(f"Candidate glosses: {len(gloss_map)} for {gloss_map.word.nunique()} words")
    no_gloss = sorted(set(match.word) - set(gloss_map.word))
    if no_gloss:
        print(f"Words with no candidate gloss at all: {no_gloss}")

    # ---- 2. Which videos? ----------------------------------------------------
    meta = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(META, f"{split}.csv"))
        df = df.rename(columns={"Participant ID": "signer", "Video file": "video",
                                "Gloss": "gloss", "ASL-LEX Code": "asl_lex_code"})
        df["gloss"] = df["gloss"].astype(str).str.strip()
        df["asl_split"] = split   # ASL Citizen's own split; for us ALL of these are training candidates
        meta.append(df)
    meta = pd.concat(meta, ignore_index=True)
    cand = gloss_map.merge(meta, on="gloss", how="inner")
    cand["path"] = cand["video"].map(lambda v: os.path.join(VIDEO_DIR, os.path.basename(v)))
    print(f"Videos to get: {len(cand)} ({cand.video.nunique()} unique)")

    # ---- 3. Download ---------------------------------------------------------
    print("Opening the zip...")
    z = open_zip(source)
    by_base = {os.path.basename(i.filename): i for i in z.infolist() if not i.is_dir()}
    os.makedirs(VIDEO_DIR, exist_ok=True)

    todo, skipped, missing = [], 0, []
    for v, p in cand[["video", "path"]].drop_duplicates().itertuples(index=False):
        info = by_base.get(os.path.basename(v))
        if info is None:
            missing.append(v)
        elif os.path.exists(p) and os.path.getsize(p) == info.file_size:
            skipped += 1
        else:
            todo.append((info, p))
    total_mb = sum(i.file_size for i, _ in todo) / 1024**2
    print(f"Already downloaded: {skipped}. To download now: {len(todo)} ({total_mb:.0f} MB). "
          f"Not in zip: {len(missing)}")

    failed = []
    with tqdm(total=len(todo), unit="video", file=sys.stderr) as bar:
        for info, p in todo:
            try:
                data = z.read(info)
                tmp = p + ".part"           # write to a temp name first, so a crash
                with open(tmp, "wb") as f:  # never leaves a half-written .mp4
                    f.write(data)
                os.replace(tmp, p)
            except Exception as e:
                failed.append((info.filename, f"{type(e).__name__}: {e}"))
            bar.update(1)

    cand["downloaded"] = cand["path"].map(os.path.exists)
    cand.to_csv(os.path.join(BASE, "candidates.csv"), index=False)

    # ---- 4. Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Videos on disk: {cand.downloaded.sum()} of {len(cand)}")
    print(f"Failed this run: {len(failed)}  (just run the script again to retry)")
    for name, err in failed[:10]:
        print(f"  {name}: {err}")
    for v in missing[:10]:
        print(f"  not in zip: {v}")
    on_disk = sum(os.path.getsize(p) for p in cand.path.unique() if os.path.exists(p))
    print(f"Space used: {on_disk / 1024**3:.2f} GB in {VIDEO_DIR}")
    per_gloss = cand.groupby(["word", "gloss"]).agg(clips=("video", "count"),
                                                     signers=("signer", "nunique"),
                                                     on_disk=("downloaded", "sum"))
    print("\nPer gloss:")
    print(per_gloss.to_string())
    print(f"\nSaved: {os.path.join(BASE, 'candidates.csv')}")


if __name__ == "__main__":
    main()