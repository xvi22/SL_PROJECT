r"""
tidy_project.py - reorganise C:\SL_PROJECT into folders, without breaking any script.

    python tidy_project.py          # DRY RUN: only shows what would happen
    python tidy_project.py --go     # actually do it

What it does:
  1. backs up every .py and .csv in the main folder to backup_before_tidy.zip
  2. moves scripts and data into folders (names stay the same, so the report is still right)
  3. updates the file paths written inside each script, so they still work
  4. writes README.md explaining the layout

After tidying, ALWAYS run scripts from the main folder, e.g.
    python scripts\week2_extraction\clean_hands.py
Untouched: venv\, models\, WLASL-master\, wlasl-processed\, report_outputs\.
"""

import os
import re
import shutil
import sys
import zipfile

GO = "--go" in sys.argv

SCRIPT_GROUPS = {
    "scripts/week1_word_selection": ["find_word.py", "count_words.py", "pick_words.py",
                                     "save_word_list.py"],
    "scripts/week2_extraction": ["test_one_clip.py", "extract_all.py", "check_clips.py",
                                 "extract_all_v2.py", "check_clips_v2.py", "compare_v1_v2.py",
                                 "find_phantoms.py", "clean_hands.py"],
    "scripts/week3_training": ["preprocess.py", "split.py", "train_baseline.py"],
    "scripts/week5_live": ["mirror_frame.py"],
}

DATA_GROUPS = {
    "data/week1": ["word_counts.csv", "word_list.csv"],
    "data/test": ["landmarks_test"],
    "data/v1": ["landmarks", "landmarks_index.csv", "span_stats.csv", "checks",
                "extraction_failures.txt"],
    "data/v2": ["landmarks_v2", "landmarks_index_v2.csv", "span_stats_v2.csv", "checks_v2",
                "extraction_failures_v2.txt", "compare_v1_v2.csv", "phantom_stats.csv",
                "phantoms"],
    "data/v3": ["landmarks_v3", "landmarks_index_v3.csv", "clean_checks"],
    "data": ["processed_v1", "split_v1.csv", "video_sizes.csv"],  # Week 3 outputs, if any
}

# every path written inside a script, as it appears between quotes -> its new location
PATH_MAP = {}
for folder, names in DATA_GROUPS.items():
    for name in names:
        PATH_MAP[name] = f"{folder}/{name}"


# names that a script uses for something else (not the moved file) -> leave them alone
SKIP = {"preprocess.py": {"checks"}}  # preprocess saves its own processed_v1\checks\ images


def plan_moves():
    moves = []
    for groups in (SCRIPT_GROUPS, DATA_GROUPS):
        for folder, names in groups.items():
            for name in names:
                if os.path.exists(name):
                    moves.append((name, os.path.join(folder, name)))
    return moves


def patch_script(path):
    """Replace quoted file names (e.g. "landmarks_v3") with their new paths."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    changes = []
    skip = SKIP.get(os.path.basename(path), set())
    for old, new in PATH_MAP.items():
        if old in skip:
            continue
        pattern = re.compile(r'(["\'])' + re.escape(old) + r'\1')
        text, n = pattern.subn(lambda m: m.group(1) + new + m.group(1), text)
        if n:
            changes.append(f"{old} -> {new} ({n}x)")
    if changes and GO:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return changes


README = """# SL_PROJECT layout

Run every script FROM THIS MAIN FOLDER, for example:
    python scripts\\week2_extraction\\clean_hands.py

scripts\\
  week1_word_selection\\   find_word, count_words, pick_words, save_word_list
  week2_extraction\\       test_one_clip, extract_all (v1), check_clips (v1),
                           extract_all_v2, check_clips_v2, compare_v1_v2,
                           find_phantoms, clean_hands (v3)
  week3_training\\         preprocess, split, train_baseline
  week5_live\\             mirror_frame (left-handed mode)

data\\
  week1\\    word_counts.csv, word_list.csv (the 80 words)
  test\\     landmarks_test\\ (one-clip test)
  v1\\       landmarks\\, landmarks_index.csv, span_stats.csv, checks\\   (record only)
  v2\\       landmarks_v2\\, index, span stats, checks_v2\\, compare_v1_v2.csv,
            phantom_stats.csv, phantoms\\                               (record only)
  v3\\       landmarks_v3\\, landmarks_index_v3.csv, clean_checks\\      (FINAL dataset)
  processed_v1\\, split_v1.csv, video_sizes.csv                         (Week 3)

report_outputs\\   script outputs for the report appendix
runs\\             training runs (Week 3 onwards)
models\\           MediaPipe model files
wlasl-processed\\, WLASL-master\\   original WLASL data (untouched)
backup_before_tidy.zip   copy of all scripts and CSVs from before the tidy-up
"""


def main():
    moves = plan_moves()
    if not moves:
        print("Nothing to move - the project already looks tidy.")
        return
    print("DRY RUN - nothing is changed yet.\n" if not GO else "TIDYING UP...\n")

    loose = [f for f in os.listdir(".") if f.endswith((".py", ".csv")) and f != "tidy_project.py"]
    if GO:
        with zipfile.ZipFile("backup_before_tidy.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in loose:
                z.write(f)
        print(f"Backup: {len(loose)} scripts/CSVs saved to backup_before_tidy.zip\n")
    else:
        print(f"Backup: {len(loose)} scripts/CSVs would be saved to backup_before_tidy.zip\n")

    print("Moves:")
    for src, dst in moves:
        if os.path.exists(dst):
            print(f"  SKIP {src} (already exists at {dst})")
            continue
        print(f"  {src:28s} -> {dst}")
        if GO:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    print("\nPaths updated inside scripts:")
    for folder, names in SCRIPT_GROUPS.items():
        for name in names:
            path = os.path.join(folder, name) if GO else name
            if not os.path.exists(path):
                continue
            changes = patch_script(path)
            print(f"  {name}: " + ("; ".join(changes) if changes else "no file paths to update"))

    if GO:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(README)
        print("\nWrote README.md. Done. Run scripts from this main folder, e.g.")
        print("  python scripts\\week2_extraction\\clean_hands.py")
    else:
        print("\nLooks right? Run:  python tidy_project.py --go")


if __name__ == "__main__":
    main()