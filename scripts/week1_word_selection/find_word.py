import json, os

WORD = "book"   # change this to any word you want to check

JSON_PATH = r"C:\SL_PROJECT\wlasl-processed\WLASL_v0.3.json"
VIDEO_DIR = r"C:\SL_PROJECT\wlasl-processed\videos"

with open(JSON_PATH) as f:
    data = json.load(f)

have = set(os.path.splitext(v)[0] for v in os.listdir(VIDEO_DIR))

for entry in data:
    if entry["gloss"] == WORD:
        for inst in entry["instances"]:
            vid = inst["video_id"]
            status = "HAVE" if vid in have else "missing"
            print(f"{vid}.mp4   signer {inst['signer_id']:>3}   {inst['split']:<5}   {status}")