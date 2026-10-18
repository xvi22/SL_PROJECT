import json, os
import pandas as pd

BASE = r"C:\SL_PROJECT\wlasl-processed"

with open(os.path.join(BASE, "WLASL_v0.3.json")) as f:
    data = json.load(f)

have = set(os.path.splitext(v)[0] for v in os.listdir(os.path.join(BASE, "videos")))
print("Video files found:", len(have))

# Work out which words are in WLASL100 (nslt_100 uses numbers; the class list turns numbers into words)
with open(os.path.join(BASE, "nslt_100.json")) as f:
    nslt100 = json.load(f)
class_ids = set(v["action"][0] for v in nslt100.values())

classes = {}
with open(os.path.join(BASE, "wlasl_class_list.txt")) as f:
    for line in f:
        idx, word = line.strip().split(maxsplit=1)
        classes[int(idx)] = word
wlasl100 = set(classes[i] for i in class_ids)

rows = []
for entry in data:
    ok = [i for i in entry["instances"] if i["video_id"] in have]
    rows.append({
        "gloss": entry["gloss"],
        "listed": len(entry["instances"]),
        "have": len(ok),
        "signers": len(set(i["signer_id"] for i in ok)),
        "in_wlasl100": entry["gloss"] in wlasl100,
    })

df = pd.DataFrame(rows).sort_values(["have", "signers"], ascending=False)
df.to_csv(r"C:\SL_PROJECT\word_counts.csv", index=False)

print(df.head(40).to_string(index=False))
print("\nWLASL100 words with 8+ clips:", (df.in_wlasl100 & (df.have >= 8)).sum())