import pandas as pd

df = pd.read_csv(r"C:\SL_PROJECT\word_counts.csv")
cand = df[df.in_wlasl100 & (df.have >= 8)].sort_values(["have", "signers"], ascending=False)

print("Candidates:", len(cand))
print(cand[["gloss", "have", "signers"]].to_string(index=False))