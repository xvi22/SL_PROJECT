import pandas as pd

df = pd.read_csv(r"C:\SL_PROJECT\word_counts.csv")
cand = df[df.in_wlasl100 & (df.have >= 8)]

DROP = ["cheat", "language", "kiss", "same", "cow", "africa", "paper", "color"]

final = cand[~cand.gloss.isin(DROP)].sort_values(["have", "signers"], ascending=False)
final.to_csv(r"C:\SL_PROJECT\word_list.csv", index=False)

print("Words kept:", len(final))
print("Total clips:", final.have.sum())
print("Fewest clips:", final.have.min(), "| Fewest signers:", final.signers.min())