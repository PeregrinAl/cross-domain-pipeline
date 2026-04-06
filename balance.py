import pandas as pd

df = pd.read_csv("data/processed/manifest.csv")
print(df.groupby(["split", "domain", "record_label", "label"]).size())
print(df[df["record_label"] == 1][["record_id", "window_idx", "start", "end", "label", "overlap_fraction"]].head(20))