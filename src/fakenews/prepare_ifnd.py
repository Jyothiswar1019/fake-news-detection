"""Convert IFND to the project schema: content, label (0 real / 1 fake), source, site, group."""
import argparse
import re

import pandas as pd

from fakenews.config import resolve
from fakenews.inspect_ifnd import read_any

LABEL_MAP = {"true": 0, "fake": 1}
MARKERS = {
    "fact check": r"fact[\s-]*check",
    "no, this is not": r"\bno,? this is not\b",
    "false claim": r"false claim",
    "viral": r"\bviral\b",
    "photo / video": r"\b(?:photo|video|image)\b",
    "question mark": r"\?",
}


def normalize(text: str) -> str:
    """Lowercase letters/digits only: also hides encoding junk such as 'Û÷' or 'â'."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw/IFND.csv")
    ap.add_argument("--out", default="data/processed/ifnd_clean.csv")
    args = ap.parse_args()

    raw = read_any(resolve(args.path))
    df = pd.DataFrame({
        # some fake rows were glued together without a space ("Balakot.No, this ...")
        "content": raw["Statement"].astype(str)
                      .str.replace(r"([a-z])\.([A-Z])", r"\1. \2", regex=True).str.strip(),
        "label": raw["Label"].astype(str).str.strip().str.lower().map(LABEL_MAP),
        "site": raw["Web"].astype(str).str.strip(),
    })
    if df["label"].isna().any():
        bad = raw.loc[df["label"].isna(), "Label"].unique()
        raise ValueError(f"Unexpected label values: {bad}")
    df["label"] = df["label"].astype(int)

    n0 = len(df)
    df["norm"] = df["content"].map(normalize)
    df = df[df["norm"].str.split().str.len() >= 3]

    # same text with conflicting labels is label noise: drop every copy
    n_labels = df.groupby("norm")["label"].nunique()
    df = df[~df["norm"].isin(n_labels[n_labels > 1].index)]
    df = df.drop_duplicates("norm")

    # near-duplicate key (augmented or templated headlines share their opening words);
    # used later so such rows never land on both sides of the split
    df["group"] = df["norm"].str.split().str[:8].str.join(" ")
    df["source"] = "ifnd"
    df = df[["content", "label", "source", "site", "group"]].reset_index(drop=True)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Kept {len(df):,} of {n0:,} rows -> {out}")
    print("Label counts:", df["label"].value_counts().to_dict())
    sizes = df.groupby("group").size()
    print(f"Near-duplicate groups: {len(sizes):,}; rows in groups of 2+: {int(sizes[sizes > 1].sum()):,}")
    print("Largest groups:\n", sizes.sort_values(ascending=False).head(6).to_string())

    ct = pd.crosstab(df["site"], df["label"])
    ct["total"] = ct.sum(axis=1)
    ct["fake_share"] = (ct.get(1, 0) / ct["total"]).round(3)
    print("\nSite vs label (top 15 sites):\n", ct.sort_values("total", ascending=False).head(15).to_string())

    rows = []
    for name, pat in MARKERS.items():
        hit = df["content"].str.contains(pat, case=False, regex=True)
        rows.append({"marker": name, "real_%": round(hit[df.label == 0].mean() * 100, 1),
                     "fake_%": round(hit[df.label == 1].mean() * 100, 1)})
    print("\nStyle markers:\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
