import argparse

import pandas as pd

from fakenews.config import resolve


def read_any(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw/IFND.csv")
    ap.add_argument("--text-col", default="Statement")
    ap.add_argument("--label-col", default="Label")
    args = ap.parse_args()

    df = read_any(resolve(args.path))
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nNulls:\n", df.isna().sum())

    if args.text_col not in df or args.label_col not in df:
        print("\nSet --text-col and --label-col to match the column names above.")
        return

    text, label = df[args.text_col].astype(str), df[args.label_col]
    print("\nLabel counts:\n", label.value_counts(dropna=False))

    words = text.str.split().str.len()
    print("\nWords per text, by label:\n", words.groupby(label).describe().round(1))

    print("\nExact duplicate texts:", int(text.duplicated().sum()))
    prefix = text.str.split().str[:8].str.join(" ")
    print("Texts sharing the same first 8 words with another (augmentation proxy):",
          int(prefix.duplicated(keep=False).sum()))

    for lab in label.dropna().unique():
        print(f"\n--- samples, label = {lab} ---")
        for t in text[label == lab].sample(5, random_state=1):
            print("-", t[:250].replace("\n", " "))


if __name__ == "__main__":
    main()