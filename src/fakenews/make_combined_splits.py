"""Merge cleaned WELFake + IFND, clean the text, and make one grouped, stratified split.

Usage:
    python -m fakenews.make_combined_splits                 # IFND without the synthetic AUGMENT rows
    python -m fakenews.make_combined_splits --keep-augment  # ablation: keep them
"""
import argparse
import logging

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from fakenews.config import load_config, resolve
from fakenews.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("make_combined_splits")


def _clean_chunk(chunk, strip, min_len, stem, dates):
    return [clean_text(t, strip, min_len, stem, dates) for t in chunk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-augment", action="store_true",
                    help="keep IFND rows produced by the dataset authors' augmentation algorithm")
    args = ap.parse_args()

    cfg = load_config()
    pc, d = cfg["preprocess"], cfg["data"]
    label = d["label_col"]

    wel = pd.read_csv(resolve(cfg["paths"]["clean_data"]))[["content", label]].copy()
    wel["source"], wel["site"] = "welfake", "welfake"
    wel["group"] = ["w%d" % i for i in range(len(wel))]  # WELFake is already deduplicated

    ifnd = pd.read_csv(resolve("data/processed/ifnd_clean.csv"))
    ifnd["site"] = ifnd["site"].str.upper().str.strip().replace({"THESTATEMAN": "THESTATESMAN"})
    n_aug = int((ifnd["site"] == "AUGMENT").sum())
    if not args.keep_augment:
        ifnd = ifnd[ifnd["site"] != "AUGMENT"]
        log.info("Dropped %d synthetic AUGMENT rows", n_aug)
    ifnd = ifnd.rename(columns={"label": label})[["content", label, "source", "site", "group"]]

    df = pd.concat([wel, ifnd], ignore_index=True).drop_duplicates("content").reset_index(drop=True)
    log.info("Merged: %d rows", len(df))

    strip, dates = pc["strip_source_tags"], pc.get("strip_dates", True)
    texts = df["content"].tolist()
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    parts = Parallel(n_jobs=-1)(
        delayed(_clean_chunk)(c, strip, pc["min_token_len"], pc["stem"], dates) for c in chunks
    )
    df["clean_text"] = [t for part in parts for t in part]
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    log.info("Dropped %d rows that were empty after cleaning", before - len(df))

    # ~33% test; rows of one near-duplicate group never straddle the split;
    # stratified by source x label so every source keeps its class mix
    strata = df["source"] + "_" + df[label].astype(str)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=d["random_state"])
    train_idx, test_idx = next(splitter.split(df, strata, groups=df["group"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    assert not set(train["group"]) & set(test["group"]), "group leakage between train and test"

    tag = "combined_with_augment" if args.keep_augment else "combined"
    out = resolve("data/processed") / tag
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    for name, part in (("train", train), ("test", test)):
        counts = part.groupby(["source", label]).size().unstack(fill_value=0)
        log.info("%s: %d rows\n%s", name, len(part), counts.to_string())
    log.info("Saved to %s", out)


if __name__ == "__main__":
    main()
