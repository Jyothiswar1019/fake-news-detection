import argparse
import logging

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from fakenews.config import load_config, resolve
from fakenews.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("make_splits")


def _clean_chunk(chunk, strip, min_len, stem, dates):
    return [clean_text(t, strip, min_len, stem, dates) for t in chunk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tags", action="store_true",
                    help="do NOT strip source tags (used for the leakage comparison)")
    args = ap.parse_args()

    cfg = load_config()
    pc, d = cfg["preprocess"], cfg["data"]
    strip = pc["strip_source_tags"] and not args.keep_tags
    dates = pc.get("strip_dates", True) and not args.keep_tags
    tag = "stripped" if strip else "with_tags"

    df = pd.read_csv(resolve(cfg["paths"]["clean_data"]))
    texts = df["content"].tolist()
    log.info("Cleaning %d documents (strip_source_tags=%s, strip_dates=%s)...",
             len(texts), strip, dates)

    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    parts = Parallel(n_jobs=-1)(
        delayed(_clean_chunk)(c, strip, pc["min_token_len"], pc["stem"], dates)
        for c in chunks
    )
    df["clean_text"] = [t for part in parts for t in part]

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    log.info("Dropped %d rows that were empty after cleaning", before - len(df))

    train, test = train_test_split(
        df, test_size=d["test_size"], random_state=d["random_state"],
        stratify=df[d["label_col"]],
    )

    out_dir = resolve("data/processed") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    for name, part in (("train", train), ("test", test)):
        share = part[d["label_col"]].value_counts(normalize=True).round(3).to_dict()
        log.info("%s: %d rows, label share %s", name, len(part), share)
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()