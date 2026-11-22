import logging

import pandas as pd

from fakenews.config import load_config, resolve

logger = logging.getLogger(__name__)


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download WELFake from Zenodo into data/raw/."
        )
    df = pd.read_csv(path)
    # WELFake ships with an unnamed index column
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    d = cfg["data"]
    required = {d["text_col"], d["title_col"], d["label_col"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(df[d["text_col"]].dropna().duplicated().sum()),
        "label_counts": {str(k): int(v) for k, v in df[d["label_col"]].value_counts(dropna=False).items()},
    }


def clean_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop unusable rows and duplicates. Returns columns: content, label."""
    d = cfg["data"]
    text, title, label = d["text_col"], d["title_col"], d["label_col"]
    n_start = len(df)

    df = df.dropna(subset=[label]).copy()
    df[text] = df[text].fillna("")
    df[title] = df[title].fillna("")

    if d["use_title"]:
        df["content"] = (df[title] + " " + df[text]).str.strip()
    else:
        df["content"] = df[text].str.strip()

    df = df[df["content"].str.len() > 0]

    # Same article with conflicting labels is label noise: drop all copies
    n_labels = df.groupby("content")[label].nunique()
    conflicting = n_labels[n_labels > 1].index
    df = df[~df["content"].isin(conflicting)]

    # Duplicates must go BEFORE splitting, otherwise copies leak into the test set
    df = df.drop_duplicates(subset="content")

    df = df[["content", label]].reset_index(drop=True)
    df[label] = df[label].astype(int)
    logger.info("Cleaning kept %d of %d rows", len(df), n_start)
    return df