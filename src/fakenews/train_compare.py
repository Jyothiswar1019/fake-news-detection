import argparse
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

from fakenews.config import load_config, resolve
from fakenews.features import build_vectorizer
from fakenews.models import get_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_compare")


def evaluate(y_true, y_pred) -> dict:
    # Positive class = 1 (fake). y_true always goes first.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stripped", choices=["stripped", "with_tags"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset, e.g. --models LogisticRegression PassiveAggressive")
    args = ap.parse_args()

    cfg = load_config()
    label = cfg["data"]["label_col"]
    data_dir = resolve("data/processed") / args.variant
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    log.info("Variant=%s | train=%d test=%d", args.variant, len(train), len(test))

    models = get_models()
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    rows, cms = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("vec", build_vectorizer(cfg)), ("clf", clf)])
        t0 = time.time()
        pipe.fit(train["clean_text"], train[label])
        preds = pipe.predict(test["clean_text"])
        secs = time.time() - t0

        m = evaluate(test[label], preds)
        rows.append({"model": name, **m, "seconds": round(secs, 1)})
        cms[name] = confusion_matrix(test[label], preds)
        log.info("%-20s acc=%.4f f1=%.4f (%.0fs)", name, m["accuracy"], m["f1"], secs)

    results = pd.DataFrame(rows).round(4)
    reports = resolve(cfg["paths"]["reports_dir"])
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / f"model_comparison_{args.variant}.csv", index=False)

    fig, axes = plt.subplots(1, len(cms), figsize=(4.5 * len(cms), 4), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax,
                    xticklabels=["real", "fake"], yticklabels=["real", "fake"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(reports / "figures" / f"confusion_{args.variant}.png", bbox_inches="tight")

    print("\n", results.to_string(index=False))


if __name__ == "__main__":
    main()