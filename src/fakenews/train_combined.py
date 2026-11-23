"""Train the combined WELFake + IFND model and report per-source, mixed-site and cross-dataset results.

Usage:  python -m fakenews.train_combined            # uses data/processed/combined/
Output: models/fakenews_combined.joblib, models/metrics_combined.json,
        reports/combined_results.csv, reports/cv_results_combined.csv, reports/top_features_combined.csv
"""
import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from fakenews.config import load_config, resolve
from fakenews.features import build_vectorizer
from fakenews.preprocess import TextCleaner

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_combined")


def slice_metrics(name, y, pred, p) -> dict:
    return {
        "slice": name,
        "n": int(len(y)),
        "fake_share": round(float(np.mean(y)), 3),
        "predicted_fake": round(float(np.mean(pred)), 3),
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"),
    }


def mixed_sites(df, label, min_each=50) -> list:
    """IFND sites that publish BOTH classes: real-vs-fake there is not just publisher style."""
    ct = (df[df["source"] == "ifnd"].groupby(["site", label]).size()
          .unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0))
    return ct[(ct[0] >= min_each) & (ct[1] >= min_each)].index.tolist()


def fit_predict(train, test, cfg, label, params):
    vec = build_vectorizer(cfg)
    X = vec.fit_transform(train["clean_text"].fillna(""))
    clf = LogisticRegression(max_iter=2000, random_state=42, **params).fit(X, train[label])
    return clf.predict_proba(vec.transform(test["clean_text"].fillna("")))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="combined")
    ap.add_argument("--max-features", type=int, default=None, help="override the vectorizer size in the config")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _v2")
    args = ap.parse_args()

    cfg = load_config()
    if args.max_features:
        cfg["features"]["max_features"] = args.max_features
    pc, label = cfg["preprocess"], cfg["data"]["label_col"]
    ddir = resolve("data/processed") / args.variant
    train, test = pd.read_csv(ddir / "train.csv"), pd.read_csv(ddir / "test.csv")
    full = pd.concat([train, test], ignore_index=True)
    reports, models_dir = resolve(cfg["paths"]["reports_dir"]), resolve(cfg["paths"]["models_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. vectorize once (fitted on train only)
    vec = build_vectorizer(cfg)
    X_train = vec.fit_transform(train["clean_text"].fillna(""))
    X_test = vec.transform(test["clean_text"].fillna(""))
    y_train, y_test = train[label].to_numpy(), test[label].to_numpy()
    log.info("Feature matrix: train %s, test %s", X_train.shape, X_test.shape)

    # 2. tune C and class weights; CV folds keep near-duplicate groups together
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        param_grid={"C": [0.003, 0.01, 0.03, 0.1, 0.3], "class_weight": ["balanced"]},
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42),
        n_jobs=-1, return_train_score=True,
    )
    grid.fit(X_train, y_train, groups=train["group"])
    cv = pd.DataFrame(grid.cv_results_)[
        ["param_C", "param_class_weight", "mean_train_score", "mean_test_score", "std_test_score"]
    ].round(4)
    cv.to_csv(reports / f"cv_results_combined{args.tag}.csv", index=False)
    print("\nCross-validation (macro F1):\n", cv.to_string(index=False))
    best = grid.best_params_
    log.info("Best params: %s", best)
    clf = grid.best_estimator_

    # 3. one-time test evaluation, overall and by slice
    p = clf.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    rows = [slice_metrics("ALL test", y_test, pred, p)]
    for src in ("welfake", "ifnd"):
        m = (test["source"] == src).to_numpy()
        rows.append(slice_metrics(f"{src} test", y_test[m], pred[m], p[m]))
    sites = mixed_sites(full, label)
    m = ((test["source"] == "ifnd") & test["site"].isin(sites)).to_numpy()
    if m.sum():
        rows.append(slice_metrics("ifnd, mixed sites only", y_test[m], pred[m], p[m]))
    for s in sites:
        m = (test["site"] == s).to_numpy()
        if m.sum() >= 30:
            rows.append(slice_metrics(f"  site {s}", y_test[m], pred[m], p[m]))

    # 4. cross-dataset: train on one source, test on ALL rows of the other
    for a, b in (("welfake", "ifnd"), ("ifnd", "welfake")):
        pb = fit_predict(train[train["source"] == a], full[full["source"] == b], cfg, label, best)
        yb = full.loc[full["source"] == b, label].to_numpy()
        rows.append(slice_metrics(f"CROSS: train {a} -> test {b}", yb, (pb >= 0.5).astype(int), pb))

    res = pd.DataFrame(rows)
    res.to_csv(reports / f"combined_results{args.tag}.csv", index=False)
    print("\nResults by slice:\n", res.round(3).to_string(index=False))

    # 5. influential words
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top = pd.DataFrame({"pushes_to_REAL": vocab[order[:30]], "pushes_to_FAKE": vocab[order[::-1][:30]]})
    top.to_csv(reports / f"top_features_combined{args.tag}.csv", index=False)
    print("\nMost influential words:\n", top.to_string(index=False))

    # 6. deployable pipeline (raw text in); must reproduce the offline predictions
    pipe = Pipeline([
        ("clean", TextCleaner(strip_source_tags=pc["strip_source_tags"], min_token_len=pc["min_token_len"],
                              stem=pc["stem"], strip_dates=pc.get("strip_dates", True))),
        ("vec", vec),
        ("clf", clf),
    ])
    n = 300
    assert (pipe.predict(test["content"].head(n)) == pred[:n]).all(), "pipeline differs from offline predictions"
    joblib.dump(pipe, models_dir / f"fakenews_combined{args.tag}.joblib")

    overall = res.iloc[0]
    summary = {
        "model": "LogisticRegression", "variant": args.variant, "max_features": cfg["features"]["max_features"],
        "best_params": {k: (None if v is None else (float(v) if k == "C" else v)) for k, v in best.items()},
        "cv_f1_macro": float(grid.best_score_),
        "test_overall": {k: float(overall[k]) for k in ("accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc")},
        "mixed_sites": sites, "label_map": {"0": "real", "1": "fake"},
        "train_rows": int(len(train)), "test_rows": int(len(test)), "sklearn_version": sklearn.__version__,
    }
    (models_dir / f"metrics_combined{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved pipeline -> %s", models_dir / f"fakenews_combined{args.tag}.joblib")


if __name__ == "__main__":
    main()