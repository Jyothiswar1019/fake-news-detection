import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from fakenews.config import load_config, resolve
from fakenews.features import build_vectorizer
from fakenews.preprocess import TextCleaner
from fakenews.train_compare import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_final")


def main() -> None:
    cfg = load_config()
    pc, label = cfg["preprocess"], cfg["data"]["label_col"]
    data_dir = resolve("data/processed") / "stripped"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # 1. Vectorize once. Unsupervised and fitted on train only, so no test leakage.
    vec = build_vectorizer(cfg)
    X_train = vec.fit_transform(train["clean_text"])
    X_test = vec.transform(test["clean_text"])
    y_train, y_test = train[label], test[label]
    log.info("Feature matrix: train %s, test %s", X_train.shape, X_test.shape)

    # 2. Tune C by cross-validation on the training set only
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
                param_grid={"C": [0.003, 0.01, 0.03, 0.1, 0.3]},
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    cv_table = pd.DataFrame(grid.cv_results_)[
        ["param_C", "mean_train_score", "mean_test_score", "std_test_score"]
    ].round(4)
    print("\nCross-validation (F1):\n", cv_table.to_string(index=False))
    log.info("Best C = %s", grid.best_params_["C"])

    clf = grid.best_estimator_

    # 3. One-time evaluation on the held-out test set
    preds = clf.predict(X_test)
    metrics = evaluate(y_test, preds)
    print("\nTest metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("\n", classification_report(y_test, preds, target_names=["real (0)", "fake (1)"], digits=4))

    reports = resolve(cfg["paths"]["reports_dir"])
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cbar=False,
                xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Final model: Logistic Regression")
    plt.savefig(reports / "figures" / "confusion_final.png", bbox_inches="tight")

    # 4. Which words does the model rely on?
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top = pd.DataFrame({
        "pushes_to_REAL": vocab[order[:30]],
        "pushes_to_FAKE": vocab[order[::-1][:30]],
    })
    top.to_csv(reports / "top_features.csv", index=False)
    print("\nMost influential words:\n", top.to_string(index=False))

    # 5. Deployable pipeline: raw text in, prediction out
    pipe = Pipeline([
        ("clean", TextCleaner(strip_source_tags=pc["strip_source_tags"],
                              min_token_len=pc["min_token_len"],
                              stem=pc["stem"],
                              strip_dates=pc.get("strip_dates", True))),
        ("vec", vec),
        ("clf", clf),
    ])

    # Sanity check: the pipeline on RAW text must reproduce the offline predictions
    n = 300
    assert (pipe.predict(test["content"].head(n)) == preds[:n]).all(), \
        "Pipeline predictions differ from offline predictions"

    models_dir = resolve(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "fakenews_pipeline.joblib")

    summary = {
        "model": "LogisticRegression",
        "best_C": float(grid.best_params_["C"]),
        "cv_f1": float(grid.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "label_map": {"0": "real", "1": "fake"},
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sklearn_version": sklearn.__version__,
    }
    (models_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved pipeline -> %s", models_dir / "fakenews_pipeline.joblib")


if __name__ == "__main__":
    main()