"""Build evaluation plots, metric tables and reports/REPORT.md for the saved model.

Usage:
    python -m fakenews.report                    # uses data/processed/stripped/test.csv
    python -m fakenews.report --variant combined # another processed variant
"""
import argparse
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from fakenews.config import load_config, resolve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("report")

sns.set_style("whitegrid")
CLASSES = ["real", "fake"]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_image(df: pd.DataFrame, title: str, path, digits: int = 4):
    cells = [[f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
             for r in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(1.6 * len(df.columns), 0.6 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cells, colLabels=list(df.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8ecf4")
    save(fig, path)


def plot_confusion(y, pred, fig_dir):
    cm = confusion_matrix(y, pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Confusion matrix (counts)")
    sns.heatmap(cmn, annot=True, fmt=".1%", cmap="Blues", cbar=False, vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title("Confusion matrix (share of each actual class)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    save(fig, fig_dir / "confusion_matrix_final.png")


def plot_curves(y, p, fig_dir):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, label=f"Average precision = {ap:.3f}")
    axes[1].axhline(float(np.mean(y)), ls="--", color="grey", label=f"Baseline = {np.mean(y):.2f}")
    axes[1].set(xlabel="Recall (fake)", ylabel="Precision (fake)", title="Precision-recall curve")
    axes[1].legend(loc="lower left")
    save(fig, fig_dir / "roc_pr_curves.png")
    return auc, ap


def confidence_table(y, p):
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    bands = pd.cut(conf, [0.5, 0.65, 0.8, 0.9, 1.0], include_lowest=True,
                   labels=["50-65%", "65-80%", "80-90%", "90-100%"])
    df = pd.DataFrame({"band": bands, "correct": pred == y})
    g = df.groupby("band", observed=True)["correct"].agg(articles="size", accuracy="mean")
    g["share_of_test"] = g["articles"] / len(df)
    g = g.reset_index()
    g["band"] = g["band"].astype(str)
    keep = (p >= 0.65) | (p <= 0.35)
    verdict = {"coverage": float(keep.mean()), "accuracy": float((pred[keep] == y[keep]).mean())}
    return g, verdict


def plot_confidence(g, fig_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=g, x="band", y="accuracy", color="#4c78a8", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by confidence band (axis starts at 0.5)")
    for i, r in g.iterrows():
        ax.text(i, r["accuracy"] + 0.005, f"{r['accuracy']:.1%}\n({r['share_of_test']:.0%} of articles)",
                ha="center", va="bottom", fontsize=9)
    save(fig, fig_dir / "confidence_bands.png")


def plot_model_comparison(rep_dir, fig_dir):
    f = rep_dir / "model_comparison_stripped.csv"
    if not f.exists():
        return None, None
    df = pd.read_csv(f)
    long = df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                   var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model comparison, default settings before tuning (axis starts at 0.5)")
    ax.set_xlabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, title=None, frameon=False)
    save(fig, fig_dir / "model_comparison.png")

    leak = None
    w = rep_dir / "model_comparison_with_tags.csv"
    if len(df) < 4:
        log.warning("model_comparison_stripped.csv has only %d model(s): %s. Restore all four rows.",
                    len(df), ", ".join(df["model"]))
    if w.exists():
        a = df.assign(variant="Source tags removed")
        b = pd.read_csv(w).assign(variant="Source tags kept")
        shared = set(a["model"]) & set(b["model"])
        if not shared:
            log.warning("The stripped and with_tags comparison files share no models; skipping the leakage chart.")
            return df, None
        both = pd.concat([a, b])
        both = both[both["model"].isin(shared)]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(data=both, x="model", y="accuracy", hue="variant", ax=ax)
        ax.set_ylim(0.8, 1.0)
        ax.set_title("Effect of removing source tags (axis starts at 0.8)")
        ax.set_xlabel("")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
        if ax.get_legend() is not None:
            sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, title=None, frameon=False)
        save(fig, fig_dir / "leakage_effect.png")
        leak = both[["model", "variant", "accuracy", "f1"]].reset_index(drop=True)
    return df, leak


def plot_top_features(vec, clf, fig_dir, n=20):
    vocab = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    real_idx, fake_idx = order[:n][::-1], order[::-1][:n][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    axes[0].barh(vocab[real_idx], -coef[real_idx], color="#1a7f4b")
    axes[0].set_title(f"Top {n} words pushing toward REAL")
    axes[1].barh(vocab[fake_idx], coef[fake_idx], color="#c0392b")
    axes[1].set_title(f"Top {n} words pushing toward FAKE")
    for ax in axes:
        ax.set_xlabel("Coefficient size")
    save(fig, fig_dir / "top_features.png")


def plot_cv(rep_dir, fig_dir):
    f = rep_dir / "cv_results.csv"
    if not f.exists():
        return False
    df = pd.read_csv(f)
    best = df.loc[df["mean_test_score"].idxmax(), "param_C"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(df["param_C"], df["mean_test_score"], yerr=df["std_test_score"],
                marker="o", capsize=3, label="Validation (5-fold CV)")
    ax.plot(df["param_C"], df["mean_train_score"], marker="o", label="Training")
    ax.axvline(best, ls="--", color="grey", label=f"Best C = {best:g}")
    ax.set_xscale("log")
    ax.set(xlabel="C (smaller = stronger regularization)", ylabel="F1",
           title="Tuning Logistic Regression")
    ax.legend()
    save(fig, fig_dir / "cv_tuning.png")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stripped", help="folder under data/processed/")
    args = ap.parse_args()

    cfg = load_config()
    label = cfg["data"]["label_col"]
    rep = resolve(cfg["paths"]["reports_dir"])
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(resolve("data/processed") / args.variant / "test.csv")
    pipe = joblib.load(resolve(cfg["paths"]["models_dir"]) / "fakenews_pipeline.joblib")
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]

    X = vec.transform(test["clean_text"].fillna(""))
    y = test[label].to_numpy()
    p = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    pred = (p >= 0.5).astype(int)
    log.info("Evaluating on %d test articles (variant=%s)", len(test), args.variant)

    plot_confusion(y, pred, fig_dir)
    auc, ap_score = plot_curves(y, p, fig_dir)
    bands, verdict = confidence_table(y, p)
    plot_confidence(bands, fig_dir)
    comparison, leak = plot_model_comparison(rep, fig_dir)
    plot_top_features(vec, clf, fig_dir)
    has_cv = plot_cv(rep, fig_dir)

    summary = pd.DataFrame([{
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": auc, "avg_precision": ap_score,
    }])
    per_class = pd.DataFrame(
        classification_report(y, pred, target_names=CLASSES, output_dict=True)
    ).T.reset_index().rename(columns={"index": "class"})

    summary.to_csv(rep / "metrics_summary.csv", index=False)
    per_class.to_csv(rep / "classification_report.csv", index=False)
    bands.to_csv(rep / "confidence_bands.csv", index=False)
    table_image(summary.round(4), "Final model: test-set metrics (fake = positive class)",
                fig_dir / "table_metrics.png")
    table_image(per_class.round(4), "Per-class report", fig_dir / "table_per_class.png")
    if comparison is not None:
        table_image(comparison.round(4), "Model comparison (same split)", fig_dir / "table_models.png")

    lines = [
        "# Evaluation report", "",
        f"Test set: {len(test):,} articles (variant `{args.variant}`). Positive class = fake.", "",
        "## Final model metrics", "", md_table(summary), "",
        "## Per-class report", "", md_table(per_class), "",
        "![Confusion matrix](figures/confusion_matrix_final.png)", "",
        "![ROC and precision-recall](figures/roc_pr_curves.png)", "",
        "## Accuracy by confidence", "", md_table(bands), "",
        f"Giving a verdict only when the fake probability is <= 35% or >= 65% covers "
        f"{verdict['coverage']:.1%} of articles at {verdict['accuracy']:.2%} accuracy.", "",
        "![Confidence bands](figures/confidence_bands.png)", "",
    ]
    if comparison is not None:
        lines += ["## Model comparison", "", md_table(comparison), "",
                  "![Model comparison](figures/model_comparison.png)", ""]
    if leak is not None:
        lines += ["## Effect of source tags", "", md_table(leak), "",
                  "![Leakage effect](figures/leakage_effect.png)", ""]
    if has_cv:
        lines += ["## Hyperparameter tuning", "", "![CV tuning](figures/cv_tuning.png)", ""]
    lines += ["## Most influential words", "", "![Top features](figures/top_features.png)", ""]
    (rep / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", rep / "REPORT.md")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()