import joblib
import numpy as np
import pandas as pd

from fakenews.config import load_config, resolve


def main() -> None:
    cfg = load_config()
    label = cfg["data"]["label_col"]
    test = pd.read_csv(resolve("data/processed/stripped/test.csv"))
    pipe = joblib.load(resolve(cfg["paths"]["models_dir"]) / "fakenews_pipeline.joblib")

    p_fake = pipe.predict_proba(test["content"])[:, 1]
    y = test[label].to_numpy()
    pred = (p_fake >= 0.5).astype(int)

    conf = np.maximum(p_fake, 1 - p_fake)
    bands = pd.cut(conf, [0.5, 0.65, 0.8, 0.9, 1.0], include_lowest=True)
    df = pd.DataFrame({"band": bands, "correct": pred == y})
    out = df.groupby("band", observed=True)["correct"].agg(share_of_test="size", accuracy="mean")
    out["share_of_test"] = (out["share_of_test"] / len(df)).round(3)
    out["accuracy"] = out["accuracy"].round(4)
    print(out)

    keep = (p_fake >= 0.65) | (p_fake <= 0.35)
    print(f"\nWith the uncertain band: {keep.mean():.1%} of articles get a verdict, "
          f"accuracy on those = {(pred[keep] == y[keep]).mean():.4f}")


if __name__ == "__main__":
    main()