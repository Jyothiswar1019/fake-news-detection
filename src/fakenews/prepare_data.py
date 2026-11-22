import json
import logging

from fakenews.config import load_config, resolve
from fakenews.data import clean_dataset, load_raw, validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    cfg = load_config()
    df = load_raw(cfg)

    report = {"raw": validate(df, cfg)}
    clean = clean_dataset(df, cfg)
    label = cfg["data"]["label_col"]
    report["clean"] = {
        "rows": int(len(clean)),
        "label_counts": {str(k): int(v) for k, v in clean[label].value_counts().items()},
    }

    out = resolve(cfg["paths"]["clean_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)

    rep = resolve(cfg["paths"]["reports_dir"]) / "data_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved clean data -> {out}")


if __name__ == "__main__":
    main()