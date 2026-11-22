import sys
from functools import lru_cache

import joblib

from fakenews.config import load_config, resolve

LABELS = {0: "real", 1: "fake"}
MIN_TOKENS = 15  # below this the cleaned text is too short for a reliable prediction


@lru_cache(maxsize=1)
def load_pipeline():
    cfg = load_config()
    path = resolve(cfg["paths"]["models_dir"]) / "fakenews_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: python -m fakenews.train_final")
    return joblib.load(path)


def predict_texts(texts: list[str]) -> list[dict]:
    pipe = load_pipeline()
    cleaned = pipe.named_steps["clean"].transform(texts)
    probs = pipe.named_steps["clf"].predict_proba(pipe.named_steps["vec"].transform(cleaned))
    fake_idx = list(pipe.classes_).index(1)

    results = []
    for clean, p in zip(cleaned, probs):
        p_fake = float(p[fake_idx])
        if p_fake >= 0.65:
            label = "fake"
        elif p_fake <= 0.35:
            label = "real"
        else:
            label = "uncertain"
        n_tokens = len(clean.split())
        results.append({
            "label": label,
            "fake_probability": round(p_fake, 4),
            "confidence": round(max(p_fake, 1 - p_fake), 4),
            "tokens_used": n_tokens,
            "warning": (f"Only {n_tokens} informative words after cleaning; "
                        "the prediction is unreliable.") if n_tokens < MIN_TOKENS else None,
        })
    return results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "The Senate voted on Tuesday to approve the budget after a long debate."
    print(predict_texts([text])[0])