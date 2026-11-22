from sklearn.feature_extraction.text import CountVectorizer


def build_vectorizer(cfg: dict) -> CountVectorizer:
    f = cfg["features"]
    return CountVectorizer(
        max_features=f["max_features"],
        ngram_range=tuple(f["ngram_range"]),
        min_df=f["min_df"],
        max_df=f["max_df"],
    )