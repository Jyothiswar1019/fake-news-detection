from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models(seed: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "PassiveAggressive": PassiveAggressiveClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=3, metric="cosine", algorithm="brute", n_jobs=-1),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
    }