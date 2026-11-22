import joblib
import pytest

from fakenews.config import resolve

PATH = resolve("models/fakenews_pipeline.joblib")


@pytest.mark.skipif(not PATH.exists(), reason="train the model first (python -m fakenews.train_final)")
def test_saved_pipeline_predicts_on_raw_text():
    pipe = joblib.load(PATH)
    preds = pipe.predict(["Senate passes budget bill after lengthy debate", ""])
    assert len(preds) == 2
    assert set(preds) <= {0, 1}