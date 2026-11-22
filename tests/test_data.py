import pandas as pd

from fakenews.data import clean_dataset

CFG = {"data": {"text_col": "text", "title_col": "title", "label_col": "label", "use_title": False}}


def test_clean_dataset_removes_empty_duplicates_and_conflicts():
    df = pd.DataFrame({
        "title": ["a", "b", "c", "d", "e"],
        "text": ["hello world", "hello world", None, "same", "same"],
        "label": [0, 0, 1, 0, 1],
    })
    out = clean_dataset(df, CFG)
    assert list(out["content"]) == ["hello world"]  # duplicate, null and conflicting rows gone
    assert set(out.columns) == {"content", "label"}