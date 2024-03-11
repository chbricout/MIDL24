import pandas as pd
from neuro_ix.datasets.qc_scorer import FileBasedQCScore


def test_file_based_qc_score():
    score_df = pd.DataFrame([["1", 1], ["2", 0]], columns=["id", "score"])
    scorer = FileBasedQCScore(score_df, "id", "score")
    assert scorer("1") == 1
    assert scorer("2") == 0
