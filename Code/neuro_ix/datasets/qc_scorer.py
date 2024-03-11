from abc import ABC, abstractmethod
from typing import Callable
import warnings
import pandas as pd


class QCScoreTrait(ABC):
    qc_scorer: Callable[[str], int]

    def qc_score(self, path):
        return self.qc_scorer(path)


class FileBasedQCScore:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_field: str,
        score_field: str,
        id_extractor: Callable[[str], str] = lambda x: x,
    ):
        self.score_df = dataframe
        self.id_field = id_field
        self.score_field = score_field
        self.id_extractor = id_extractor

    def __call__(self, raw_id):
        id = self.id_extractor(raw_id)
        if (self.score_df[self.id_field] == id).sum() > 0:
            return self.score_df[self.score_df[self.id_field] == id][
                self.score_field
            ].iloc[0]
        else:
            warnings.warn(f"this id ({id}) does not exist in dataframe", UserWarning)
            return -1
