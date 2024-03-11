import os
import pickle
import pandas as pd

from neuro_ix.datasets.neuro_ix_dataset import MRIModality
from neuro_ix.datasets.qc_scorer import QCScoreTrait, FileBasedQCScore
from neuro_ix.datasets.bids_loader import (
    BIDSDataset,
    extract_bids_name,
    load_T1w_from_dirs,
)


class MRArtDataset(BIDSDataset, QCScoreTrait):
    dataset_std_name = "MR-ART"

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.score_df = pd.read_csv(
            os.path.join(self.root_dir, "derivatives", "scores.tsv"), sep="\t"
        )
        self.qc_scorer = FileBasedQCScore(
            self.score_df, "bids_name", "score", extract_bids_name
        )

    @classmethod
    def get_filename_template(cls, modality: MRIModality, magnitude: int = -1):
        match modality:
            case MRIModality.T1w:
                match magnitude:
                    case 1:
                        return "*standard_T1w.nii*"
                    case 2:
                        return "*headmotion1_T1w.nii*"
                    case 3:
                        return "*headmotion2_T1w.nii*"
                    case _:
                        return BIDSDataset.get_filename_template(modality)

    def get_images_path(self, modality: MRIModality, magnitude: int = -1):
        """Accept -1, 0,1,2"""
        path = load_T1w_from_dirs(
            [self.root_dir], MRArtDataset.get_filename_template(modality, magnitude)
        )
        path.sort()
        return path


