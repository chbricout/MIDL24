import os
import pandas as pd
from neuro_ix.datasets import bids_loader
from neuro_ix.datasets.bids_loader import BIDSDataset, extract_bids_name, extract_ses, extract_sub, load_T1w_from_dirs
from neuro_ix.datasets.neuro_ix_dataset import MRIModality
from neuro_ix.datasets.qc_scorer import FileBasedQCScore, QCScoreTrait


class AMPSCZDataset(BIDSDataset, QCScoreTrait):
    dataset_std_name = "AMPSCZ"

    def __init__(self, root_dir: str, score_path:str=None):
        super().__init__(root_dir)
        self.rawdata_dir =os.path.join(self.root_dir, "rawdata") 
        self.score_path = os.path.join(self.rawdata_dir,"derivatives", "scores.tsv")
        if score_path == None:
            self.score_path = os.path.join(self.rawdata_dir,"derivatives", "scores.tsv")
        else:
            self.score_path = score_path

        self.score_df = pd.read_csv(
           self.score_path, sep="\t"
        )

        self.score_df["score"] = self.score_df["T1w"].replace('-',-1).fillna(-1).astype(int)
        self.qc_scorer = FileBasedQCScore(
            self.score_df, "sub-*/ses-*_gs", "score", lambda x : extract_sub(x) + "/" + extract_ses(x)
        )
        self.has_score = self.score_df[self.score_df['score']>-1]["sub-*/ses-*_gs"].to_list()
    
    def get_score_id(self, path):
        return bids_loader.extract_sub(path) + "/" + bids_loader.extract_ses(path)

    @classmethod
    def get_filename_template(cls, modality: MRIModality, auxiliary=False):
        match modality:
            case MRIModality.T1w :
                if auxiliary:
                    return "*_T1w_auxiliary.nii.gz"
                return "*_T1w.nii.gz"
            case MRIModality.T2w :
                if auxiliary:
                    return "*_T2w_auxiliary.nii.gz"
                return "*_T2w.nii.gz"

    def get_images_path(self, modality: MRIModality, auxiliary=False):
        path= load_T1w_from_dirs(
            [self.rawdata_dir], AMPSCZDataset.get_filename_template(modality, auxiliary=auxiliary)
        )
        path = list(filter(lambda x:self.get_score_id(x) in self.has_score, path))
        path.sort()
        return path
    
    @classmethod
    def narval(cls):
        return cls("/home/cbricout/projects/def-sbouix/data", score_path="data-test/AMPSCZ/rawdata/derivatives/scores.tsv")
