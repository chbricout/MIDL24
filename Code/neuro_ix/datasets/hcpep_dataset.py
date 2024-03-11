import os
import re

import pandas as pd
from neuro_ix.datasets.bids_loader import BIDSDataset
from neuro_ix.datasets.qc_scorer import FileBasedQCScore, QCScoreTrait

def convert_to_float(score:str):
    score =score.replace(" ","").replace("?","")
    if "/" in score or "\\" in score:
        base = min(int(score[0]), int(score[2]))
        return base+0.5
    return float(score)
    
 
class HCPEPDataset(BIDSDataset, QCScoreTrait):
    dataset_std_name = "HCPEP"

    def __init__(self, root_dir, score_path=None):
        super().__init__(root_dir)
        self.root_dir = os.path.join(self.root_dir, "rawdata")
        self.participant_file_path = os.path.join(self.root_dir, "participants.tsv")
        self.participant_df = None
        if os.path.exists(self.participant_file_path):
            self.participant_df = pd.read_csv(self.participant_file_path, sep="\t")

        if score_path==None:
            score_path = os.path.join(self.root_dir, "scores.csv")
        self.score_df = pd.read_csv(score_path)
        self.score_df = self.score_df[self.score_df["T1w"].notna()]
        self.score_df["T1w"] = self.score_df["T1w"].apply(convert_to_float)

        self.qc_scorer=FileBasedQCScore(self.score_df, "Subject ID", "T1w",HCPEPDataset.get_qc_id)

    @staticmethod
    def get_qc_id(path):
        match_re = r".*sub-(\d+)_ses-(\d+)_.*"
        match_res = re.match(match_re, path)
        if match_res:
            sub_id = match_res.group(1)
            ses_id= match_res.group(2)
            return f"{sub_id}_MR{ses_id}"
        return -1