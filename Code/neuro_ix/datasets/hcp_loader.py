import logging
from typing import List
import glob
import os
import re
import warnings
import pandas as pd

from neuro_ix.datasets.neuro_ix_dataset import MRIModality, NeuroiXDataset
from neuro_ix.datasets.qc_scorer import QCScoreTrait


class HCPDataset(NeuroiXDataset, QCScoreTrait):
    participant_file = "participants.csv"
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.with_qc = None
        
        self.participant_file_path=os.path.join(self.root_dir, self.participant_file)
        if os.path.exists(self.participant_file_path):
            self.participant_df = pd.read_csv(self.participant_file_path)
            pretreat_subject_df(self.participant_df)
            self.with_qc = self.participant_df[self.participant_df["A_Error"] == True][
                "Subject"
            ].to_list()

    def extract_id(self, path):
        return extract_id_from_path(path)

    def get_subject_id(self, path):
        return self.extract_id(path)

    def get_file_id(self, path):
        # One file by subject
        return extract_file_from_path(path)

    @classmethod
    def get_filename_template(cls, modality: MRIModality):
        match modality:
            case MRIModality.T1w:
                    return "*/unprocessed/3T/T1w_MPR1/*_T1w_MPR1.nii.gz"
            case MRIModality.T2w:
                    return "*/unprocessed/3T/T2w_SPC1/*_T2w_SPC1.nii.gz"


    def get_images_path(self, modality: MRIModality, qc_issue=-1):
        path = glob.glob(
            os.path.join(self.root_dir, self.get_filename_template(modality))
        )
        logging.info(f"Images in dataset : {len(path)}")
        if qc_issue == 0:
            path = exclude_qc(path, self.with_qc)
        elif qc_issue == 1:
            excluded = exclude_qc(path, self.with_qc)
            path = list(set(path).difference(excluded))
        logging.info(f"Images in dataset : {len(path)} (qc_issue : {qc_issue})")
        path.sort()
        return path

    def get_participant_info(self, path):
        if self.has_participant_df():
            part_id = extract_id_from_path(path)
            part_extract = self.participant_df["Subject"] == part_id
            if part_extract.sum() == 1:
                return self.participant_df[part_extract]
            else:
                warnings.warn(
                    f"participant id ({part_id}) do not exist or is not unique",
                    UserWarning,
                )
        else:
            warnings.warn(
                "Trying to retrieve participants in a dataset without participants.tsv !",
                UserWarning,
            )
            return None


class HCPYADataset(HCPDataset):
    dataset_std_name = "HCP-YA-1200"

    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.participant_file_path = os.path.join(self.root_dir, "participants.csv")
        self.participant_df = None
        if os.path.exists(self.participant_file_path):
            self.participant_df = pd.read_csv(self.participant_file_path)




def exclude_qc(files: List[str], with_qc: List[str]):
    excluded = list(filter(lambda x: not extract_id_from_path(x) in with_qc, files))
    return excluded






def pretreat_subject_df(df: pd.DataFrame) -> pd.DataFrame:
    df["QC_Issue_Filled"] = df["QC_Issue"].fillna("F")
    df["Relevant_QC"] = df["QC_Issue_Filled"].str.contains(pat="[ABC]", regex=True)
    df["A_Error"] = df["QC_Issue_Filled"].str.contains(pat="[A]", regex=True)
    df["B_Error"] = df["QC_Issue_Filled"].str.contains(pat="[B]", regex=True)
    df["C_Error"] = df["QC_Issue_Filled"].str.contains(pat="[C]", regex=True)


def extract_id_from_path(path: str) -> int:
    match_re = r".*\/HCP-YA-1200\/(\d+)\/.*"
    match_res = re.match(match_re, path)
    if match_res:
        return int(match_res.group(1))
    return -1


def extract_file_from_path(path: str) -> str:
    match_re = r".*\/(.+)\.nii\.gz"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return -1
