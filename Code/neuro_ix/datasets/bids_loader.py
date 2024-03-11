import glob
import os
import re
from typing import List
import warnings

import pandas as pd

from neuro_ix.datasets.neuro_ix_dataset import MRIModality, NeuroiXDataset


def load_T1w_from_dirs(dirs: List[str], file_templ: str = "*T1w.nii*") -> List[str]:
    images = []

    for root_dir in dirs:
        mask = get_file_mask(root_dir, file_templ)
        
        images += glob.glob(os.path.join(root_dir, mask))
    return images


def extract_id(path: str) -> int:
    match_re = ".*sub-([0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    if match_res:
        return int(match_res.group(1))
    return -1


def extract_sub(path: str):
    match_re = ".*(sub-[0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return ""

def extract_ses(path: str):
    match_re = ".*(ses-[0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return ""

def extract_bids_name(path: str) -> str:
    match_re = r".*(sub-[^\.\\/]*)(\.nii|\.json|\.nii\.gz)?[^\\/]*$"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return -1


def has_session(root_dir: str) -> bool:
    return len(glob.glob(os.path.join(root_dir, "sub-*/ses-*"))) != 0


def get_file_mask(root_dir: str, file_templ: str) -> str:
    mask = ""
    if has_session(root_dir):
        mask = f"sub-*/ses-*/anat/{file_templ}"
    else:
        mask = f"sub-*/anat/{file_templ}"
    return mask


class BIDSDataset(NeuroiXDataset):
    participant_file = "participants.tsv"

    def __init__(self, root_dir, dataset_std_name=None):
        if dataset_std_name != None:
            self.dataset_std_name = dataset_std_name
        super().__init__(root_dir)
        self.participant_file_path=os.path.join(self.root_dir, self.participant_file)
        if os.path.exists(self.participant_file_path):
            self.participant_df = pd.read_csv(self.participant_file_path, sep="\t")

    def extract_id(self, path):
        return extract_id(path)

    def extract_bids_name(self, path):
        return extract_bids_name(path)

    def extract_sub(self, path):
        return extract_sub(path)

    def get_subject_id(self, path):
        return self.extract_sub(path)

    def get_file_id(self, path):
        return self.extract_bids_name(path)

    def get_participant_info(self, path):
        if self.has_participant_df():
            part_id = extract_sub(path)
            part_extract = self.participant_df["participant_id"] == part_id
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

    @classmethod
    def get_filename_template(cls, modality: MRIModality, **kwargs):
        match modality:
            case MRIModality.T1w:
                return "*T1w.nii*"
            case MRIModality.T2w:
                return "*T2w.nii*"
            case _:
                return "*"

    def get_images_path(self, modality: MRIModality, **kwargs):
        path = load_T1w_from_dirs(
            [self.root_dir], self.get_filename_template(modality, **kwargs)
        )
        path.sort()
        return path
