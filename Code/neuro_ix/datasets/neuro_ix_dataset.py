from abc import ABC, abstractmethod
import os
from enum import Enum
import warnings

from sklearn.model_selection import train_test_split


class MRIModality(Enum):
    T1w = "T1-weighted"
    T2w = "T2-weighted"
    T2star = "T2*-weighted"
    FLAIR = "FLAIR"
    DWI = "Diffusion-weighted imaging"
    rsFMRI = "Resting-state fMRI"
    taskFMRI = "Task-based fMRI"
    fieldMap = "Field maps"


class QCFilter(Enum):
    All = "All"
    GoodOnly = "Good only"
    BadOnly = "Bad only"


class NeuroiXDataset(ABC):
    dataset_std_name: str

    def __init__(self, root_dir: str):
        self.root_dir = os.path.join(root_dir, self.dataset_std_name)
        self.rawdata_dir = self.root_dir
        self.participant_df = None
        pass

    def get_split_img_path(self, test_ratio: float, modality: MRIModality, **kwargs):
        paths = self.get_images_path(modality, **kwargs)
        train, test = train_test_split(paths, test_size=test_ratio)
        return train, test

    @abstractmethod
    def get_images_path(self, modality: MRIModality, **kwargs):
        pass

    @abstractmethod
    def extract_id(self, path):
        pass

    @abstractmethod
    def get_subject_id(self, path):
        pass

    @abstractmethod
    def get_file_id(self, path):
        pass

    def has_participant_df(self):
        return self.participant_df is not None

    def get_praticipant_df(self):
        if not self.has_participant_df:
            warnings.warn(
                "Trying to retrieve participants in a dataset without participants.tsv !",
                UserWarning,
            )

        return self.participant_df

    @abstractmethod
    def get_participant_info(self, path):
        pass

    @classmethod
    def narval(cls):
        return cls("/home/cbricout/projects/def-sbouix/data")

    @classmethod
    def lab(cls):
        return cls("/home/at70870/narval/data")

    @classmethod
    def test(cls):
        # TODO : Change this when lab is done
        return cls("data-test")
