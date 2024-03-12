
import glob
import logging
import os
import pickle
import re
import pandas as pd
import neuro_ix.datasets.bids_loader as bids
import neuro_ix.datasets.hcp_loader as hcp
from neuro_ix.datasets.hcpep_dataset import HCPEPDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality, NeuroiXDataset
from neuro_ix.datasets.qc_scorer import FileBasedQCScore, QCScoreTrait

def extract_mrart_filename(path: str) -> str:
    match_re = r".*(sub-[^\.\\\/]*)[\\\/]mri.*"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return ''

class MRIQCDataset(NeuroiXDataset, QCScoreTrait):
    dataset_std_name = "MRIQC"

    def __init__(self, root_dir: str, score_path:str=None):
        super().__init__(root_dir)
        self.rawdata_dir =self.root_dir 
        self.participant_file=None
        if score_path == None:
            self.score_path = os.path.join(self.rawdata_dir,"derivatives", "scores.csv")
        else:
            self.score_path = score_path

        self.score_df = pd.read_csv(
           self.score_path, sep=",", index_col=0
        )

        self.qc_scorer = FileBasedQCScore(
            self.score_df, "bids_name", "qc_issue", self.get_score_id
        )
        self.dataset_score = FileBasedQCScore(
            self.score_df, "bids_name", "score", self.get_score_id
        )
        self.with_qc = self.score_df[self.score_df['qc_issue']==True]["bids_name"].to_list()
        self.without_qc = self.score_df[(self.score_df['qc_issue']==False) & ((self.score_df['dataset']!="AMPSCZ") | (self.score_df['score']>-1))]["bids_name"].to_list()
        self.has_qc = self.with_qc + self.without_qc

    def get_mrart_score(self, path):
        match_res = re.match(".*acq-([^\\._])_T1w.*", path)
        if match_res:
                mag =  match_res.group(1)
                if mag == 'headmotion1':
                    return 1
                elif mag == 'headmotion2':
                    return 2
                elif mag=="standard":
                    return 0
        return -1

    def get_dataset(self, path):
        if "HCP-YA-1200" in path:
            return "HCP-YA-1200"
        elif "MR-ART" in path:
            return "MR-ART"
        elif "AMPSCZ" in path:
            return "AMPSCZ"
        elif "HCPEP" in path:
            return "HCPEP"

    def extract_id(self, path):
        if "AMPSCZ" in path or "MR-ART" in path or "HCPEP" in path:
            return bids.extract_id(path)
        elif "HCP-YA-1200" in path:
            match_re = r".*sub-HCP-YA-1200-([0-9]*).*"
            match_res = re.match(match_re, path)
            if match_res:
                return match_res.group(1)
            return -1

    

    def get_subject_id(self, path):
        if "AMPSCZ" in path or "MR-ART" in path or "HCPEP" in path:
            return bids.extract_sub(path)
        else:
            return self.extract_id(path)

    def get_file_id(self, path):
        match_re = r".*[\\\/]([^\\\/]*)[\\\/]mri.*"
        match_res = re.match(match_re, path)
        if match_res:
            return match_res.group(1)
        return -1
    
    def get_score_id(self, path):
        if "AMPSCZ" in path:
            return bids.extract_sub(path) + "/" + bids.extract_ses(path)
        elif "MR-ART" in path:
            return extract_mrart_filename(path)
        elif  "HCPEP" in path:
            return HCPEPDataset.get_qc_id(path)
        else:
            return self.extract_id(path)


    @classmethod
    def get_filename_template(cls, modality: MRIModality, auxiliary=False):
        match modality:
            case MRIModality.T1w :
                return "*/mri/reg_extracted_orig_nu.nii.gz"

    def get_images_path(self, modality: MRIModality, qc_issue=None):
        path= glob.glob(
            os.path.join(self.rawdata_dir,"*", self.get_filename_template(modality))
        )
        logging.info("we got the paths, start filtering")
        if not qc_issue is None:
            if qc_issue == True:
                path = list(filter(lambda x:self.get_score_id(x) in self.with_qc, path))
            elif qc_issue == False:
                excluded =  list(filter(lambda x:self.get_score_id(x) in self.with_qc, path))
                path = list(set(path).difference(excluded))
        logging.info("filtered!")
        path.sort()

        return path
    
    def get_participant_info(self, path):
        return None
    
    @classmethod
    def narval(cls):
        return cls("/home/cbricout/scratch", score_path="data-test/MRIQC/derivatives/scores.csv")

    @classmethod
    def test(cls):
        return cls("data-test")

    @classmethod
    def lab(cls):
        return cls("/home/at70870/narval/scratch",  score_path="data-test/MRIQC/derivatives/scores.csv")
    
class MIDLMRArtDataset(MRIQCDataset):
    def __init__(self, root_dir: str, score_path:str=None):
        super().__init__(root_dir,score_path)
        with open("neuro_ix/datasets/midl", "rb") as data_file:
             self.train_set, self.val_set, self.test_set = pickle.load( data_file)

    
    def get_train_path(self):
        path = list(map(lambda x : os.path.join(self.root_dir, x), self.train_set))
        path.sort()
        return path
    
    def get_test_path(self):
        path = list(map(lambda x : os.path.join(self.root_dir, x), self.test_set))
        path.sort()
        return path
    
    def get_val_path(self):
        path = list(map(lambda x : os.path.join(self.root_dir, x), self.val_set))
        path.sort()
        return path
    
class MIDLAMPSCZDataset(MRIQCDataset):
    def __init__(self, root_dir: str, score_path:str=None, motion_score_path:str=None):
        super().__init__(root_dir,score_path)
        self.motion_score_df = pd.read_csv(
           motion_score_path, sep=",", index_col=0
        )
        self.motion_score = FileBasedQCScore(
            self.motion_score_df, "bids_name", "motion_mag", self.get_score_id
        )
        self.motion_label = FileBasedQCScore(
            self.motion_score_df, "bids_name", "motion", self.get_score_id
        )
    
    def get_images_path(self, modality: MRIModality, qc_issue=None):
        path= glob.glob(
            os.path.join(self.rawdata_dir,"sub-AMPSCZ-*", self.get_filename_template(modality))
        )
        logging.info("we got the paths, start filtering")
        if not qc_issue is None:
            if qc_issue == True:
                path = list(filter(lambda x:self.get_score_id(x) in self.with_qc, path))
            elif qc_issue == False:
                excluded =  list(filter(lambda x:self.get_score_id(x) in self.with_qc, path))
                path = list(set(path).difference(excluded))
        logging.info("filtered!")
        path.sort()

        return path

    @classmethod
    def narval(cls):
        return cls("/home/cbricout/scratch", score_path="data-test/MRIQC/derivatives/scores.csv", motion_score_path="data-test/AMPSCZ/rawdata/derivatives/motion_scores.csv")

    @classmethod
    def test(cls):
        return cls("data-test", motion_score_path="data-test/AMPSCZ/rawdata/derivatives/motion_scores.csv")

    @classmethod
    def lab(cls):
        return cls("/home/at70870/narval/scratch",  score_path="data-test/MRIQC/derivatives/scores.csv", motion_score_path="data-test/AMPSCZ/rawdata/derivatives/motion_scores.csv")