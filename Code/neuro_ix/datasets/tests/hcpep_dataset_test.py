import os
from neuro_ix.datasets.hcpep_dataset import HCPEPDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality

DATASET_NAME = "HCPEP"
ROOT_DIR = "data-test"
HCPEPTEST_PATH = os.path.join(ROOT_DIR, DATASET_NAME, "rawdata")
ds = HCPEPDataset.test()

def test_hcpep_dataset_init():
    assert ds.dataset_std_name == DATASET_NAME
    assert os.path.samefile(ds.root_dir, HCPEPTEST_PATH)
    assert os.path.samefile(
        ds.participant_file_path, f"{HCPEPTEST_PATH}/participants.tsv"
    )
    assert ds.has_participant_df()

def test_hcpep_qc_id():
    assert ds.get_qc_id("data-test/HCPEP/rawdata/sub-1001/ses-1/anat/sub-1001_ses-1_T1w.nii.gz") == "1001_MR1"
    assert ds.get_qc_id("data-test/HCPEP/rawdata/sub-1001/ses-2/anat/sub-1001_ses-2_T1w.nii.gz") == "1001_MR2"

def test_hcpep_qc_score():
    assert ds.qc_score("data-test/HCPEP/rawdata/sub-1001/ses-1/anat/sub-1001_ses-1_T1w.nii.gz") == 3
    assert ds.qc_score("data-test/HCPEP/rawdata/sub-4045/ses-1/anat/sub-4045_ses-1_T1w.nii.gz") == 3

def test_hcpep_get_file():
    assert len(ds.get_images_path(MRIModality.T1w))==1