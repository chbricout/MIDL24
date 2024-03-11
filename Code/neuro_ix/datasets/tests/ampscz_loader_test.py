
import os
from neuro_ix.datasets.ampscz import AMPSCZDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality


def test_dataset_init():
    dataset = AMPSCZDataset("data-test")
    assert dataset.dataset_std_name == "AMPSCZ"
    assert os.path.samefile(dataset.root_dir, "data-test/AMPSCZ")
    assert os.path.samefile(dataset.rawdata_dir, "data-test/AMPSCZ/rawdata")



def test_dataset_loading():
    ## WARNING: THIS WILL ONLY WORK IF YOU HAVE THE DATASET !!!
    dataset = AMPSCZDataset("data-test")
    assert len(dataset.get_images_path(MRIModality.T1w)) > 0

def test_qc_score():
    dataset = AMPSCZDataset("data-test")
    assert dataset.qc_score("sub-BI02450/ses-202306231") == 2
    assert (
        dataset.qc_score(
            r"data-test\AMPSCZ\sub-BI04369\ses-202312081\anat\sub-BI04369_ses-202312081_rec-norm_run-1_T1w.nii.gz"
        )
        == -1
    )

def test_extract_id_bids_name():
    dataset = AMPSCZDataset("data-test")

    idx = dataset.extract_bids_name("data-test\AMPSCZ\rawdata\sub-BI02450\ses-202304111\anat\sub-BI02450_ses-202304111_rec-norm_run-1_T1w.json")
    assert idx == "sub-BI02450_ses-202304111_rec-norm_run-1_T1w"