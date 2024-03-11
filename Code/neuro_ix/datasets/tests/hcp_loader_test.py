import os
import pytest
from neuro_ix.datasets.hcp_loader import HCPYADataset, exclude_qc, extract_id_from_path
from neuro_ix.datasets.neuro_ix_dataset import MRIModality
from neuro_ix.utils.datasets import zip_dataset

DATASET_NAME = "HCP-YA-1200"
ROOT_DIR = "data-test"
HCPTEST_PATH = os.path.join(ROOT_DIR, DATASET_NAME)


def test_hcp_dataset_init():
    ds = HCPYADataset(ROOT_DIR)
    assert ds.dataset_std_name == DATASET_NAME
    assert os.path.samefile(ds.root_dir, HCPTEST_PATH)
    assert os.path.samefile(
        ds.participant_file_path, f"{HCPTEST_PATH}/participants.csv"
    )
    assert ds.has_participant_df()


def test_hcp_get_file_id():
    ds = HCPYADataset(ROOT_DIR)
    path = r"data-test/HCP-YA-1200/100206/unprocessed/3T/T1w_MPR1/100206_3T_T1w_MPR1.nii.gz"
    file = "100206_3T_T1w_MPR1"
    assert ds.get_file_id(path) == file


def test_hcp_get_participant_info():
    ds = HCPYADataset(ROOT_DIR)

    assert (
        ds.get_participant_info(
            r"data-test/HCP-YA-1200/100206/unprocessed/3T/T1w_MPR1/100206_3T_T1w_MPR1.nii.gz"
        )
        is not None
    )


def test_hcp_participant_warning():
    ds = HCPYADataset(ROOT_DIR)

    with pytest.warns(UserWarning):
        ds.get_participant_info("dfsdfbfd")


def test_get_images_path():
    ds = HCPYADataset(ROOT_DIR)

    assert len(ds.get_images_path(MRIModality.T1w)) > 0


def test_test_and_narval_config():
    assert os.path.normpath(HCPYADataset.test().root_dir) == os.path.normpath(
        HCPTEST_PATH
    )
    assert os.path.normpath(HCPYADataset.narval().root_dir) == os.path.normpath(
        f"/home/cbricout/projects/def-sbouix/data/{DATASET_NAME}"
    )


def test_exclude_qc():
    paths = [
        r"C:/Users/Brico/OneDrive/Code/Master/cinamon-cookie/data-test/HCP-YA-1200/1/unprocessed/3T/T1w_MPR1/1_3T_T1w_MPR1.nii.gz",
        r"C:/Users/Brico/OneDrive/Code/Master/cinamon-cookie/data-test/HCP-YA-1200/2/unprocessed/3T/T1w_MPR1/2_3T_T1w_MPR1.nii.gz",
        r"C:/Users/Brico/OneDrive/Code/Master/cinamon-cookie/data-test/HCP-YA-1200/3/unprocessed/3T/T1w_MPR1/3_3T_T1w_MPR1.nii.gz",
    ]
    with_qc = [1, 2]
    should_get = r"C:/Users/Brico/OneDrive/Code/Master/cinamon-cookie/data-test/HCP-YA-1200/3/unprocessed/3T/T1w_MPR1/3_3T_T1w_MPR1.nii.gz"
    excluded = exclude_qc(paths, with_qc)
    assert len(excluded) == 1
    assert excluded[0] == should_get


def test_zip_dataset_with_qc_issue():
    ds = HCPYADataset(ROOT_DIR)

    should_be_empty = zip_dataset(
        ds.get_images_path(modality=MRIModality.T1w, qc_issue=1)
    )
    should_be_one = list(
        zip_dataset(ds.get_images_path(modality=MRIModality.T1w, qc_issue=0))
    )[0]

    assert len(list(should_be_empty)) == 0
    assert len(list(should_be_one)) == 2
