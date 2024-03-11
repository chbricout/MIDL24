import os

import pytest
from neuro_ix.datasets.bids_loader import (
    BIDSDataset,
    extract_id,
    extract_bids_name,
    has_session,
    get_file_mask,
)
from neuro_ix.datasets.neuro_ix_dataset import MRIModality

MRART_PATH = "data-test/MR-ART"


### TEST FOR : extract_id
def test_extract_id_json():
    idx = extract_id(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.json"
    )
    assert idx == 103


def test_extract_id_nii():
    idx = extract_id(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.nii"
    )
    assert idx == 103


def test_extract_id_nii_gz():
    idx = extract_id(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.nii.gz"
    )
    assert idx == 103


def test_extract_id_bids_name():
    idx = extract_id("sub-000103_acq-headmotion2_T1w")
    assert idx == 103


### TEST FOR : extract_bids_name


def test_extract_bids_name_json():
    bids_name = extract_bids_name(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.json"
    )
    assert bids_name == "sub-000103_acq-headmotion2_T1w"


def test_extract_bids_name_nii():
    bids_name = extract_bids_name(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.nii"
    )
    assert bids_name == "sub-000103_acq-headmotion2_T1w"


def test_extract_bids_name_nii_gz():
    bids_name = extract_bids_name(
        rf"{MRART_PATH}\sub-000103\anat\sub-000103_acq-headmotion2_T1w.nii.gz"
    )
    assert bids_name == "sub-000103_acq-headmotion2_T1w"


### TEST FOR: has_session
def test_has_session():
    should_have_session = has_session(r"data-test/test-mri-2")
    should_not_have_session = has_session(r"data-test/test-mri")
    assert should_have_session == True
    assert should_not_have_session == False


### TEST FOR : get_file_mask
def test_get_file_mask():
    assert get_file_mask(r"data-test/test-mri-2", "yeah") == f"sub-*/ses-*/anat/yeah"
    assert get_file_mask(r"data-test/test-mri", "yeah") == f"sub-*/anat/yeah"


def test_bids_dataset_init():
    ds = BIDSDataset("data-test", "MR-ART")
    assert ds.dataset_std_name == "MR-ART"
    assert os.path.samefile(ds.root_dir, MRART_PATH)
    assert os.path.samefile(ds.participant_file_path, f"{MRART_PATH}/participants.tsv")
    assert ds.has_participant_df()


def test_bids_get_participant_info():
    ds = BIDSDataset("data-test", "MR-ART")

    assert ds.get_participant_info("sub-000103_acq-standard_T1w") is not None
    assert (
        ds.get_participant_info(
            rf"{MRART_PATH}\sub-988484\anat\sub-988484_acq-headmotion1_T1w.json"
        )
        is not None
    )


def test_bids_participant_warning():
    ds = BIDSDataset("data-test", "MR-ART")

    with pytest.warns(UserWarning):
        ds.get_participant_info("dfsdfbfd")


def test_filename_template():
    assert BIDSDataset.get_filename_template(MRIModality.T1w) == "*T1w.nii*"


def test_get_images_path():
    ds = BIDSDataset("data-test", "MR-ART")

    assert len(ds.get_images_path(MRIModality.T1w)) > 1
