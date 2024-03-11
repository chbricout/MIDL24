import os
from neuro_ix.datasets.mr_art import MRArtDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality
from neuro_ix.utils.datasets import get_dataset_iterator


def test_dataset_init():
    dataset = MRArtDataset("data-test")
    assert dataset.dataset_std_name == "MR-ART"
    assert os.path.samefile(dataset.root_dir, "data-test/MR-ART")


def test_dataset_loading():
    ## WARNING: THIS WILL ONLY WORK IF YOU HAVE THE DATASET !!!
    dataset = MRArtDataset("data-test")
    assert len(dataset.get_images_path(MRIModality.T1w)) > 0
    assert len(dataset.get_images_path(MRIModality.T1w, 1)) > 0
    assert len(dataset.get_images_path(MRIModality.T1w, 2)) > 0
    assert len(dataset.get_images_path(MRIModality.T1w, 3)) > 0


def test_qc_score():
    dataset = MRArtDataset("data-test")
    assert dataset.qc_score("sub-000103_acq-standard_T1w") == 1
    assert (
        dataset.qc_score(
            r"data-test\MR-ART\sub-000148\anat\sub-000148_acq-headmotion1_T1w.nii.gz"
        )
        == 3
    )


def test_dataset_iterator():
    ## WARNING: THIS WILL ONLY WORK IF YOU HAVE THE DATASET !!!
    dataset = MRArtDataset("data-test")
    iterator = list(
        get_dataset_iterator(
            dataset=dataset, modality=MRIModality.T1w, options={"magnitude": [1, 2, 3]}
        )
    )
    assert len(iterator) == 3
    first_el = iterator[0]
    assert len(first_el) == 2
    assert first_el[0] == 1
    paths, ds = next(first_el[1])
    assert len(ds) > 1
    assert len(paths) == len(ds)
