
import os
from neuro_ix.datasets.mriqc_dataset import MRIQCDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality


def test_init_dataset():
    ds=MRIQCDataset.test()

    assert os.path.samefile(ds.rawdata_dir, "data-test/MRIQC")

def test_retrieve_qc():
    ds=MRIQCDataset.test()

    assert ds.qc_score(r"data-test\MRIQC\sub-AMPSCZ-BI02450\sub-BI02450_ses-202304111_rec-norm_run-1_T1w\mri\reg_extracted_orig_nu.nii.gz")==False
    assert ds.qc_score(r"data-test\MRIQC\sub-MR-ART-986786\sub-986786_acq-headmotion2_T1w\mri\reg_extracted_orig_nu.nii.gz")==True
    assert ds.qc_score(r"data-test\MRIQC\sub-HCP-YA-1200-117930\117930_3T_T1w_MPR1\mri\reg_extracted_orig_nu.nii.gz")==False


def test_get_image_path():
    ds=MRIQCDataset.test()
    assert len(ds.get_images_path(MRIModality.T1w))==6
    assert len(ds.get_images_path(MRIModality.T1w, qc_issue=False))==2
    assert len(ds.get_images_path(MRIModality.T1w, qc_issue=True))==4

def test_get_split_img_path():
    ds=MRIQCDataset.test()
    split= ds.get_split_img_path(modality=MRIModality.T1w, qc_issue=False, test_ratio=0.5)
    assert len(split[0])==1
    assert len(split[1])==1
