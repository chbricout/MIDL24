from typing import Dict, List
from monai.data import IterableDataset
from monai.transforms import Identity

from neuro_ix.datasets.neuro_ix_dataset import MRIModality, NeuroiXDataset


def zip_dataset(paths, transform=Identity(), dataset_type=IterableDataset):
    return zip(paths, dataset_type(paths, transform=transform))


def get_dataset_iterator(
    dataset: NeuroiXDataset,
    modality: MRIModality,
    options: Dict[str, List[int]],
    transform=Identity(),
    dataset_type=IterableDataset,
):
    for key, iter_on in options.items():
        for value in iter_on:
            yield value, zip_dataset(
                dataset.get_images_path(modality=modality, **{key: value}),
                transform=transform,
                dataset_type=dataset_type,
            )

def to_bids_dataset_format(sub_id:str, dataset_name:str):
    '''Convert a sub_id name to a bids compliant name with dataset provenance informations'''
    sub_id_place = sub_id.find("sub-")
    if sub_id_place == -1:
        return f"sub-{dataset_name}-{sub_id}"
    else:
        pref = f"sub-{dataset_name}-"
        return pref + sub_id[sub_id_place+4:]