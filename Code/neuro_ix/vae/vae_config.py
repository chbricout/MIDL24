import logging
import os
from typing import List

from sklearn.model_selection import train_test_split

from neuro_ix.datasets.mriqc_dataset import MRIQCDataset, MIDLMRArtDataset
from neuro_ix.datasets.neuro_ix_dataset import MRIModality
from neuro_ix.vae.argparser import create_arg_parser

import comet_ml
from neuro_ix.utils.slurm import get_parameters
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    Orientation,
    CropForeground,
    Resize,
    ScaleIntensity,
    Transform,
    SpatialPad
)

LEARNING_RATE_TO_TRY = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]
BETA_TO_TRY = [1, 10, 50, 100, 150, 200]
LATENT_SIZE_TO_TRY = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BATCH_SIZE_TO_TRY = [4, 8, 16, 32]


def threshold_one(x):
    return x >= 0.01



class VAETrainConfig:
    def __init__(
        self,
        max_epochs=300,
        learning_rate=1e-4,
        batch_train=4,
        is_test=False,
        array_id=None,
        exclude_qc=False,
        modality=MRIModality.T1w,
        project_dataset=False,
        use_decoder=True,
        beta=0.1
    ):
        self.conv_k=5
        self.beta=beta
        self.simple=True
        self.act="PRELU"
        self.T=50
        self.L=10
        self.use_decoder=use_decoder
        self.experiment = comet_ml.Experiment(
            api_key="WmA69YL7Rj2AfKqwILBjhJM3k", project_name="brain-vae"
        )
        self.exclude_qc = exclude_qc
        self.array_id = array_id
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_train
        self.is_test = is_test
        self.project_dataset = project_dataset

        self.modality = modality

        # Define datasets
        self.main_dataset = MRIQCDataset.test() if is_test else MIDLMRArtDataset.narval()
        if is_test:
            self.test_frac = 0.5
        else:
            self.test_frac = 0.2

        logging.info("asking for images")
        if is_test:
            paths = self.main_dataset.get_images_path(
                modality=self.modality,
                qc_issue=False if exclude_qc else None,
            )
        else:
            paths = self.main_dataset.get_images_path()

        # paths=  list(filter(lambda x: self.main_dataset.get_dataset(x)=="MR-ART", paths))


        # paths=  list(filter(lambda x : self.main_dataset.get_score_id(x) in self.main_dataset.has_qc, paths))
        subjects=set(map(lambda x:self.main_dataset.get_subject_id(x), paths))
        self.train_subjects, self.val_subjects = train_test_split(list(subjects), test_size = self.test_frac)
        scores = list(map(lambda x: int(self.main_dataset.dataset_score(x))-1, paths))
        self.train_path = list(filter(lambda x : self.main_dataset.get_subject_id(x) in self.train_subjects, paths))
        self.train_scores = list(map(lambda x: int(self.main_dataset.dataset_score(x))-1, self.train_path))
        self.val_path = list(filter(lambda x : self.main_dataset.get_subject_id(x) in self.val_subjects, paths))
        self.val_scores = list(map(lambda x: int(self.main_dataset.dataset_score(x))-1, self.val_path))




        logging.info(f"sorted train : {self.train_subjects}")
        logging.info(f"sorted val : {self.val_subjects}")
        logging.info(f"score val : {self.val_scores}")


        logging.info("images path retrieved!")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # VAE constructor needs image shape
        self.im_shape = (1, 160, 192, 160) if self.is_test else (1, 160, 192, 160)
        self.out_channels = 1
        self.transform = Compose(
            [
                LoadImage(ensure_channel_first=True, image_only=True),
                Orientation(axcodes="RAS"),
                ScaleIntensity(0, 1),
                CropForeground(threshold_one, allow_smaller=True),
                Resize(self.im_shape[1:]),
            ]
        )

        logging.info("Begin to save file...")
        if not os.path.exists("runs"):
            logging.info("Creating global runs directory")
            os.makedirs("runs")

        if self.array_id == None:
            existing_runs = os.listdir("runs")
            self.run_name = f"run-{len(existing_runs)+1}-{self.experiment.id}"
        else:
            self.run_name = f"run-array-{self.array_id}-{self.experiment.id}"
        self.run_dir = os.path.join("runs", self.run_name)
        self.best_model_path = f"runs/{self.run_name}/best_run"
        logging.info(f"Creating '{self.run_name}' directory")
        os.makedirs(f"runs/{self.run_name}")
        self.to_comet()

    @staticmethod
    def fromArrayId(
        id: int,
    ):
        conf = VAETrainConfig.fromArgs
        conv_k, act, simple = get_parameters(
            id,
            [3,5],
            ["PRELU","RELU"],
            [True, False]
        )
        
        return conf

    @staticmethod
    def fromArgs():
        parser = create_arg_parser()
        conf= VAETrainConfig(
            max_epochs=parser.max_epochs,
            learning_rate=parser.learning_rate,
            batch_train=parser.batch_train if not parser.test else 2,
            is_test=parser.test,
            exclude_qc=parser.exclude_qc,
            project_dataset=parser.project_dataset,
            modality=MRIModality.T1w,
            use_decoder=parser.use_decoder,
            beta=parser.beta
        )
        
        return conf
        

    def __str__(self):
        return f"max_epochs: {self.max_epochs}, learning_rate : {self.learning_rate},\
          beta:{self.beta}, batch_train: {self.batch_size}, is_test: {self.is_test}, exclude_qc:{self.exclude_qc}"

    def to_comet(self):
        self.experiment.log_parameters(
            {
                "max_epochs": self.max_epochs,
                "learning_rate": self.learning_rate,
                "batch_train": self.batch_size,
                "im_shape": self.im_shape,
                "is_test": self.is_test,
                "exclude_qc": self.exclude_qc,
                "project_dataset": self.project_dataset,
                "Conv Kernel": self.conv_k,
                "Activation": self.act,
                "Sup layer": self.simple,
                "folder":self.run_dir
            }
        )
        logging.info("params logged !")
