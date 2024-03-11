import logging
import os
from typing import List
import pandas as pd
import torch
import monai
from neuro_ix.datasets.mriqc_dataset import MRIQCDataset
from neuro_ix.utils.datasets import zip_dataset
from neuro_ix.utils.log import save_volume_as_gif
from neuro_ix.vae.vae_config import VAETrainConfig
from monai.data import DataLoader
from monai.transforms import SaveImage
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric


def threshold_at_one(x):
    # threshold at 1
    return x >= 1


def plot_diff(img, dataset, num, run_name, experiment):
    path = f"runs/{run_name}/{dataset}-{num}.gif"
    save_volume_as_gif(img, path)
    experiment.log_image(path, name=f"diff-{dataset}", image_format="gif", step=num)
    os.remove(path)



class DataProjector:
    def __init__(
        self, vae_model, config: VAETrainConfig, path_to_filter: List[str] = None
    ):
        self.config = config
        self.vae_model = vae_model.to(config.device)
        self.mse_loss = torch.nn.MSELoss(reduction="none").to(config.device)
        self.transform_normal = config.transform

        self.qc_dataset = (
            MRIQCDataset.test() if config.is_test else MRIQCDataset.lab()
        )
        no_issue_path = self.qc_dataset.get_images_path(
            modality=self.config.modality, qc_issue=False
        )
        before = len(no_issue_path)
        if not path_to_filter is None:
            no_issue_path = list(set(no_issue_path).intersection(path_to_filter))
        logging.info(f"removed : {before - len(no_issue_path)}")
        self.dataset = zip_dataset(no_issue_path, transform=self.transform_normal)

        issue_path = self.qc_dataset.get_images_path(
            modality=self.config.modality, qc_issue=True
        )
        self.dataset_qc = zip_dataset(
            issue_path,
            transform=self.transform_normal,
        )

    def get_dataloader(self, dataset):
        return DataLoader(
            list(dataset),
            batch_size=self.config.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    def project(self, save_volumes=False):
        save_recon = SaveImage(output_dir=self.config.run_dir, output_postfix="recon")
        save_diff = SaveImage(output_dir=self.config.run_dir, output_postfix="diff")
        save_orig = SaveImage(output_dir=self.config.run_dir, output_postfix="orig")
        rows = []
        header = [
            "id",
            "qc_issue",
            "dataset",
            "score",
            "MSE loss",
            "sum_of_diff",
            "PSNR",
            "SSIM",
            *[f"X{i}" for i in range(self.vae_model.latent_size)],
        ]
        ssim_metric = SSIMMetric(spatial_dims=3, reduction="none")
        psnr_metric = PSNRMetric(1, reduction="none")
        with torch.no_grad():
            logging.info("starting projection of Dataset")
            num_by_score = {"normal": 0, "qc": 0}
            for name, dataset in zip(
                ["normal", "qc"],
                [self.dataset, self.dataset_qc],
            ):
                i = 0
                for path, img in self.get_dataloader(dataset):
                    logging.info(f"batch {i}  ")
                    i += 1
                    deviced_input = img.to(self.config.device)
                    mu, logvar = self.vae_model.encode_forward(deviced_input)
                    encode = self.vae_model.reparameterize(mu, logvar)
                    decode = self.vae_model.decode_forward(encode)
                    difference = torch.abs(decode - deviced_input)
                    psnr = psnr_metric(decode, deviced_input)
                    ssim = ssim_metric(decode, deviced_input)
                    loss = self.mse_loss(decode, deviced_input).mean(axis=[1, 2, 3, 4])
                    conc = [
                        [p]
                        + [name]
                        + [dataset_name]
                        + [dataset_score]
                        + [l]
                        + [d]
                        + ss
                        + ps
                        + e.tolist()
                        for p, e, dataset_name, dataset_score, d, l, ss, ps in zip(
                            path,
                            encode,
                            map(self.qc_dataset.get_dataset, path),
                            map(self.qc_dataset.dataset_score, path),
                            difference.sum(dim=(1, 2, 3, 4)).tolist(),
                            loss.tolist(),
                            ssim.tolist(),
                            psnr.tolist(),
                        )
                    ]
                    rows += conc
                    if (name == "normal" and num_by_score[name] < 100) or name == "qc" or save_volumes:
                        for p,dec, im in zip(path,decode, deviced_input):
                            diff = torch.abs(dec - im)
                            plot_diff(
                                diff[0],
                                dataset=name,
                                num=num_by_score[name],
                                run_name=self.config.run_name,
                                experiment=self.config.experiment,
                            )
                            if save_volumes:
                                idx = self.qc_dataset.get_file_id(p)
                                im.meta["filename_or_obj"]=idx
                                dec.meta["filename_or_obj"]=idx
                                diff.meta["filename_or_obj"]=idx

                                save_orig(im)
                                save_recon(dec)
                                save_diff(diff)
                            num_by_score[name] += 1
                logging.info(f"projected all dataset : {name}")

        data_to_store = pd.DataFrame(rows, columns=header)
        self.config.experiment.log_dataframe_profile(
            data_to_store,
            name="dataset_projection",
            log_raw_dataframe=True,
            dataframe_format="csv",
        )
        logging.info(f"Data succesfully stored ! ({len(data_to_store)} rows)")
