import logging
import sys

sys.path.append(".")
import comet_ml
from monai.data import CacheDataset, DataLoader, ZipDataset, Dataset
from monai.transforms import (
    RandFlip,
    RandRotate,
    Compose,
)
import torch
import torch.nn as nn
import lightning
from comet_ml.integration.pytorch import log_model

from neuro_ix.vae.vae_config import VAETrainConfig
from neuro_ix.models.baseline import BaselineModel
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def launch_train(config):
    train_ds = CacheDataset(
        data=config.train_path, transform=config.transform, num_workers=10
    )
    flipped_train_ds = Dataset(
        data=train_ds,
        transform=Compose(
            [
                RandFlip(prob=0.5, spatial_axis=0),
                RandRotate(
                    range_x=[0.0, 0.2],
                    range_y=[0.0, 0.2],
                    range_z=[0.0, 0.2],
                    prob=0.7,
                    keep_size=True,
                ),
            ]
        ),
    )
    train_loader = DataLoader(
        ZipDataset([flipped_train_ds, config.train_scores]),
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_ds = CacheDataset(
        data=config.val_path, transform=config.transform, num_workers=10
    )
    val_loader = DataLoader(
        ZipDataset([val_ds, config.val_scores]), batch_size=config.batch_size
    )
    logging.info(f"Dataset contain {len(train_ds)} datas")

    base_enc = BaselineModel(
        1,
        config.im_shape,
        act=config.act,
        kernel_size=config.conv_k,
        run_name=config.run_name,
        lr=config.learning_rate,
        use_decoder=config.use_decoder,
    )
    base_enc.apply(init_weights)

    name = "Baseline"
    if not config.use_decoder:
        name += "-NoDec"
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name="midl2024",
        experiment_name=f"{name}-lr{config.learning_rate}",
    )
    check = ModelCheckpoint(monitor="val_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        default_root_dir=config.run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_label_loss", mode="min", patience=50),
            check,
        ],
    )
    trainer.fit(base_enc, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log_model(
        comet_logger.experiment,
        BaselineModel.load_from_checkpoint(check.best_model_path),
        name,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    config = VAETrainConfig.fromArgs()
    logging.info(str(config))
    torch.set_float32_matmul_precision("high")
    launch_train(config)
