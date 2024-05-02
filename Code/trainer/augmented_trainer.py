import logging
import sys

sys.path.append(".")
import comet_ml
from monai.data import CacheDataset, DataLoader, Dataset, ZipDataset
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
from neuro_ix.models.augmented_cnn import AugBaselineModel
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def seek_nan_batchnorm(comet_logger:lightning.pytorch.loggers.CometLogger,m):

    find_nan = comet_logger.experiment.get_parameter("find_nan")
    if isinstance(m, nn.BatchNorm3d):
        if m.running_mean.isnan().any():
            print(f"{m} : running mean contains nans")
            m.running_mean.fill_(0)
            find_nan=True
        if m.running_var.isnan().any():
            print(f"{m} : running var contains nans")
            m.running_var.fill_(1)
            find_nan=True

        if not  m.running_mean.isnan().any() and not m.running_var.isnan().any():
            print(f"{m} : no nans")
    comet_logger.log_hyperparams({"find_nan": find_nan})


def launch_train(config):
    name = "Augmented"
    if not config.use_decoder:
        name += "-NoDec"
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name="nan-investigate-midl2024",
        experiment_name=f"{name}-beta{config.beta}-lr{config.learning_rate}",
    )
    comet_logger.log_hyperparams({"find_nan": False})

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

    aug_net = AugBaselineModel(
        1,
        config.im_shape,
        act=config.act,
        kernel_size=config.conv_k,
        run_name=config.run_name,
        lr=config.learning_rate,
        beta=config.beta,
        use_decoder=config.use_decoder,
    )
    print("-----------------------------")
    print("Pre Init")
    aug_net.apply(lambda x: seek_nan_batchnorm(comet_logger,x))
    print("-----------------------------")
    
    if comet_logger.experiment.get_parameter("find_nan"):
        aug_net.apply(init_weights)

    
    
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
        print("Logging weight")
        log_model(
            comet_logger.experiment,
            aug_net,
            "Pretrain",
        )
        trainer.fit(aug_net, train_dataloaders=train_loader, val_dataloaders=val_loader)

        log_model(
            comet_logger.experiment,
            AugBaselineModel.load_from_checkpoint(check.best_model_path),
            name,
        )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    config = VAETrainConfig.fromArgs()
    logging.info(str(config))
    torch.set_float32_matmul_precision("high")
    launch_train(config)
