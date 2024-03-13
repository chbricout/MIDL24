import os
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from torchvision.transforms import RandomErasing
import lightning

from neuro_ix.utils.log import save_volume_as_gif

class ConvModule(nn.Module):
    def __init__(
        self,
        conv_kernel,
        in_channel,
        out_channel,
        stride,
        padding=None,
        act="PRELU",
    ):
        super().__init__()
        self.key = f"{in_channel}-{out_channel}"
        if padding == None:
            padding = conv_kernel // 2

        self.conv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            padding=padding,
            strides=stride,
            norm="BATCH",
            act=act,
        )
        self.conv_mid = Convolution(
            spatial_dims=3,
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            strides=1,
            padding=conv_kernel // 2,
            norm="BATCH",
            act=act,
        )

    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_mid(y)

        return y


class DeConvModule(nn.Module):
    def __init__(
        self,
        conv_kernel,
        in_channel,
        out_channel,
        stride=2,
        padding=None,
        act="PRELU",
    ):
        super().__init__()
        if padding == None:
            padding = conv_kernel // 2


        self.deconv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            strides=stride,
            padding=padding,
            is_transposed=True,
            norm="BATCH",
            act=act,
        )
        self.deconv_mid = Convolution(
            spatial_dims=3,
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            norm="BATCH",
            strides=1,
            padding=conv_kernel // 2,
            act=act,
        )

    def forward(self, x):
        y = self.deconv_in(x)
        y = self.deconv_mid(y)
        return y



class AugBaselineModel(lightning.LightningModule):
   

    def __init__(
        self,
        in_channel,
        im_shape,
        act="PRELU",
        kernel_size=5,
        run_name="",
        lr=1e-4,
        beta=0.1,
        use_decoder=True

    ):
        super().__init__()

        self.im_shape = im_shape
        self.lr=lr
        self.beta=beta
        self.use_decoder = use_decoder
        self.run_name = run_name

        self.encoder = nn.Sequential(
            ConvModule(kernel_size, in_channel, 32,  2, act=act),
            ConvModule(kernel_size, 32, 64,  2, act=act),
            ConvModule(kernel_size, 64, 128, 2, act=act),
            ConvModule(kernel_size, 128, 256,  2, act=act),
            ConvModule(kernel_size, 256, 512,  2, act=act),
            Convolution(3,512,3,kernel_size=kernel_size, padding=kernel_size//2, norm="BATCH"),
        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like))
        self.latent_size = self.out_encoder.numel()
        print(self.latent_size)
        if self.use_decoder :
            self.decoder = nn.Sequential(
                Convolution(3,3,512,kernel_size=kernel_size, padding=kernel_size//2, norm="BATCH"),
                DeConvModule(kernel_size, 512, 256,  act=act),
                DeConvModule(kernel_size, 256, 128,  act=act),
                DeConvModule(kernel_size, 128, 64,  act=act),
                DeConvModule(kernel_size, 64, 32,  act=act),
                DeConvModule(kernel_size, 32, in_channel,act=act),
            )
        else :
            self.lr = self.lr*self.beta ## mimick the scaled loss in the learning rate

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),

            nn.Linear(self.latent_size, 450),
            nn.BatchNorm1d(450, affine=False),
            nn.Dropout(0.5),
            nn.PReLU(),

            nn.Linear(450, 450),
            nn.BatchNorm1d(450, affine=False),
            nn.PReLU(),
            nn.Dropout(0.5),

            nn.Linear(450, 128),
            nn.BatchNorm1d(128, affine=False),
            nn.PReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 3)
        )


        self.recon_to_plot = None
        self.test_to_plot = None

        self.label=[]
        self.classe=[]
        self.save_hyperparameters()

    def encode_forward(self, input):
        z =self.encoder(input)
        return z

    def decode_forward(self, z):
        res = self.decoder(z)
        return res

    def classify_emb(self, z):
        return self.classifier(torch.flatten(z, start_dim=1))

    def forward(self, x):
        z = self.encode_forward(x)
        if self.use_decoder :
            recon = self.decode_forward(z)
        else :
            recon = x
        classe = self.classify_emb(z)
        return [recon, z, classe]

    def training_step(self, batch, batch_idx):
        volume, label = batch
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = torch.nn.functional.mse_loss(recon_batch, volume)
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot = recon_loss  + self.beta * label_loss
            self.log("train_recon_loss", recon_loss)
        else: 
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot =  label_loss
        self.log("train_loss", model_loss_tot)
        self.log("train_label_loss", label_loss)

        return model_loss_tot

    def validation_step(self, batch, batch_idx):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume, label = batch
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = torch.nn.functional.mse_loss(recon_batch, volume)
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot = recon_loss  + self.beta * label_loss
            self.log("val_recon_loss", recon_loss)
        else:
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot =  label_loss
        self.log("val_loss", model_loss_tot)
        self.log("val_label_loss", label_loss)

        self.recon_to_plot = recon_batch[0][0].cpu()
        self.test_to_plot = volume[0][0].cpu()
        self.label += label.cpu().tolist()
        self.classe  += classe.cpu().tolist()
        return model_loss_tot


    def on_validation_epoch_end(self) -> None:
        self.logger.experiment.log_confusion_matrix(self.label, self.classe, epoch=self.current_epoch)
        classe = torch.Tensor(self.classe)
        lab =  torch.Tensor(self.label)
        accuracy= (classe.argmax(dim=1) == lab).sum()/(lab.numel())
        self.label=[]
        self.classe=[]
        self.log("val_accuracy", accuracy.mean())
        self.plot_recon()
        self.plot_test()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim

    def plot_recon(self):
        path = f"runs/{self.run_name}/recon-{self.current_epoch}.gif"
        save_volume_as_gif(self.recon_to_plot, path)
        self.logger.experiment.log_image(
            path, name="reconstruction", image_format="gif", step=self.current_epoch
        )
        os.remove(path)

    def plot_test(self):
        path = f"runs/{self.run_name}/test-{self.current_epoch}.gif"
        save_volume_as_gif(self.test_to_plot, path)
        self.logger.experiment.log_image(
            path, name="test", image_format="gif", step=self.current_epoch
        )
        os.remove(path)
