import pandas as pd
import lightning.pytorch as pl
from typing import Any, Callable, List, Union

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment

import sys
sys.path.append("/hahmwj/expand_tube/models")
from Backbone_models import build_model
from models.dataset import VideoDataset

def load_data(dataset_name: str, 
            path: str, 
            batch_size: int = 32,
            num_workers: int = 16,
            num_frames: int = 8,
            video_size: int = 224
                ):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            ToTensorVideo(),  # C, T, H, W
            Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
            RandAugment(magnitude=10, num_layers=2),
            Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
            T.Resize(size=(video_size, video_size)),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=(video_size, video_size)),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    if dataset_name == 'hmdb_sample':
        dataframe = pd.read_csv(f'{path}/hmdb_sample.csv')
        num_classes = 51
    elif dataset_name == 'hmdb':
        dataframe = pd.read_csv(f'{path}/hmdb.csv')
        num_classes = 51
    elif dataset_name == 'ucf_sample':
        dataframe = pd.read_csv(f'{path}/ucf_sample.csv')
        num_classes = 101
    elif dataset_name == 'ucf':
        dataframe = pd.read_csv(f'{path}/ucf.csv')
        num_classes = 101
    elif dataset_name == 'k400_sample':
        dataframe = pd.read_csv(f'{path}/k400_sample.csv')
        num_classes = 400
    elif dataset_name == 'k400':
        dataframe = pd.read_csv(f'{path}/k400.csv')
        num_classes = 400
    elif dataset_name == 'ssv2_sample':
        dataframe = pd.read_csv(f'{path}/ssv2_sample.csv')
        num_classes = 174
    elif dataset_name == 'ssv2':
        dataframe = pd.read_csv(f'{path}/ssv2.csv')
        num_classes = 174

    dataset_train = VideoDataset(
        dataframe,
        num_frames,
        'train',
        train_transform
    )
    dataset_val = VideoDataset(
        dataframe,
        num_frames,
        'valid',
        test_transform
    )

    dataset_test = VideoDataset(
        dataframe,
        num_frames,
        'test',
        test_transform
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        # sampler=torch.utils.data.DistributedSampler(dataset_train),
        num_workers=num_workers,
        pin_memory=True,
        )
    dataloader_val = torch.utils.data.DataLoader(
        # torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        )
    
    dataloader_test = torch.utils.data.DataLoader(
        # torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        )
    
    return dataloader_train, dataloader_val, dataloader_test, num_classes
    
class expand_tube_lightening(pl.LightningModule):
    def __init__(
            self,
            model_name,
            pre_trained_model_save_path,
            num_classes,
            lr: float = 3e-4,
            weight_decay: float = 0,
            max_epochs: int = None,
            **kwargs,
    ):
        self.save_hyperparameters()
        
        super().__init__()
        self.model_name = model_name
        self.pre_trained_model_save_path = pre_trained_model_save_path
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.model = build_model(self.model_name, self.pre_trained_model_save_path)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("train_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("val_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        y_pred = torch.softmax(y_hat, dim=-1)

        return {"y": y, "y_pred": torch.argmax(y_pred, dim=-1), "y_prob": y_pred}