import os
import argparse
import sys
sys.path.append("/hahmwj/expand_tube/models")
from Backbone_models import build_model
from util import *

import lightning.pytorch as pl
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from lightning.pytorch.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base Image model') # R50, R101, ViT-B/16, ViT-L/14
    parser.add_argument('--dataset', type=str, help='Train dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs.')
    parser.add_argument('--fast_dev_run', type=bool, default=True, help='True: Training without error')

    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    fast_dev_run = args.fast_dev_run
    
    
    print('-----------------------------------------------------------------Creating dataset-----------------------------------------------------------------')
    
    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(dataset_name = dataset, 
            path = '/hahmwj/csv_files/', 
            batch_size = batch_size,
            num_workers = 16,
            num_frames = 8,
            video_size=224)

    print('-----------------------------------------------------------------Creating model-----------------------------------------------------------------')
    model = expand_tube_lightening(model_name = model_name,
            pre_trained_model_save_path = '/hahmwj/expand_tube/weights/pre_trained_models/',
            num_classes = num_classes,
            max_epochs = epochs)


    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')     
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch"), early_stopping]
    logger = TensorBoardLogger("logs", name="Expand_Yube")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.save_checkpoint(f"/hahmwj/expand_tube/weights/Expand_model/{model_name}.ckpt")
if __name__ == '__main__': main()