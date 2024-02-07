import os
import wandb
import argparse
import sys
sys.path.append("/home/lab/hahmwj/Expand_Tube/models")
from Backbone_models import build_model
from util import *

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base Image model') # R50, R101, ViT-B/16, ViT-L/14
    parser.add_argument('--dataset', type=str, help='Train dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs.')
    parser.add_argument('--fast_dev_run', action='store_true', default=False, help='True: Training without error')
    parser.add_argument('--classifier', type=str, default=True, help='Aggregation classifiers')

    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    fast_dev_run = args.fast_dev_run
    classifier = args.classifier
       
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    model_path = model_name.replace('/', '-')

    logger = WandbLogger(project=f"Expand_Tube", name = f'{model_path}_{classifier}')
    trainer = pl.Trainer(
        max_epochs=epochs,
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices = [0, 1, 2, 3],
        strategy='ddp_find_unused_parameters_true'
    )

    print(f'-----------------------------------------------------------------Creating dataset   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    
    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(dataset_name = dataset, 
            path = '/home/lab/hahmwj/data/csv_files/', 
            batch_size = batch_size,
            num_workers = 16,
            num_frames = 8,
            video_size=224)

    print(f'-----------------------------------------------------------------Creating model   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    model = expand_tube_lightening(model_name = model_name,
            pre_trained_model_save_path = '/home/lab/hahmwj/Expand_Tube/pre_trained_models/',
            num_classes = num_classes,
            max_epochs = epochs,
            classifier = classifier,
            warmup_steps = 3,
            num_frames = 8)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.save_checkpoint(f"//home/lab/hahmwj/Expand_Tube/weights/{model_path}/{classifier}.ckpt")
if __name__ == '__main__': main()