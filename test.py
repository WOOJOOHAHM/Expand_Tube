import os
import wandb
import argparse
import sys
sys.path.append("/home/lab/hahmwj/Expand_Tube/models")
from Backbone_models import build_model
import matplotlib.pyplot as plt
import seaborn as sns
from util import *

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base Image model') # RN50, RN101, ViT-B/16, ViT-L/14
    parser.add_argument('--dataset', type=str, help='Train dataset')
    parser.add_argument('--classifier', type=str, default=True, help='Aggregation classifiers')
    parser.add_argument('--num_frames', type=int, default=True, help='Aggregation classifiers')

    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    classifier = args.classifier
    num_frames = args.num_frames
    model_path = model_name.replace('/', '-')

    logger = WandbLogger(project=f"Testing", name = f'{model_path}_{classifier}_{dataset}_{num_frames}')
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]

    data_csv = pd.read_csv(f'/home/lab/hahmwj/data/csv_files/{dataset}.csv')
    labels = list(set(data_csv['label']))
    trainer = pl.Trainer(
        max_epochs=100,
        fast_dev_run=False,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices = [0, 1, 2, 3],
        strategy='ddp_find_unused_parameters_true'
    )

    print(f'-----------------------------------------------------------------Creating dataset   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    
    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(dataset_name = dataset, 
            path = '/home/lab/hahmwj/data/csv_files/', 
            batch_size = 32,
            num_workers = 16,
            num_frames = num_frames,
            video_size=224)

    print(f'-----------------------------------------------------------------Creating model   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    model = expand_tube_lightening(model_name = model_name,
            pre_trained_model_save_path = '/home/lab/hahmwj/Expand_Tube/pre_trained_models/',
            num_classes = num_classes,
            classifier = classifier,
            warmup_steps = 3,
            num_frames = num_frames)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    predictions = trainer.predict(model, dataloaders=dataloader_test)
    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])

    print("accuracy:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes))
    print("accuracy_top5:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5))
    print("auroc:", auroc(y_prob, y, task="multiclass", num_classes=num_classes))
    print("f1_score:", f1_score(y_prob, y, task="multiclass", num_classes=num_classes))

    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)

    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("output.png", dpi=300)
if __name__ == '__main__': main()