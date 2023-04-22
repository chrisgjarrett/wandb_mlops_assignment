## This script comes from 04_refactor_baseline.ipynb

import argparse, os
import wandb
from pathlib import Path
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from utils import get_predictions, create_predictions_table
from sklearn.metrics import f1_score, balanced_accuracy_score 
import params
from utils import final_metrics, t_or_f, get_df, download_data, get_data, log_predictions

# defaults
default_config = SimpleNamespace(
    framework="fastai",
    img_size=180, #(180, 320) in 16:9 proportions,
    batch_size=8, #8 keep small in Colab to be manageable
    augment=True, # use data augmentation
    epochs=10, # for brevity, increase for better results :)
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=True, # use automatic mixed precision
    arch="resnet18",
    seed=42,
    log_preds=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=default_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def train(config):
    set_seed(config.seed)
    
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config)        
    
    # good practice to inject params using sweeps
    config = wandb.config

    # prepare data
    processed_dataset_dir = download_data()
    proc_df = get_df(processed_dataset_dir)
    dls = get_data(processed_dataset_dir, proc_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    metrics = [F1Score(), BalancedAccuracy()]

    cbs = [
        SaveModelCallback(monitor='f1_score'),
        WandbCallback(log_preds=False, log_model=True)
    ]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    learn = vision_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                         metrics=metrics)

    learn.fit_one_cycle(config.epochs, config.lr, cbs=cbs)

    if config.log_preds:
        log_predictions(learn)
    final_metrics(learn)
    
    wandb.finish()


if __name__ == '__main__':
    parse_args()
    train(default_config)


