import wandb
import json
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params
from utils import final_metrics, download_data, get_data, get_df, log_predictions, display_diagnostics, check_data_partition


def count_by_class(arr, cidxs): 
    return [np.sum(np.where(arr == n)) for n in cidxs]

def log_hist(c):
    _, bins, _ = plt.hist(target_counts[c],  bins=2, alpha=0.5, density=True, label='target')
    _ = plt.hist(pred_counts[c], bins=bins, alpha=0.5, density=True, label='pred')
    plt.legend(loc='upper right')
    plt.title(params.CLASSES[c])
    img_path = f'hist_val_{params.CLASSES[c]}'
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})


run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="evaluation", tags=['staging'])

# Get model registry
with open("secrets.json") as f:
    data = json.load(f)
    registry_link = data["model_registry"]

# Get model artifact and path
artifact = run.use_artifact(registry_link, type='model')
artifact_dir = Path(artifact.download())
_model_pth = artifact_dir.ls()[0]
model_path = _model_pth.parent.absolute()/_model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config

processed_dataset_dir = download_data()
test_valid_df = get_df(processed_dataset_dir, is_test=True)
test_valid_dls = get_data(processed_dataset_dir, test_valid_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

# Check distribution of classes 
check_data_partition(test_valid_df, f'Class dist test')

metrics = [F1Score(), BalancedAccuracy()]

cbs = [
    SaveModelCallback(monitor='f1_score'),
    WandbCallback(log_preds=False, log_model=True)
]
cbs += ([MixedPrecision()] if config.mixed_precision else [])

learn = vision_learner(test_valid_dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                        metrics=metrics)

learn.load(model_path)

val_metrics = learn.validate(ds_idx=1)
test_metrics = learn.validate(ds_idx=0)

val_metric_names = ['val_loss'] + [f'val_{x.name}' for x in learn.metrics]
val_results = {val_metric_names[i] : val_metrics[i] for i in range(len(val_metric_names))}
for k,v in val_results.items(): 
    wandb.summary[k] = v

test_metric_names = ['test_loss'] + [f'test_{x.name}' for x in learn.metrics]
test_results = {test_metric_names[i] : test_metrics[i] for i in range(len(test_metric_names))}
for k,v in test_results.items(): 
    wandb.summary[k] = v
    
log_predictions(learn)

val_probs, val_targs = learn.get_preds(ds_idx=1)
val_preds = val_probs.argmax(dim=1)
class_idxs = params.CLASSES.keys()

target_counts = count_by_class(val_targs, class_idxs)
pred_counts = count_by_class(val_preds, class_idxs)

for c in class_idxs:
    log_hist(c)
    
val_count_df, val_disp = display_diagnostics(learner=learn, ds_idx=1, return_vals=True)
wandb.log({'val_confusion_matrix': val_disp.figure_})
val_ct_table = wandb.Table(dataframe=val_count_df)
wandb.log({'val_count_table': val_ct_table})

test_count_df, test_disp = display_diagnostics(learner=learn, ds_idx=0, return_vals=True)
wandb.log({'test_confusion_matrix': test_disp.figure_})
test_ct_table = wandb.Table(dataframe=test_count_df)
wandb.log({'test_count_table': test_ct_table})

final_metrics(learn)

run.finish()
