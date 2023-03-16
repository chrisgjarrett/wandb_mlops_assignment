import wandb
from fastai.vision.all import *
import params
from utils import get_df, download_data, get_data, check_data_partition
import seaborn as sns

run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="partition_check")        

# good practice to inject params using sweeps
config = wandb.config

# prepare data
processed_dataset_dir = download_data()
proc_df = get_df(processed_dataset_dir)

print(proc_df.head())

# Plot proportion of test and train
check_data_partition(proc_df, f'Class dist train-valid')