from fastai.vision.all import *
import wandb
import params

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False


def get_predictions(learner, test_dl=None, max_n=None):
    """Return the samples = (x,y) and outputs (model predictions decoded), and predictions (raw preds)"""
    test_dl = learner.dls.valid if test_dl is None else test_dl
    inputs, predictions, targets, outputs = learner.get_preds(
        dl=test_dl, with_input=True, with_decoded=True
    )
    # x, y, samples, outputs = learner.dls.valid.show_results(
    #     tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=max_n
    # )
    return inputs, outputs


def create_predictions_table(samples, predictions):
    "Creates a wandb table with predictions and targets side by side"

    items = list(zip(samples, predictions))
    
    table = wandb.Table(
        columns=["Image", "Label"])
    
    # we create one row per sample
    for item in progress_bar(items):
        image, label = item
        image = image.permute(1, 2, 0)
        table.add_data(
            wandb.Image(image),
            label,
    )
    
    return table


def download_data():
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir


def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')

    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage == 'test'].reset_index(drop=True)
          
    # assign paths
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.Filename.values]    
    
    return df


def get_data(processed_dataset_dir, df:pd.DataFrame, bs=1, img_size=(180, 320), augment=True):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=ColReader(0, pref=processed_dataset_dir/"images"),
                  get_y=ColReader("Class"),
                  splitter=ColSplitter(),
                  item_tfms=Resize(img_size),
                #   batch_tfms=aug_transforms() if augment else None,                 )
            )
    return block.dataloaders(df, bs=bs)


def log_predictions(learn):
    "Log a Table with model predictions and metrics"
    samples, predictions = get_predictions(learn)
    table = create_predictions_table(samples, predictions)
    wandb.log({"pred_table":table})
    
