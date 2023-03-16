from fastai.vision.all import *
import wandb
import params
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display, Markdown
import seaborn as sns

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
        df = df[df.Stage != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
        # when passed to datablock, this will return test at index 0 and valid at index 1
          
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
    

def display_diagnostics(learner, ds_idx=1, return_vals=False):
    """
    Display a confusion matrix for the unet learner.
    If `dls` is None it will get the validation set from the Learner
    
    You can create a test dataloader using the `test_dl()` method like so:
    >> dls = ... # You usually create this from the DataBlocks api, in this library it is get_data()
    >> tdls = dls.test_dl(test_dataframe, with_labels=True)
    
    See: https://docs.fast.ai/tutorial.pets.html#adding-a-test-dataloader-for-inference
    
    """
    probs, targs = learner.get_preds(ds_idx=ds_idx)
    preds = probs.argmax(dim=1)
    classes = list(params.CLASSES.values())
    y_true = targs.flatten().numpy()
    y_pred = preds.flatten().numpy()
    
    tdf, pdf = [pd.DataFrame(r).value_counts().to_frame(c) for r,c in zip((y_true, y_pred) , ['y_true', 'y_pred'])]
    countdf = tdf.join(pdf, how='outer').reset_index(drop=True).fillna(0).astype(int).rename(index= params.CLASSES)
    countdf = countdf/countdf.sum() 
    display(Markdown('### % Of Pixels In Each Class'))
    display(countdf.style.format('{:.1%}'))
    
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=classes,
                                                   normalize='pred')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10) 
    disp.ax_.set_title('Confusion Matrix', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    fig.show()
    fig.autofmt_xdate(rotation=45)

    if return_vals: return countdf, disp


def check_data_partition(df, img_path):
    
    (df
    .groupby("Stage")["Class"]
    .value_counts(normalize=True)
    .mul(100)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot,'data'), x="Stage",y='percent',hue="Class",kind='bar'))

    # Log image to wandb
    plt.ylabel("Percentage of dataset")
    plt.title("Class distribution across stage")
    plt.ylabel("Percentage of dataset")
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})