from fastai.vision.all import *
import wandb


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
