import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torchmetrics.functional as FM
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import argparse
from utils.ymlParser import parse_yml
from data.reader import prepre_geo_data
from models.model import NormalCNN1, NormalCNN3, NormalLSTM
from data.torch_data import get_val_data

def evaluate_model(model, dataloader, n_classes, device):
    model = model.eval()
    y_true, y_pred = None, None
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(x)
            prediction = logits.argmax(dim=-1, keepdim=True)

        if y_true == None or y_pred == None:
            y_true = y
            y_pred = prediction
        else:
            y_true = torch.cat((y_true, y))
            y_pred = torch.cat((y_pred, prediction))

    wa = FM.accuracy(y_pred, y_true)
    cm = FM.confusion_matrix(y_pred, y_true, normalize='true', num_classes=n_classes)
    ua = torch.diag(cm).mean()
    print(classification_report(y_true.numpy().flatten(), y_pred.numpy().flatten()))
    return wa, ua, cm

def main(args=None):
    """
    evaluate pipeline for pytorch models.
    """
    # argument parser
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    # get data
    val_dataloader = get_val_data(config.batch_size, config.val_data_root, config.image_size, config.in_memory, config.feature_list)

    # init configurable parameter
    model_name = config.model_name

    # load model
    device = torch.device(config.device)
    model = torch.load(model_name).to(device)
    wa, ua, cm = evaluate_model(model, val_dataloader, n_classes=4, device=device)

    labels = ["cassava", "rice", "maize", "sugarcane"]

    cm_df = pd.DataFrame(cm.to("cpu").numpy(), index=labels, columns=labels)

    sns.heatmap(cm_df, annot=True)
    print("Save confusion matrix to: {}".format(model_name.replace(".pt", ".png")))
    plt.savefig(model_name.replace(".pt", ".png"))
    plt.show()

if __name__ == '__main__':
    main()