import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import pandas as pd

import argparse
from utils.ymlParser import parse_yml
from data.reader import prepre_geo_data
from models.model import NormalCNN1, NormalCNN3, NormalLSTM
from data.torch_data import get_test_data

def predict(model, dataloader, device):
    model = model.eval()
    y_pred = None
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(x)
            prediction = logits.argmax(dim=-1, keepdim=True)

        if y_pred == None:
            y_pred = prediction
        else:
            y_pred = torch.cat((y_pred, prediction))

    return y_pred

def main(args=None):
    """
    predict pipeline for pytorch models.
    """
    # argument parser
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    # get data
    test_dataloader = get_test_data(config.batch_size, config.val_data_root, config.image_size, config.in_memory, config.feature_list)

    # init configurable parameter
    model_name = config.model_name

    # load model
    device = torch.device(config.device)
    model = torch.load(model_name).to(device)
    y_pred = predict(model, test_dataloader, device=device)

    df = pd.DataFrame({"crop_type": y_pred.numpy().flatten()})
    df.to_csv("submit_result.csv")

    print("Export submit result to: submit_result.csv")

if __name__ == '__main__':
    main()