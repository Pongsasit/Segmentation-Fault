import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import glob
from tqdm import tqdm
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import argparse
from utils.ymlParser import parse_yml

def main(args=None):
    """
    training pipeline for pytorch models.
    """
    # argument parser
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    train_size = config.train_size
    val_size = config.val_size
    train_number = config.train_number
    train_path = config.train_path
    val_path = config.val_path
    if not os.path.exists(val_path + "/2021/"):
        os.makedirs(val_path + "/2021/")
    seed = config.seed

    if train_size + val_size != 100:
        raise "train_size + val_size must equal 100"

    
    order_list = [i for i in range(train_number)]
    train, val = train_test_split(order_list, test_size=val_size/100, random_state=seed)
    for i in val:
        folders = glob.glob(os.path.join(train_path, "2021", "*"))
        for folder in folders:
            if not os.path.exists(os.path.join(val_path, "2021", folder.split("/")[-1], str(i))):
                os.makedirs(os.path.join(val_path, "2021", folder.split("/")[-1], str(i)))
            os.system("mv {} {} && rm -rf {}".format(os.path.join(folder, str(i), "*.tif"), os.path.join(val_path, "2021", folder.split("/")[-1], str(i)), os.path.join(folder, str(i))))


if __name__ == '__main__':
    main()